from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'Nonfinite {name}')


def _masked_mean(value, mask):
    mask = mask.to(device=value.device, dtype=torch.bool)
    if mask.shape != value.shape or not mask.any(-1).all():
        raise ValueError('Mask must match scores and contain tokens in every completion')
    _finite(value[mask], 'valid token scores')
    return torch.where(mask, value, 0).sum(-1) / mask.sum(-1)


def grpo_loss(current, old, reference, rewards, mask, epsilon=.2, beta=.01):
    """[G,T] or [B,G,T] scores; returns equal-group/equal-sequence scalar loss.

    Old/reference and rewards are always detached. Actual EOS belongs in mask.
    """
    if current.ndim not in (2, 3) or old.shape != current.shape or reference.shape != current.shape:
        raise ValueError('Expected matching [G,T] or [B,G,T] log probabilities')
    rewards = torch.as_tensor(rewards, device=current.device, dtype=current.dtype).detach()
    if rewards.shape != current.shape[:-1]:
        raise ValueError('Rewards must match completion dimensions')
    if epsilon < 0 or beta < 0:
        raise ValueError('epsilon and beta must be nonnegative')
    _finite(rewards, 'rewards')
    advantage = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, unbiased=False, keepdim=True) + 1e-6)
    valid = mask.to(device=current.device, dtype=torch.bool)
    # Remove padding before exponentiation, including nonfinite padding values.
    c = torch.where(valid, current, 0)
    o = torch.where(valid, old.detach(), 0)
    r = torch.where(valid, reference.detach(), 0)
    ratio = (c - o).exp()
    d = r - c
    kl = d.exp() - d - 1
    objective = -torch.minimum(ratio * advantage.unsqueeze(-1), ratio.clamp(1-epsilon, 1+epsilon) * advantage.unsqueeze(-1)) + beta * kl
    return _masked_mean(objective, valid).mean()


def sft_loss(logprobs, mask=None):
    """Mean response NLL per example, then equal example mean."""
    if mask is None:
        mask = torch.ones_like(logprobs, dtype=torch.bool)
    return -_masked_mean(logprobs, mask).mean()


@dataclass
class TrainingConfig:
    profile: str = 'smoke'
    objective: str = 'sft'
    effective_batch_size: int = 1
    microbatch_size: int = 1
    group_size: int = 4
    max_new_tokens: int = 128
    epsilon: float = .2
    beta: float = .01
    decoder_lr: float = 2e-5
    projector_lr: float = 1e-4
    weight_decay: float = .01
    max_grad_norm: float = 1.
    total_updates: int = 1
    warmup_fraction: float = .05
    warmup_updates: int = 1
    continuation_updates: int = 1
    text_updates: int = 1
    alignment_updates: int = 1
    audio_updates: int = 1
    checkpoint_every: int = 1
    overfit_cap: int = 200
    precision: str = 'fp32'

    @classmethod
    def for_profile(cls, name, objective='sft', **overrides):
        if name not in ('smoke', 'local_pilot', 'main'):
            raise ValueError(f'Unknown profile {name}')
        values = dict(profile=name, objective=objective)
        if name == 'local_pilot':
            values.update(effective_batch_size=4, total_updates=32, warmup_updates=64,
                          continuation_updates=32, text_updates=8, alignment_updates=8,
                          audio_updates=16, checkpoint_every=8)
        elif name == 'main':
            values.update(effective_batch_size=8, total_updates=512, warmup_updates=256,
                          continuation_updates=512, text_updates=256, alignment_updates=256,
                          audio_updates=512, checkpoint_every=32)
        if objective == 'grpo':
            values.update(decoder_lr=1e-6, projector_lr=5e-6)
        values.update(overrides)
        return cls(**values)


def make_optimizer(policy, config):
    decoder_ids = {id(p) for p in policy.decoder.parameters()}
    groups = [[], []]
    for p in policy.model.parameters():
        if p.requires_grad:
            groups[0 if id(p) in decoder_ids else 1].append(p)
    if not any(groups):
        raise ValueError('No trainable policy parameters')
    return torch.optim.AdamW([{'params': ps, 'lr': lr} for ps, lr in
                             zip(groups, (config.decoder_lr, config.projector_lr)) if ps],
                            weight_decay=config.weight_decay)


def capture_rng_state():
    state = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state['numpy'] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'numpy' in state:
        import numpy as np
        np.random.set_state(state['numpy'])


def save_resume_state(path, optimizer, *, scheduler=None, scaler=None, **metadata):
    state = dict(metadata, optimizer=optimizer.state_dict(), rng=capture_rng_state(),
                 scheduler=scheduler.state_dict() if scheduler else None,
                 scaler=scaler.state_dict() if scaler else None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_resume_state(path, optimizer, *, scheduler=None, scaler=None):
    # Trusted local training checkpoints contain Python/NumPy RNG objects.
    state = torch.load(path, map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    if scheduler and state['scheduler'] is not None:
        scheduler.load_state_dict(state['scheduler'])
    if scaler and state['scaler'] is not None:
        scaler.load_state_dict(state['scaler'])
    restore_rng_state(state['rng'])
    return state


def _identity(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Trainer:
    """One optimizer update per call. Scoring microbatches run sequentially.

    Caller owns training-pool selection and frozen encoder configuration. Reference
    is copied before the first update and persisted independently on checkpoint.
    """
    def __init__(self, policy, config=None, *, reward_fn=None, reference_policy=None,
                 optimizer=None, ledger_path=None, metadata=None):
        self.policy = policy
        self.config = config or TrainingConfig()
        if self.config.objective not in ('sft', 'grpo'):
            raise ValueError('objective must be sft or grpo')
        if self.config.group_size != 4 or self.config.microbatch_size < 1:
            raise ValueError('GRPO requires G=4; microbatch_size must be positive')
        if self.config.precision not in ('fp32', 'fp16', 'bf16'):
            raise ValueError('Unsupported precision')
        if self.config.precision != 'fp32' and torch.device(policy.device).type != 'cuda':
            raise ValueError('Mixed precision requires CUDA')
        if self.config.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('CUDA device does not support BF16')
        if torch.device(policy.device).type == 'cuda' and hasattr(policy, '_autocast'):
            policy_autocast = policy._autocast()
            if (getattr(policy_autocast, '_enabled', False)
                    and getattr(policy_autocast, 'fast_dtype', None) == torch.float16
                    and self.config.precision != 'fp16'):
                raise ValueError("Policy uses internal CUDA FP16 autocast; set Trainer precision='fp16' for gradient scaling")
        self.reference = None
        if self.config.objective == 'grpo':
            if reference_policy is not None:
                if {id(p) for p in reference_policy.model.parameters()} & {id(p) for p in policy.model.parameters()}:
                    raise ValueError('Reference and policy must own distinct parameters')
                self.reference = reference_policy
            else:
                self.reference = copy.deepcopy(policy)
            self.reference.model.requires_grad_(False)
            self.reference.model.eval()
        self.reference_identity = _identity(self.reference.model) if self.reference else None
        self.optimizer = optimizer or make_optimizer(policy, self.config)
        warmup = max(1, math.ceil(self.config.total_updates * self.config.warmup_fraction))
        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            return max(0., (self.config.total_updates-step) / max(1, self.config.total_updates-warmup))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=128., enabled=self.config.precision == 'fp16')
        self.reward_fn = reward_fn
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.metadata = metadata or {}
        self.step = 0
        self.examples_seen = 0
        self.tokens_seen = 0
        self.sampler_position = 0
        self.last_rollouts = []

    def _autocast(self):
        return torch.autocast(device_type=torch.device(self.policy.device).type,
                              dtype=torch.float16 if self.config.precision == 'fp16' else torch.bfloat16,
                              enabled=self.config.precision != 'fp32')

    def _score(self, policy, example, ids):
        tensor = torch.tensor([ids], dtype=torch.long, device=policy.device)
        with self._autocast():
            scores = policy.completion_logprobs(example, tensor, torch.ones_like(tensor))
        if scores.shape != tensor.shape:
            raise ValueError('completion_logprobs must return [1,T]')
        _finite(scores, 'completion log probabilities')
        return scores.float()

    def _begin(self, examples):
        if not examples:
            raise ValueError('An update needs examples')
        self.policy.model.eval()  # Disable dropout while preserving autograd.
        self.optimizer.zero_grad(set_to_none=True)
        if torch.device(self.policy.device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.policy.device)

    def _finish(self, loss, examples, tokens, started, **metrics):
        _finite(torch.tensor(loss), 'loss')
        self.scaler.unscale_(self.optimizer)
        parameters = [p for p in self.policy.model.parameters() if p.requires_grad and p.grad is not None]
        if not parameters:
            raise RuntimeError('No policy gradients')
        for p in parameters:
            _finite(p.grad, 'gradient')
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm, error_if_nonfinite=True)
        learning_rates_used = [g['lr'] for g in self.optimizer.param_groups]
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        for p in self.policy.model.parameters():
            _finite(p, 'updated parameter')
        self.step += 1
        self.examples_seen += len(examples)
        self.sampler_position += len(examples)
        self.tokens_seen += tokens
        row = dict(step=self.step, objective=self.config.objective, profile=self.config.profile,
                   loss=loss, gradient_norm=float(norm), examples=len(examples), tokens=tokens,
                   examples_seen=self.examples_seen, tokens_seen=self.tokens_seen,
                   elapsed_seconds=time.perf_counter()-started,
                   precision=self.config.precision, loss_scale=float(self.scaler.get_scale()), reference_identity=self.reference_identity,
                   learning_rates=learning_rates_used,
                   next_learning_rates=[g['lr'] for g in self.optimizer.param_groups], **metrics)
        row['peak_memory_bytes'] = torch.cuda.max_memory_allocated(self.policy.device) if torch.device(self.policy.device).type == 'cuda' else None
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
        return row

    def _diagnostic(self, error, examples):
        self.optimizer.zero_grad(set_to_none=True)
        if self.ledger_path:
            path = self.ledger_path.with_suffix('.failure.pt')
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'error': repr(error), 'step': self.step, 'examples': examples,
                        'rollouts': self.last_rollouts, 'rng': capture_rng_state()}, path)

    def sft_step(self, examples, targets=None):
        started = time.perf_counter()
        examples = list(examples)
        self._begin(examples)
        if targets is None:
            targets = [x['target_text'] for x in examples]
        if len(targets) != len(examples):
            raise ValueError('One target per example is required')
        # Adapter IDs may differ from the upstream tokenizer's pruned IDs.
        eos = getattr(self.policy, 'eos_token_id', None)
        if eos is None:
            eos = getattr(getattr(self.policy, 'tokenizer', None), 'eos_token_id', None)
        if eos is None:
            raise ValueError('Policy must expose tokenizer.eos_token_id or eos_token_id')
        total, tokens = 0., 0
        try:
            # Sequential singleton scoring is a memory-safe microbatch of one;
            # each example has exactly 1/N weight independent of its length.
            for example, target in zip(examples, targets):
                ids = list(self.policy.encode(target)) if isinstance(target, str) else list(target)
                if not ids or ids[-1] != eos:
                    ids.append(eos)
                loss = sft_loss(self._score(self.policy, example, ids)) / len(examples)
                self.scaler.scale(loss).backward()
                total += float(loss.detach())
                tokens += len(ids)
            return self._finish(total, examples, tokens, started)
        except Exception as error:
            self._diagnostic(error, examples)
            raise

    def grpo_step(self, examples):
        if self.reference is None or self.reward_fn is None:
            raise ValueError('GRPO needs a fixed reference and reward_fn')
        started = time.perf_counter()
        examples = list(examples)
        self._begin(examples)
        self.last_rollouts = []
        total, tokens, equal_groups, reward_sum, kl_sum, entropy_sum = 0., 0, 0, 0., 0., 0.
        config = dict(do_sample=True, temperature=1., top_k=0, top_p=1.,
                      max_new_tokens=self.config.max_new_tokens)
        try:
            # Collect all old scores before any optimizer mutation.
            with torch.no_grad():
                for example in examples:
                    group = []
                    for _ in range(4):
                        rollout = dict(self.policy.generate(example, config))
                        ids = list(rollout['ids'])
                        if not ids:
                            raise ValueError('Empty rollout cannot be scored')
                        reward = float(self.reward_fn(example, rollout))
                        if reward not in (0., 1.):
                            raise ValueError('Main reward must be binary')
                        # A capped response is a failure even if its text parses.
                        if rollout.get('length_capped', False):
                            reward = 0.
                        group.append(dict(rollout=rollout, ids=ids, reward=reward,
                                          old=self._score(self.policy, example, ids).detach(),
                                          reference=self._score(self.reference, example, ids).detach()))
                    self.last_rollouts.append(group)
            for example, group in zip(examples, self.last_rollouts):
                rewards = torch.tensor([r['reward'] for r in group], device=self.policy.device)
                advantages = (rewards-rewards.mean()) / (rewards.std(unbiased=False)+1e-6)
                equal_groups += int(rewards.std(unbiased=False) == 0)
                reward_sum += float(rewards.sum())
                for index, row in enumerate(group):
                    current = self._score(self.policy, example, row['ids'])
                    ratio = (current-row['old']).exp()
                    d = row['reference']-current
                    kl = d.exp()-d-1
                    terms = -torch.minimum(ratio*advantages[index], ratio.clamp(1-self.config.epsilon, 1+self.config.epsilon)*advantages[index]) + self.config.beta*kl
                    loss = terms.mean() / (len(examples)*4)
                    _finite(loss, 'GRPO loss')
                    self.scaler.scale(loss).backward()
                    total += float(loss.detach())
                    tokens += len(row['ids'])
                    kl_sum += float(kl.detach().mean())
                    entropy_sum += float(-current.detach().mean())
            return self._finish(total, examples, tokens, started,
                                reward_mean=reward_sum/(len(examples)*4),
                                equal_reward_group_rate=equal_groups/len(examples),
                                kl=kl_sum/(len(examples)*4),
                                sampled_token_surprisal=entropy_sum/(len(examples)*4),
                                reward_variance=float(torch.tensor([r['reward'] for g in self.last_rollouts for r in g]).var(unbiased=False)),
                                generation_config=config)
        except Exception as error:
            self._diagnostic(error, examples)
            raise

    def save_checkpoint(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = dict(step=self.step, examples_seen=self.examples_seen, tokens_seen=self.tokens_seen,
                     sampler_position=self.sampler_position, config=asdict(self.config),
                     metadata=self.metadata, parameter_manifest=self.policy.parameter_manifest(),
                     reference_identity=self.reference_identity, rollout_status='discard_incomplete_group')
        self.policy.save_checkpoint(path / 'policy', state)
        if self.reference:
            self.reference.save_checkpoint(path / 'reference', {'identity': self.reference_identity})
        save_resume_state(path / 'resume.pt', self.optimizer, scheduler=self.scheduler,
                          scaler=self.scaler, **state)
        return state

    def load_checkpoint(self, path):
        path = Path(path)
        header = torch.load(path / 'resume.pt', map_location='cpu', weights_only=False)
        if header['config'] != asdict(self.config) or header['metadata'] != self.metadata:
            raise ValueError('Resume configuration or data/schema metadata differs')
        self.policy.load_checkpoint(path / 'policy')
        if self.reference:
            self.reference.load_checkpoint(path / 'reference')
            self.reference.model.requires_grad_(False)
            self.reference.model.eval()
            if _identity(self.reference.model) != header['reference_identity']:
                raise ValueError('Reference checkpoint identity mismatch')
        if self.policy.parameter_manifest() != header['parameter_manifest']:
            raise ValueError('Parameter manifest changed on resume')
        state = load_resume_state(path / 'resume.pt', self.optimizer, scheduler=self.scheduler, scaler=self.scaler)
        for name in ('step', 'examples_seen', 'tokens_seen', 'sampler_position', 'reference_identity'):
            setattr(self, name, state[name])
        self.last_rollouts = []
        return state
