"""Core training algorithms for supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO).

Implements:
  - Masked response Negative Log-Likelihood (NLL) for SFT.
  - Group Relative Policy Optimization (GRPO) with group-normalized advantages, clipped surrogate ratios, and KL penalty.
  - Microbatch gradient accumulation with mixed precision (FP16 / BF16) and gradient scaling.
  - Deterministic checkpointing with exact RNG, optimizer, scheduler, and reference policy persistence.
"""
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


def _finite(value: torch.Tensor, name: str) -> None:
    """Raise FloatingPointError immediately if any NaN or Inf is encountered in the tensor."""
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite values detected in {name}")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean across tokens for each sequence, considering only unmasked (valid) positions."""
    mask = mask.to(device=value.device, dtype=torch.bool)
    if mask.shape != value.shape or not mask.any(-1).all():
        raise ValueError("Mask must match scores shape and contain at least one valid token per sequence")
    _finite(value[mask], "valid token scores")
    return torch.where(mask, value, 0).sum(-1) / mask.sum(-1)


def grpo_loss(
    current: torch.Tensor,
    old: torch.Tensor,
    reference: torch.Tensor,
    rewards: torch.Tensor | list,
    mask: torch.Tensor,
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> torch.Tensor:
    """Compute the Group Relative Policy Optimization (GRPO) loss for a group of sampled rollouts.
    
    Mathematical formulation:
      1. Advantage normalization within group G (typically G=4):
           A_i = (R_i - mean(R)) / (std(R) + 1e-6)
      2. Probability ratio between current policy and generation policy:
           r_i = exp(log_p_current - log_p_old)
      3. PPO-style clipped surrogate objective:
           surr = min(r_i * A_i, clip(r_i, 1-eps, 1+eps) * A_i)
      4. Token-level KL divergence penalty against frozen reference model:
           d = log_p_ref - log_p_current
           KL = exp(d) - d - 1
      5. Combined loss:
           Loss = -surr + beta * KL
    
    Args:
      current: Log probabilities of generated completion tokens under the trainable policy [G, T] or [B, G, T].
      old: Log probabilities under the policy at rollout generation time (detached).
      reference: Log probabilities under the frozen shared-initialization reference policy (detached).
      rewards: Task rewards earned by each rollout [G] or [B, G].
      mask: Boolean attention/valid-token mask for completions [G, T].
      epsilon: PPO-style clipping boundary parameter (default: 0.2).
      beta: Coefficient for the KL divergence penalty against the reference model (default: 0.01).
    """
    if current.ndim not in (2, 3) or old.shape != current.shape or reference.shape != current.shape:
        raise ValueError("Expected matching [G, T] or [B, G, T] log probability tensors")
    rewards = torch.as_tensor(rewards, device=current.device, dtype=current.dtype).detach()
    if rewards.shape != current.shape[:-1]:
        raise ValueError("Rewards shape must match completion batch dimensions")
    if epsilon < 0 or beta < 0:
        raise ValueError("epsilon and beta must be nonnegative")
    _finite(rewards, "rewards")

    # Step 1: Normalize rewards across the group to derive relative advantages
    advantage = (rewards - rewards.mean(-1, keepdim=True)) / (
        rewards.std(-1, unbiased=False, keepdim=True) + 1e-6
    )

    valid = mask.to(device=current.device, dtype=torch.bool)
    # Mask padding tokens to zero before exponentiating
    c = torch.where(valid, current, 0)
    o = torch.where(valid, old.detach(), 0)
    r = torch.where(valid, reference.detach(), 0)

    # Step 2: Importance sampling probability ratio
    ratio = (c - o).exp()

    # Step 3: Exact token-level KL divergence approximation: D_KL(pi_ref || pi_theta)
    d = r - c
    kl = d.exp() - d - 1

    # Step 4: Clipped surrogate objective with KL divergence regularization
    surrogate1 = ratio * advantage.unsqueeze(-1)
    surrogate2 = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage.unsqueeze(-1)
    objective = -torch.minimum(surrogate1, surrogate2) + beta * kl

    # Step 5: Average over valid completion tokens, then average over all completions in the group
    return _masked_mean(objective, valid).mean()


def sft_loss(logprobs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the mean response Negative Log-Likelihood (NLL) per example.
    
    Averages over valid target tokens per sequence, then averages equally across examples.
    """
    if mask is None:
        mask = torch.ones_like(logprobs, dtype=torch.bool)
    return -_masked_mean(logprobs, mask).mean()


@dataclass
class TrainingConfig:
    """Hyperparameter and hardware configuration for SFT and GRPO training."""
    profile: str = "smoke"
    objective: str = "sft"               # 'sft' or 'grpo'
    reward_mode: str = "binary"          # 'binary' (strict 0 or 1) or 'verifiable_partial'
    effective_batch_size: int = 1
    microbatch_size: int = 1            # Microbatch 1 avoids padding artifacts and memory spikes
    group_size: int = 4                 # Rollout group size G=4 for GRPO
    max_new_tokens: int = 128           # Maximum tokens generated per tool call completion
    epsilon: float = 0.2                # GRPO clipping boundary
    beta: float = 0.01                  # GRPO KL divergence penalty coefficient
    decoder_lr: float = 2e-5            # Language model decoder learning rate
    projector_lr: float = 1e-4          # Audio projector learning rate
    weight_decay: float = 0.01          # AdamW weight decay
    max_grad_norm: float = 1.0          # Maximum gradient clipping norm
    total_updates: int = 1              # Planned total optimizer updates
    warmup_fraction: float = 0.05       # Linear warmup fraction of total updates
    warmup_updates: int = 1
    continuation_updates: int = 1
    text_updates: int = 1
    alignment_updates: int = 1
    audio_updates: int = 1
    checkpoint_every: int = 1           # Evaluation and checkpoint interval
    overfit_cap: int = 200              # Diagnostic maximum update cap
    precision: str = "fp32"             # 'fp32', 'fp16', or 'bf16'

    @classmethod
    def for_profile(cls, name: str, objective: str = "sft", **overrides) -> TrainingConfig:
        """Create a preset configuration tailored to an experiment profile."""
        if name not in ("smoke", "local_pilot", "main"):
            raise ValueError(f"Unknown experiment profile: {name}")
        values = dict(profile=name, objective=objective)
        if name == "local_pilot":
            values.update(
                effective_batch_size=4,
                total_updates=32,
                warmup_updates=64,
                continuation_updates=32,
                text_updates=8,
                alignment_updates=8,
                audio_updates=16,
                checkpoint_every=8,
            )
        elif name == "main":
            values.update(
                effective_batch_size=8,
                total_updates=512,
                warmup_updates=256,
                continuation_updates=512,
                text_updates=256,
                alignment_updates=256,
                audio_updates=512,
                checkpoint_every=32,
            )
        # GRPO uses smaller learning rates to prevent policy collapse during RL updates
        if objective == "grpo":
            values.update(decoder_lr=1e-6, projector_lr=5e-6)
        values.update(overrides)
        return cls(**values)


def make_optimizer(policy, config: TrainingConfig) -> torch.optim.AdamW:
    """Create an AdamW optimizer with distinct parameter groups for the decoder and audio projector.
    
    The newly initialized audio projector typically requires a larger learning rate (e.g. 1e-4)
    than the pretrained language model decoder layers (e.g. 2e-5).
    """
    decoder_ids = {id(p) for p in policy.decoder.parameters()}
    groups = [[], []]
    for p in policy.model.parameters():
        if p.requires_grad:
            # Group 0: Decoder parameters; Group 1: Projector and other parameters
            groups[0 if id(p) in decoder_ids else 1].append(p)
    if not any(groups):
        raise ValueError("No trainable parameters found in the policy model")
    return torch.optim.AdamW(
        [
            {"params": ps, "lr": lr}
            for ps, lr in zip(groups, (config.decoder_lr, config.projector_lr))
            if ps
        ],
        weight_decay=config.weight_decay,
    )


def capture_rng_state() -> dict:
    """Capture the exact random number generator state across Python, PyTorch (CPU/CUDA), and NumPy."""
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict) -> None:
    """Restore the random number generator state across Python, PyTorch (CPU/CUDA), and NumPy."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])


def save_resume_state(path: Path | str, optimizer, *, scheduler=None, scaler=None, **metadata) -> None:
    """Atomically persist complete optimizer, scheduler, scaler, and RNG state to disk."""
    state = dict(
        metadata,
        optimizer=optimizer.state_dict(),
        rng=capture_rng_state(),
        scheduler=scheduler.state_dict() if scheduler else None,
        scaler=scaler.state_dict() if scaler else None,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_resume_state(path: Path | str, optimizer, *, scheduler=None, scaler=None) -> dict:
    """Load optimizer, scheduler, scaler, and RNG state from a saved checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng_state(state["rng"])
    return state


def _identity(model: torch.nn.Module) -> str:
    """Compute a deterministic hash fingerprint of all parameter names, shapes, and weights in a model."""
    hasher = hashlib.sha256()
    for name, value in model.state_dict().items():
        hasher.update(name.encode())
        hasher.update(str(value.shape).encode())
        hasher.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


class Trainer:
    """Execution engine for SFT and GRPO training updates.
    
    Supports:
      - Sequential single-example microbatching (memory efficient, no batch padding artifacts).
      - Mixed-precision autocast (FP16 / BF16) and gradient scaling via torch.amp.GradScaler.
      - Full deterministic checkpointing and resumable execution.
      - Rollout sampling, reward calculation, and KL regularized policy optimization.
    """

    def __init__(
        self,
        policy,
        config: TrainingConfig | None = None,
        *,
        reward_fn=None,
        reference_policy=None,
        optimizer=None,
        ledger_path: Path | str | None = None,
        metadata: dict | None = None,
    ):
        self.policy = policy
        self.config = config or TrainingConfig()

        # Validate objective and batch configurations
        if self.config.objective not in ("sft", "grpo"):
            raise ValueError("objective must be either 'sft' or 'grpo'")
        if self.config.group_size != 4 or self.config.microbatch_size < 1:
            raise ValueError("GRPO requires group_size G=4 and positive microbatch_size")
        if self.config.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("Unsupported precision: must be fp32, fp16, or bf16")
        if self.config.precision != "fp32" and torch.device(policy.device).type != "cuda":
            raise ValueError("Mixed precision (fp16/bf16) requires a CUDA device")
        if self.config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("The current CUDA device does not support BF16")

        # Set up frozen reference policy for GRPO KL penalty calculation
        self.reference = None
        if self.config.objective == "grpo":
            if reference_policy is not None:
                if {id(p) for p in reference_policy.model.parameters()} & {id(p) for p in policy.model.parameters()}:
                    raise ValueError("Reference and policy must own distinct parameter instances")
                self.reference = reference_policy
            else:
                self.reference = copy.deepcopy(policy)
            self.reference.model.requires_grad_(False)
            self.reference.model.eval()

        self.reference_identity = _identity(self.reference.model) if self.reference else None
        self.optimizer = optimizer or make_optimizer(policy, self.config)

        # Setup linear warmup and linear decay learning rate schedule
        warmup = max(1, math.ceil(self.config.total_updates * self.config.warmup_fraction))

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            return max(
                0.0,
                (self.config.total_updates - step) / max(1, self.config.total_updates - warmup),
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.scaler = torch.amp.GradScaler(
            "cuda", init_scale=128.0, enabled=self.config.precision == "fp16"
        )
        self.reward_fn = reward_fn
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.metadata = metadata or {}

        # Tracking state counters
        self.step = 0
        self.examples_seen = 0
        self.tokens_seen = 0
        self.sampler_position = 0
        self.last_rollouts = []

    def _autocast(self):
        """Context manager for PyTorch AMP autocasting based on configured precision."""
        return torch.autocast(
            device_type=torch.device(self.policy.device).type,
            dtype=torch.float16 if self.config.precision == "fp16" else torch.bfloat16,
            enabled=self.config.precision != "fp32",
        )

    def _score(self, policy, example: dict, ids: list[int]) -> torch.Tensor:
        """Compute the log probability of completion token sequence under the specified policy."""
        tensor = torch.tensor([ids], dtype=torch.long, device=policy.device)
        with self._autocast():
            scores = policy.completion_logprobs(example, tensor, torch.ones_like(tensor))
        if scores.shape != tensor.shape:
            raise ValueError("completion_logprobs must return tensor with shape [1, T]")
        _finite(scores, "completion log probabilities")
        return scores.float()

    def _begin(self, examples: list) -> None:
        """Prepare for an optimizer update: zero gradients and reset memory stats."""
        if not examples:
            raise ValueError("An update requires at least one example")
        self.policy.model.eval()  # Disable dropout layers while preserving autograd
        self.optimizer.zero_grad(set_to_none=True)
        if torch.device(self.policy.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.policy.device)

    def _finish(self, loss: float, examples: list, tokens: int, started: float, **metrics) -> dict:
        """Complete an optimizer update: clip gradients, step optimizer/scheduler, and record ledger."""
        _finite(torch.tensor(loss), "loss")
        self.scaler.unscale_(self.optimizer)

        # Clip gradient norms to prevent destabilizing large gradient steps
        parameters = [p for p in self.policy.model.parameters() if p.requires_grad and p.grad is not None]
        if not parameters:
            raise RuntimeError("No gradients computed for trainable policy parameters")
        for p in parameters:
            _finite(p.grad, "gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.config.max_grad_norm, error_if_nonfinite=True
        )

        learning_rates_used = [g["lr"] for g in self.optimizer.param_groups]
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        for p in self.policy.model.parameters():
            _finite(p, "updated parameter")

        # Advance step counters
        self.step += 1
        self.examples_seen += len(examples)
        self.sampler_position += len(examples)
        self.tokens_seen += tokens

        row = dict(
            step=self.step,
            objective=self.config.objective,
            profile=self.config.profile,
            loss=loss,
            gradient_norm=float(norm),
            examples=len(examples),
            tokens=tokens,
            examples_seen=self.examples_seen,
            tokens_seen=self.tokens_seen,
            elapsed_seconds=time.perf_counter() - started,
            precision=self.config.precision,
            loss_scale=float(self.scaler.get_scale()),
            reference_identity=self.reference_identity,
            learning_rates=learning_rates_used,
            next_learning_rates=[g["lr"] for g in self.optimizer.param_groups],
            **metrics,
        )
        row["peak_memory_bytes"] = (
            torch.cuda.max_memory_allocated(self.policy.device)
            if torch.device(self.policy.device).type == "cuda"
            else None
        )

        # Append step metrics to training ledger file
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def _diagnostic(self, error: Exception, examples: list) -> None:
        """Save a failure checkpoint dump with error details for offline debugging."""
        self.optimizer.zero_grad(set_to_none=True)
        if self.ledger_path:
            path = self.ledger_path.with_suffix(".failure.pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "error": repr(error),
                    "step": self.step,
                    "examples": examples,
                    "rollouts": self.last_rollouts,
                    "rng": capture_rng_state(),
                },
                path,
            )

    def sft_step(self, examples: list[dict], targets: list[str] | None = None) -> dict:
        """Perform one supervised fine-tuning (SFT) update on a batch of examples.
        
        Computes response token NLL for each example sequentially (microbatch 1)
        and accumulates gradients across the batch before taking an optimizer step.
        """
        started = time.perf_counter()
        examples = list(examples)
        self._begin(examples)

        if targets is None:
            targets = [x["target_text"] for x in examples]
        if len(targets) != len(examples):
            raise ValueError("Target list length must match examples length")

        eos = getattr(self.policy, "eos_token_id", None)
        if eos is None:
            eos = getattr(getattr(self.policy, "tokenizer", None), "eos_token_id", None)
        if eos is None:
            raise ValueError("Policy must expose tokenizer.eos_token_id or eos_token_id")

        total, tokens = 0.0, 0
        try:
            # Sequential single-item scoring gives each example exactly 1/N weight without padding ambiguity
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

    def grpo_step(self, examples: list[dict]) -> dict:
        """Perform one Group Relative Policy Optimization (GRPO) update on a batch of speech prompts.
        
        For each speech prompt:
          1. Sample G=4 unconstrained completions (temperature=1.0).
          2. Score each completion with exact semantic binary reward (0 or 1).
          3. Evaluate token log probabilities under current policy and frozen reference policy.
          4. Normalize rewards across the group into advantage estimates.
          5. Compute clipped surrogate loss + KL divergence penalty.
          6. Accumulate gradients and update parameters.
        """
        if self.reference is None or self.reward_fn is None:
            raise ValueError("GRPO requires a fixed reference policy and a reward_fn")
        started = time.perf_counter()
        examples = list(examples)
        self._begin(examples)
        self.last_rollouts = []
        total, tokens, equal_groups, reward_sum, kl_sum, entropy_sum = 0.0, 0, 0, 0.0, 0.0, 0.0

        # Unconstrained rollout sampling configuration (no top-k/top-p truncation)
        config = dict(
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            max_new_tokens=self.config.max_new_tokens,
        )

        try:
            # 1. Rollout phase: generate completions and score under current and reference policies
            with torch.no_grad():
                for example in examples:
                    group = []
                    for _ in range(4):
                        rollout = dict(self.policy.generate(example, config))
                        ids = list(rollout["ids"])
                        if not ids:
                            raise ValueError("Empty rollout cannot be scored")
                        reward = float(self.reward_fn(example, rollout))
                        if self.config.reward_mode not in ("binary", "verifiable_partial"):
                            raise ValueError("Unknown reward mode")
                        if not math.isfinite(reward) or not 0 <= reward <= 1:
                            raise ValueError("Reward must be finite and in [0, 1]")
                        if self.config.reward_mode == "binary" and reward not in (0.0, 1.0):
                            raise ValueError("Main task reward must be binary (0 or 1)")
                        # Truncated responses that hit max_new_tokens without EOS earn 0 reward
                        if rollout.get("length_capped", False):
                            reward = 0.0
                        group.append(
                            dict(
                                rollout=rollout,
                                ids=ids,
                                reward=reward,
                                old=self._score(self.policy, example, ids).detach(),
                                reference=self._score(self.reference, example, ids).detach(),
                            )
                        )
                    self.last_rollouts.append(group)

            # 2. Optimization phase: calculate advantages, clipped surrogate loss, and KL penalty
            for example, group in zip(examples, self.last_rollouts):
                rewards = torch.tensor([r["reward"] for r in group], device=self.policy.device)
                std = rewards.std(unbiased=False)
                advantages = (rewards - rewards.mean()) / (std + 1e-6)

                # Groups where all 4 rollouts earn identical rewards (all 0 or all 1) provide no gradient signal
                equal_groups += int(std == 0)
                reward_sum += float(rewards.sum())

                for index, row in enumerate(group):
                    current = self._score(self.policy, example, row["ids"])
                    ratio = (current - row["old"]).exp()
                    d = row["reference"] - current
                    kl = d.exp() - d - 1

                    # PPO-style clipped objective
                    surr1 = ratio * advantages[index]
                    surr2 = ratio.clamp(1 - self.config.epsilon, 1 + self.config.epsilon) * advantages[index]
                    terms = -torch.minimum(surr1, surr2) + self.config.beta * kl

                    loss = terms.mean() / (len(examples) * 4)
                    _finite(loss, "GRPO loss")
                    self.scaler.scale(loss).backward()

                    total += float(loss.detach())
                    tokens += len(row["ids"])
                    kl_sum += float(kl.detach().mean())
                    entropy_sum += float(-current.detach().mean())

            return self._finish(
                total,
                examples,
                tokens,
                started,
                reward_mean=reward_sum / (len(examples) * 4),
                equal_reward_group_rate=equal_groups / len(examples),
                kl=kl_sum / (len(examples) * 4),
                sampled_token_surprisal=entropy_sum / (len(examples) * 4),
                reward_variance=float(
                    torch.tensor([r["reward"] for g in self.last_rollouts for r in g]).var(unbiased=False)
                ),
                generation_config=config,
            )
        except Exception as error:
            self._diagnostic(error, examples)
            raise

    def save_checkpoint(self, path: Path | str) -> dict:
        """Persist complete policy, reference, optimizer, and training metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = dict(
            step=self.step,
            examples_seen=self.examples_seen,
            tokens_seen=self.tokens_seen,
            sampler_position=self.sampler_position,
            config=asdict(self.config),
            metadata=self.metadata,
            parameter_manifest=self.policy.parameter_manifest(),
            reference_identity=self.reference_identity,
            rollout_status="discard_incomplete_group",
        )
        self.policy.save_checkpoint(path / "policy", state)
        if self.reference:
            self.reference.save_checkpoint(path / "reference", {"identity": self.reference_identity})
        save_resume_state(
            path / "resume.pt",
            self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            **state,
        )
        return state

    def load_checkpoint(self, path: Path | str) -> dict:
        """Restore training state and model parameters from a saved checkpoint."""
        path = Path(path)
        header = torch.load(path / "resume.pt", map_location="cpu", weights_only=False)
        if header["config"] != asdict(self.config) or header["metadata"] != self.metadata:
            raise ValueError("Resume configuration or metadata differs from checkpoint")
        self.policy.load_checkpoint(path / "policy")
        if self.reference:
            self.reference.load_checkpoint(path / "reference")
            self.reference.model.requires_grad_(False)
            self.reference.model.eval()
            if _identity(self.reference.model) != header["reference_identity"]:
                raise ValueError("Reference checkpoint identity mismatch")
        if self.policy.parameter_manifest() != header["parameter_manifest"]:
            raise ValueError("Parameter manifest changed on resume")
        state = load_resume_state(
            path / "resume.pt", self.optimizer, scheduler=self.scheduler, scaler=self.scaler
        )
        for name in ("step", "examples_seen", "tokens_seen", "sampler_position", "reference_identity"):
            setattr(self, name, state[name])
        self.last_rollouts = []
        return state
