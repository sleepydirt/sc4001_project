"""Offline training correctness fixtures; these are not speech evidence."""
import copy
import os
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

import torch
import pytest
from nano_audio.training import TrainingConfig, Trainer, grpo_loss, sft_loss


class TinyPolicy:
    device = 'cpu'
    eos_token_id = 2

    def __init__(self):
        self.model = torch.nn.Linear(3, 3)
        self.decoder = self.model

    def encode(self, text):
        return [int(text)]

    def completion_logprobs(self, example, completion_ids, completion_mask=None):
        logits = self.model(torch.tensor([[1., 0., 1.]]))
        return logits.log_softmax(-1).gather(1, completion_ids)

    def generate(self, example, generation_config):
        logits = self.model(torch.tensor([[1., 0., 1.]]))
        token = int(torch.multinomial(logits.softmax(-1), 1))
        return dict(ids=[token, 2], text=str(token), eos=True, length_capped=False)

    def parameter_manifest(self):
        return {n: list(p.shape) for n, p in self.model.named_parameters()}

    def save_checkpoint(self, path, training_state):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), Path(path) / 'model.pt')

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(Path(path) / 'model.pt', weights_only=True))


class TrainingTests(unittest.TestCase):
    def test_grpo_sign_detachment_and_mask(self):
        current = torch.zeros(4, 3, requires_grad=True)
        old = torch.zeros_like(current, requires_grad=True)
        ref = torch.zeros_like(current, requires_grad=True)
        mask = torch.tensor([[1, 1, 0]] * 4)
        loss = grpo_loss(current, old, ref, [1, 0, 0, 0], mask)
        loss.backward()
        self.assertLess(float(current.grad[0, 0]), 0)
        self.assertGreater(float(current.grad[1, 0]), 0)
        self.assertEqual(float(current.grad[:, 2].abs().sum()), 0)
        self.assertIsNone(old.grad)
        self.assertIsNone(ref.grad)

    def test_equal_rewards_and_population_std(self):
        current = torch.zeros(4, 2, requires_grad=True)
        loss = grpo_loss(current, current, current, [1]*4, torch.ones_like(current))
        loss.backward()
        self.assertEqual(float(current.grad.abs().sum()), 0)
        current = torch.tensor([[.1], [0.], [0.], [0.]], requires_grad=True)
        rewards = torch.tensor([1., 0., 0., 0.])
        advantages = (rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6)
        expected = -(current.exp().squeeze()*advantages).mean()
        actual = grpo_loss(current, torch.zeros_like(current), current.detach(), rewards, torch.ones_like(current))
        torch.testing.assert_close(actual, expected)

    def test_microbatch_equivalence(self):
        torch.manual_seed(4)
        scores = torch.randn(2, 4, 5, requires_grad=True)
        old = scores.detach().clone()
        ref = old + .1
        rewards = torch.tensor([[1., 0., 0., 0.], [0., 1., 1., 0.]])
        mask = torch.ones_like(scores)
        whole = grpo_loss(scores, old, ref, rewards, mask)
        accumulated = sum(grpo_loss(scores[i], old[i], ref[i], rewards[i], mask[i])/2 for i in range(2))
        torch.testing.assert_close(whole, accumulated)
        a = torch.autograd.grad(whole, scores, retain_graph=True)[0]
        b = torch.autograd.grad(accumulated, scores)[0]
        torch.testing.assert_close(a, b)

    def test_sft_masks_and_eos(self):
        scores = torch.tensor([[-1., -3., float('nan')]], requires_grad=True)
        loss = sft_loss(scores, torch.tensor([[1, 1, 0]]))
        self.assertEqual(float(loss.detach()), 2.)
        policy = TinyPolicy()
        policy.tokenizer = SimpleNamespace(eos_token_id=999)
        seen = []
        original = policy.completion_logprobs
        def score(example, ids, completion_mask=None):
            seen.append(ids.tolist())
            return original(example, ids, completion_mask)
        policy.completion_logprobs = score
        Trainer(policy).sft_step([{'target_text': '1'}])
        self.assertEqual(seen, [[[1, 2]]])

    def test_sft_accumulation_and_resume(self):
        torch.manual_seed(11)
        policy = TinyPolicy()
        trainer = Trainer(policy, TrainingConfig(total_updates=2))
        examples = [{'target_text': '1'}, {'target_text': '0'}]
        baseline = copy.deepcopy(policy)
        optimizer = torch.optim.AdamW(baseline.model.parameters(), lr=2e-5, weight_decay=.01)
        loss = sum(sft_loss(baseline.completion_logprobs(x, torch.tensor([[int(x['target_text']), 2]])))/2 for x in examples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline.model.parameters(), 1.)
        optimizer.step()
        trainer.sft_step(examples)
        for a, b in zip(policy.model.parameters(), baseline.model.parameters()):
            torch.testing.assert_close(a, b)
        with tempfile.TemporaryDirectory() as folder:
            trainer.save_checkpoint(folder)
            before_second = copy.deepcopy(policy.model.state_dict())
            second_row = trainer.sft_step(examples)
            self.assertGreater(second_row['learning_rates'][0], 0)
            self.assertTrue(any(not torch.equal(v, policy.model.state_dict()[n]) for n, v in before_second.items()))
            expected = copy.deepcopy(policy.model.state_dict())
            resumed = Trainer(TinyPolicy(), TrainingConfig(total_updates=2))
            self.assertGreater(resumed.optimizer.param_groups[0]['lr'], 0)
            resumed.load_checkpoint(folder)
            resumed.sft_step(examples)
            for name, value in expected.items():
                torch.testing.assert_close(value, resumed.policy.model.state_dict()[name], rtol=0, atol=0)

    def test_grpo_reference_and_rng_resume(self):
        torch.manual_seed(8)
        config = TrainingConfig.for_profile('smoke', 'grpo', total_updates=3)
        reward = lambda example, rollout: float(rollout['ids'][0] == 1)
        trainer = Trainer(TinyPolicy(), config, reward_fn=reward)
        reference = copy.deepcopy(trainer.reference.model.state_dict())
        with tempfile.TemporaryDirectory() as folder:
            trainer.grpo_step([{}])
            trainer.save_checkpoint(folder)
            expected_row = trainer.grpo_step([{}])
            expected = copy.deepcopy(trainer.policy.model.state_dict())
            resumed = Trainer(TinyPolicy(), config, reward_fn=reward)
            resumed.load_checkpoint(folder)
            actual_row = resumed.grpo_step([{}])
            self.assertEqual(expected_row['reward_mean'], actual_row['reward_mean'])
            for name, value in expected.items():
                torch.testing.assert_close(value, resumed.policy.model.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in trainer.reference.model.parameters()))
        self.assertTrue(all(p.grad is None for p in trainer.reference.model.parameters()))
        for name, value in reference.items():
            torch.testing.assert_close(value, trainer.reference.model.state_dict()[name])

    def test_one_step_learning_rate_ledger(self):
        trainer = Trainer(TinyPolicy())
        row = trainer.sft_step([{'target_text': '1'}])
        self.assertEqual(row['learning_rates'], [2e-5])
        self.assertEqual(row['next_learning_rates'], [0.])

    def test_profiles(self):
        self.assertEqual(TrainingConfig.for_profile('main', 'grpo').effective_batch_size, 8)
        self.assertEqual(TrainingConfig.for_profile('local_pilot').warmup_updates, 64)


class CausalDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 4)
        self.head = torch.nn.Linear(4, 5)
        self.config = SimpleNamespace(max_position_embeddings=2048)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, use_cache=False):
        return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(1)))


def adapter_fixture(device='cpu'):
    # Exercise actual adapter methods without constructing/downloading Samsone.
    from nano_audio.models.policy import SamsonePolicy
    policy = SamsonePolicy.__new__(SamsonePolicy)
    policy.device = torch.device(device)
    policy.model = torch.nn.Module()
    policy.model.projector = torch.nn.Linear(3, 4)
    policy.model.audio_encoder = torch.nn.Linear(3, 3).requires_grad_(False)
    policy.decoder = CausalDecoder()
    policy.model.add_module('decoder', policy.decoder)
    policy.model.to(device)
    policy.eos_token_id = 2
    policy.tokenizer = SimpleNamespace(eos_token_id=999)
    policy.encode = lambda text: [1]
    policy.build_prefix = lambda example: policy.model.projector(torch.ones(1, 2, 3, device=device))
    policy.parameter_manifest = lambda: {'fixture': True}
    return policy


def test_actual_policy_causal_scoring_and_trainer():
    policy = adapter_fixture()
    prefix = policy.build_prefix({})
    ids = torch.tensor([[1, policy.eos_token_id]])
    inputs = torch.cat([prefix, policy.decoder.embedding(ids)], 1)
    logits = policy.decoder(inputs_embeds=inputs, attention_mask=None).logits
    expected = logits[:, prefix.shape[1]-1:prefix.shape[1]+1].log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    actual = policy.completion_logprobs({}, ids)
    torch.testing.assert_close(actual, expected)
    trainer = Trainer(policy, TrainingConfig(total_updates=2))
    before = copy.deepcopy(policy.model.state_dict())
    row = trainer.sft_step([{}], targets=['target'])
    assert row['tokens'] == 2
    for name in ('projector.weight', 'decoder.head.weight'):
        assert not torch.equal(before[name], policy.model.state_dict()[name])
    assert all(p.grad is None for p in policy.model.audio_encoder.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_policy_internal_fp16_requires_scaler():
    policy = adapter_fixture('cuda')
    with pytest.raises(ValueError, match='internal CUDA FP16'):
        Trainer(policy, TrainingConfig(precision='fp32'))
    trainer = Trainer(policy, TrainingConfig(precision='fp16', total_updates=2))
    assert trainer.scaler.is_enabled()
    row = trainer.sft_step([{}], targets=['target'])
    assert row['gradient_norm'] > 0
    assert all(p.dtype == torch.float32 for p in policy.model.parameters())


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get('NANO_AUDIO_TRAINING_INTEGRATION_EXAMPLE'),
                    reason='Set local example JSON path to opt into real-model smoke; never downloads')
def test_local_samsone_two_step_checkpoint_continuity(tmp_path, monkeypatch):
    import json
    from nano_audio.models.policy import SamsonePolicy
    # Force all dependency loaders into offline mode, including upstream APIs.
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    example = json.loads(Path(os.environ['NANO_AUDIO_TRAINING_INTEGRATION_EXAMPLE']).read_text())
    policy = SamsonePolicy(device='cuda' if torch.cuda.is_available() else 'cpu')
    config = TrainingConfig.for_profile('smoke', total_updates=2,
                precision='fp16' if policy.device.type == 'cuda' else 'fp32')
    trainer = Trainer(policy, config)
    first = trainer.sft_step([example])
    assert first['learning_rates'][0] > 0
    checkpoint = tmp_path / 'checkpoint'
    trainer.save_checkpoint(checkpoint)
    # Save only intended trainable connector tensors to bound test memory.
    def connector_state():
        return {n: p.detach().cpu().clone() for n, p in policy.model.named_parameters()
                if p.requires_grad and ('projector' in n or 'sep_token' in n)}
    before = connector_state()
    expected_row = trainer.sft_step([example])
    expected = connector_state()
    assert expected_row['learning_rates'][0] > 0
    assert any(not torch.equal(before[n], v) for n, v in expected.items())
    # Release the previous Adam state before constructing a fresh optimizer.
    del trainer
    # A new optimizer/scheduler/scaler must load step-one state, not retain step two.
    resumed = Trainer(policy, config)
    assert resumed.optimizer.param_groups[0]['lr'] > 0
    resumed.load_checkpoint(checkpoint)
    row = resumed.sft_step([example])
    assert row['learning_rates'] == expected_row['learning_rates']
    assert row['loss'] == pytest.approx(expected_row['loss'], rel=1e-5, abs=1e-6)
    for name, value in connector_state().items():
        torch.testing.assert_close(value, expected[name], rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
