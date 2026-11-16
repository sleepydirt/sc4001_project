"""Offline unit and regression tests for training routines and loss functions.

These tests verify algorithmic correctness independently of large audio models:
1. GRPO Loss Properties: Advantage normalization, gradient sign consistency, detachment
   of old/reference policies, population standard deviation, and mask behavior.
2. Microbatch Invariance: Equivalence between single-pass batching and sequential microbatch
   accumulation with size 1.
3. SFT Training & Checkpointing: Response token masking, automatic EOS token appending,
   gradient accumulation, and state resumption across checkpoints.
4. Reference Model Immutability: Ensuring frozen reference policy weights remain unchanged
   and detached from autograd graphs.
"""

from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nano_audio.training import Trainer, TrainingConfig, grpo_loss, sft_loss


class TinyPolicy:
    """Minimal lightweight policy mockup for fast, offline unit testing without GPU dependencies."""

    device = "cpu"
    eos_token_id = 2

    def __init__(self):
        # A simple linear layer standing in for a language model decoder
        self.model = torch.nn.Linear(3, 3)
        self.decoder = self.model

    def encode(self, text: str) -> list[int]:
        """Mock tokenizer encoding returning an integer token ID from a numeric string."""
        return [int(text)]

    def completion_logprobs(self, example: dict, completion_ids: torch.Tensor, completion_mask=None) -> torch.Tensor:
        """Compute mock completion log-probabilities by gathering logits on target token IDs."""
        logits = self.model(torch.tensor([[1.0, 0.0, 1.0]]))
        return logits.log_softmax(-1).gather(1, completion_ids)

    def generate(self, example: dict, generation_config: dict | None = None) -> dict:
        """Generate a mock sample rollout using multinomial sampling, terminating with EOS."""
        logits = self.model(torch.tensor([[1.0, 0.0, 1.0]]))
        token = int(torch.multinomial(logits.softmax(-1), 1))
        return {
            "ids": [token, self.eos_token_id],
            "text": str(token),
            "eos": True,
            "length_capped": False,
        }

    def parameter_manifest(self) -> dict:
        """Return parameter shape manifest dictionary."""
        return {n: list(p.shape) for n, p in self.model.named_parameters()}

    def save_checkpoint(self, path: str | Path, training_state: dict | None = None) -> None:
        """Save model state dict to disk."""
        target_dir = Path(path)
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), target_dir / "model.pt")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state dict from disk."""
        self.model.load_state_dict(torch.load(Path(path) / "model.pt", weights_only=True))


class TrainingTests(unittest.TestCase):
    """Test suite covering GRPO and SFT mathematical properties and trainer operations."""

    def test_grpo_sign_detachment_and_mask(self):
        """Verify GRPO advantage gradient signs, reference detachment, and padding masking.
        
        Test Setup:
        - 4 completion rollouts with 3 tokens each.
        - Token 0 and 1 are valid; token 2 is padding (mask = [1, 1, 0]).
        - Rewards: [1, 0, 0, 0] -> candidate 0 is correct, candidates 1..3 are incorrect.
        
        Expected Mathematical Behaviors:
        1. Advantage Signs:
           Mean reward = 0.25. Candidate 0 advantage > 0; Candidate 1 advantage < 0.
        2. Gradient Directions:
           Objective is -min(ratio * A, ...) + beta * KL.
           For candidate 0 (A > 0), minimizing loss increases log-probability -> d(loss)/d(current) < 0.
           For candidate 1 (A < 0), minimizing loss decreases log-probability -> d(loss)/d(current) > 0.
        3. Padding Mask:
           Token index 2 has mask 0, so its loss gradient must be identically 0.
        4. Autograd Detachment:
           Old policy and reference policy tensors must receive no gradients (grad is None).
        """
        current = torch.zeros(4, 3, requires_grad=True)
        old = torch.zeros_like(current, requires_grad=True)
        ref = torch.zeros_like(current, requires_grad=True)
        # Padding mask: token 0 and 1 are active; token 2 is padding
        mask = torch.tensor([[1, 1, 0]] * 4)
        rewards = [1, 0, 0, 0]

        loss = grpo_loss(current, old, ref, rewards, mask)
        loss.backward()

        # Candidate 0 (high reward, advantage > 0): loss gradient is negative (pushes logprob up)
        self.assertLess(float(current.grad[0, 0]), 0)
        # Candidate 1 (low reward, advantage < 0): loss gradient is positive (pushes logprob down)
        self.assertGreater(float(current.grad[1, 0]), 0)
        # Masked token at index 2 must have zero gradient across all candidates
        self.assertEqual(float(current.grad[:, 2].abs().sum()), 0)
        # Old rollout policy and reference policy must be detached from backward graph
        self.assertIsNone(old.grad)
        self.assertIsNone(ref.grad)

    def test_equal_rewards_and_population_std(self):
        """Verify zero advantage when rewards are equal, and check population std calculation.
        
        1. Equal Rewards Case:
           When all rollouts receive the exact same reward [1, 1, 1, 1], advantage is 0.
           With advantage = 0 and equal reference logprobs, gradients must be identically 0.
        2. Population Standard Deviation Formula:
           GRPO advantage normalization uses population standard deviation:
             A_i = (R_i - mean(R)) / (std(R, unbiased=False) + 1e-6)
           Verify that grpo_loss matches manual formula calculation.
        """
        # Equal rewards case: advantages become 0, resulting in zero policy gradient
        current = torch.zeros(4, 2, requires_grad=True)
        loss = grpo_loss(current, current, current, [1] * 4, torch.ones_like(current))
        loss.backward()
        self.assertEqual(float(current.grad.abs().sum()), 0)

        # Mathematical formula verification with population std (unbiased=False)
        current = torch.tensor([[0.1], [0.0], [0.0], [0.0]], requires_grad=True)
        rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        expected = -(current.exp().squeeze() * advantages).mean()
        actual = grpo_loss(
            current,
            torch.zeros_like(current),
            current.detach(),
            rewards,
            torch.ones_like(current),
        )
        torch.testing.assert_close(actual, expected)

    def test_microbatch_equivalence(self):
        """Verify that microbatch gradient accumulation is mathematically identical to full batching.
        
        Accumulating sequential microbatches (size 1) must produce identical losses and
        exact gradients to a single aggregated batch, proving zero padding bias.
        """
        torch.manual_seed(4)
        scores = torch.randn(2, 4, 5, requires_grad=True)
        old = scores.detach().clone()
        ref = old + 0.1
        rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
        mask = torch.ones_like(scores)

        # Full batch loss (evaluated in one call)
        whole = grpo_loss(scores, old, ref, rewards, mask)

        # Microbatch accumulated loss (accumulated over two individual microbatches of size 1)
        accumulated = sum(grpo_loss(scores[i], old[i], ref[i], rewards[i], mask[i]) / 2 for i in range(2))
        torch.testing.assert_close(whole, accumulated)

        # Gradient equivalence check: autograd gradients must be identical
        grad_whole = torch.autograd.grad(whole, scores, retain_graph=True)[0]
        grad_accumulated = torch.autograd.grad(accumulated, scores)[0]
        torch.testing.assert_close(grad_whole, grad_accumulated)

    def test_sft_masks_and_eos(self):
        """Verify SFT NLL loss with masking and automatic EOS token suffixing.
        
        1. Masked NLL:
           Positions where mask is 0 are ignored. Even if unmasked scores contain NaN,
           the masked loss must evaluate to a finite value.
        2. Automatic EOS Appending:
           If an example target text lacks the EOS token, Trainer.sft_step must automatically
           append policy.eos_token_id before scoring.
        """
        # Masked loss calculation: token 0 is -1, token 1 is -3, token 2 is NaN (masked out with 0)
        # Loss = -((-1.0 + -3.0) / 2) = 2.0
        scores = torch.tensor([[-1.0, -3.0, float("nan")]], requires_grad=True)
        loss = sft_loss(scores, torch.tensor([[1, 1, 0]]))
        self.assertEqual(float(loss.detach()), 2.0)

        # Verify automatic EOS appending when target text lacks EOS token
        policy = TinyPolicy()
        policy.tokenizer = SimpleNamespace(eos_token_id=999)
        seen_completions = []
        original_logprobs = policy.completion_logprobs

        def record_score(example, ids, completion_mask=None):
            seen_completions.append(ids.tolist())
            return original_logprobs(example, ids, completion_mask)

        policy.completion_logprobs = record_score
        Trainer(policy).sft_step([{"target_text": "1"}])
        # Target token '1' is automatically followed by policy's eos_token_id '2'
        self.assertEqual(seen_completions, [[[1, 2]]])

    def test_sft_accumulation_and_resume(self):
        """Verify SFT optimizer step fidelity and exact state restoration after checkpoint loading.
        
        1. Manual Baseline Comparison:
           Compares Trainer.sft_step parameter updates directly against an independent manual
           PyTorch AdamW step with identical learning rate and gradient clipping.
        2. Checkpoint State Restoration:
           Saves checkpoint after step 1, runs step 2, instantiates a fresh Trainer instance,
           loads step 1 checkpoint, runs step 2, and verifies parameter weights match bit-for-bit.
        """
        torch.manual_seed(11)
        policy = TinyPolicy()
        trainer = Trainer(policy, TrainingConfig(total_updates=2))
        examples = [{"target_text": "1"}, {"target_text": "0"}]

        # Step A: Independent baseline step using standard PyTorch AdamW
        baseline = copy.deepcopy(policy)
        optimizer = torch.optim.AdamW(baseline.model.parameters(), lr=2e-5, weight_decay=0.01)
        loss = sum(
            sft_loss(baseline.completion_logprobs(x, torch.tensor([[int(x["target_text"]), 2]]))) / 2
            for x in examples
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline.model.parameters(), 1.0)
        optimizer.step()

        # Step B: Trainer step using sequential microbatch accumulation
        trainer.sft_step(examples)

        # Model weights must match baseline exactly
        for a, b in zip(policy.model.parameters(), baseline.model.parameters()):
            torch.testing.assert_close(a, b)

        # Step C: Test checkpoint saving and resumption
        with tempfile.TemporaryDirectory() as folder:
            # Save checkpoint after first update
            trainer.save_checkpoint(folder)
            before_second = copy.deepcopy(policy.model.state_dict())
            second_row = trainer.sft_step(examples)
            self.assertGreater(second_row["learning_rates"][0], 0)
            self.assertTrue(any(not torch.equal(v, policy.model.state_dict()[n]) for n, v in before_second.items()))

            # Record expected weights after second step
            expected = copy.deepcopy(policy.model.state_dict())

            # Instantiate fresh Trainer and resume from saved checkpoint
            resumed = Trainer(TinyPolicy(), TrainingConfig(total_updates=2))
            self.assertGreater(resumed.optimizer.param_groups[0]["lr"], 0)
            resumed.load_checkpoint(folder)
            resumed.sft_step(examples)

            # Resumed parameters must match expected weights exactly
            for name, value in expected.items():
                torch.testing.assert_close(value, resumed.policy.model.state_dict()[name], rtol=0, atol=0)

    def test_grpo_reference_and_rng_resume(self):
        """Verify GRPO reference model freezing and RNG state reproducibility across resume.
        
        1. Reference Model Immobility:
           The reference model used to compute KL divergence must have requires_grad=False,
           receive no gradients (grad is None), and have unchanged weights across training updates.
        2. Stochastic Rollout Reproducibility:
           Because GRPO samples rollouts stochastically, exact reproducibility requires restoring
           Python, NumPy, and PyTorch RNG states from the checkpoint.
        """
        torch.manual_seed(8)
        config = TrainingConfig.for_profile("smoke", "grpo", total_updates=3)
        # Reward function: reward is 1 if rollout starts with token 1, else 0
        reward_fn = lambda example, rollout: float(rollout["ids"][0] == 1)
        trainer = Trainer(TinyPolicy(), config, reward_fn=reward_fn)
        reference_snapshot = copy.deepcopy(trainer.reference.model.state_dict())

        with tempfile.TemporaryDirectory() as folder:
            trainer.grpo_step([{}])
            trainer.save_checkpoint(folder)
            expected_row = trainer.grpo_step([{}])
            expected_weights = copy.deepcopy(trainer.policy.model.state_dict())

            # Resume in a new Trainer instance from checkpoint
            resumed = Trainer(TinyPolicy(), config, reward_fn=reward_fn)
            resumed.load_checkpoint(folder)
            actual_row = resumed.grpo_step([{}])

            # RNG resumption guarantees identical rollout generation, rewards, and weights
            self.assertEqual(expected_row["reward_mean"], actual_row["reward_mean"])
            for name, value in expected_weights.items():
                torch.testing.assert_close(value, resumed.policy.model.state_dict()[name], rtol=0, atol=0)

        # Reference model must have no gradient tracking and remain strictly identical
        self.assertTrue(all(not p.requires_grad for p in trainer.reference.model.parameters()))
        self.assertTrue(all(p.grad is None for p in trainer.reference.model.parameters()))
        for name, value in reference_snapshot.items():
            torch.testing.assert_close(value, trainer.reference.model.state_dict()[name])

    def test_one_step_learning_rate_ledger(self):
        """Verify learning rate scheduler metric reporting in step dictionary.
        
        Checks that 'learning_rates' records the LR used during the step (2e-5),
        and 'next_learning_rates' records the scheduled LR for the subsequent update (0.0).
        """
        trainer = Trainer(TinyPolicy())
        row = trainer.sft_step([{"target_text": "1"}])
        self.assertEqual(row["learning_rates"], [2e-5])
        self.assertEqual(row["next_learning_rates"], [0.0])

    def test_profiles(self):
        """Verify hyperparameter settings across predefined configuration profiles."""
        self.assertEqual(TrainingConfig.for_profile("main", "grpo").effective_batch_size, 8)
        self.assertEqual(TrainingConfig.for_profile("local_pilot").warmup_updates, 64)


class CausalDecoder(torch.nn.Module):
    """Mock causal decoder for adapter verification."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 4)
        self.head = torch.nn.Linear(4, 5)
        self.config = SimpleNamespace(max_position_embeddings=2048)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(self, inputs_embeds: torch.Tensor, attention_mask=None, use_cache: bool = False) -> SimpleNamespace:
        return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(dim=1)))


def adapter_fixture(device: str = "cpu"):
    """Exercise actual adapter methods without downloading real Samsone checkpoint weights.
    
    Constructs a lightweight mock SamsonePolicy structure:
      - Trainable MLP projector: Linear(3, 4)
      - Frozen Whisper audio encoder: Linear(3, 3) with requires_grad=False
      - Text decoder: CausalDecoder
    """
    from nano_audio.models.policy import SamsonePolicy

    policy = SamsonePolicy.__new__(SamsonePolicy)
    policy.device = torch.device(device)
    policy.model = torch.nn.Module()
    policy.model.projector = torch.nn.Linear(3, 4)
    policy.model.audio_encoder = torch.nn.Linear(3, 3).requires_grad_(False)
    policy.decoder = CausalDecoder()
    policy.model.add_module("decoder", policy.decoder)
    policy.model.to(device)
    policy.eos_token_id = 2
    policy.tokenizer = SimpleNamespace(eos_token_id=999)
    policy.encode = lambda text: [1]
    policy.build_prefix = lambda example: policy.model.projector(torch.ones(1, 2, 3, device=device))
    policy.parameter_manifest = lambda: {"fixture": True}
    return policy


def test_actual_policy_causal_scoring_and_trainer():
    """Verify adapter causal scoring alignment and parameter gradient isolation in Trainer.
    
    Verifies:
      1. Completion log-probability calculation matches manual causal slicing.
      2. Projector and decoder weights are updated after an SFT step.
      3. Audio encoder weights receive no gradients and remain frozen.
    """
    policy = adapter_fixture()
    prefix = policy.build_prefix({})
    ids = torch.tensor([[1, policy.eos_token_id]])
    inputs = torch.cat([prefix, policy.decoder.embedding(ids)], dim=1)
    logits = policy.decoder(inputs_embeds=inputs, attention_mask=None).logits
    expected = logits[:, prefix.shape[1] - 1 : prefix.shape[1] + 1].log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    actual = policy.completion_logprobs({}, ids)
    torch.testing.assert_close(actual, expected)

    trainer = Trainer(policy, TrainingConfig(total_updates=2))
    before = copy.deepcopy(policy.model.state_dict())
    row = trainer.sft_step([{}], targets=["target"])
    assert row["tokens"] == 2

    # Projector and decoder parameters must update
    for name in ("projector.weight", "decoder.head.weight"):
        assert not torch.equal(before[name], policy.model.state_dict()[name])

    # Audio encoder parameters must receive NO gradients
    assert all(p.grad is None for p in policy.model.audio_encoder.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_policy_internal_fp16_requires_scaler():
    """Verify that FP16 training on CUDA enforces the use of GradScaler.
    
    - Specifying precision='fp32' with internal FP16 model raises ValueError.
    - Specifying precision='fp16' activates GradScaler.
    - Master parameters remain in FP32 precision to prevent gradient underflow.
    """
    policy = adapter_fixture("cuda")
    with pytest.raises(ValueError, match="internal CUDA FP16"):
        Trainer(policy, TrainingConfig(precision="fp32"))
    trainer = Trainer(policy, TrainingConfig(precision="fp16", total_updates=2))
    assert trainer.scaler.is_enabled()
    row = trainer.sft_step([{}], targets=["target"])
    assert row["gradient_norm"] > 0
    assert all(p.dtype == torch.float32 for p in policy.model.parameters())


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NANO_AUDIO_TRAINING_INTEGRATION_EXAMPLE"),
    reason="Set local example JSON path to opt into real-model smoke; never downloads",
)
def test_local_samsone_two_step_checkpoint_continuity(tmp_path, monkeypatch):
    """Integration test verifying checkpoint continuity on full Samsone weights.
    
    Verifies:
      1. SFT update steps advance learning rates and modify connector weights (projector / sep_token).
      2. Checkpoint saving and resumption restores exact model state and optimizer momentum.
      3. Resumed step produces identical losses and updated parameter tensors.
    """
    import json
    from nano_audio.models.policy import SamsonePolicy

    # Force all dependency loaders into offline mode, including upstream APIs
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    example = json.loads(Path(os.environ["NANO_AUDIO_TRAINING_INTEGRATION_EXAMPLE"]).read_text())
    policy = SamsonePolicy(device="cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig.for_profile(
        "smoke",
        total_updates=2,
        precision="fp16" if policy.device.type == "cuda" else "fp32",
    )
    trainer = Trainer(policy, config)
    first = trainer.sft_step([example])
    assert first["learning_rates"][0] > 0
    checkpoint = tmp_path / "checkpoint"
    trainer.save_checkpoint(checkpoint)

    # Save only intended trainable connector tensors to bound test memory
    def connector_state():
        return {
            n: p.detach().cpu().clone()
            for n, p in policy.model.named_parameters()
            if p.requires_grad and ("projector" in n or "sep_token" in n)
        }

    before = connector_state()
    expected_row = trainer.sft_step([example])
    expected = connector_state()
    assert expected_row["learning_rates"][0] > 0
    assert any(not torch.equal(before[n], v) for n, v in expected.items())

    # Release the previous Adam state before constructing a fresh optimizer
    del trainer

    # A new optimizer/scheduler/scaler must load step-one state, not retain step two
    resumed = Trainer(policy, config)
    assert resumed.optimizer.param_groups[0]["lr"] > 0
    resumed.load_checkpoint(checkpoint)
    row = resumed.sft_step([example])
    assert row["learning_rates"] == expected_row["learning_rates"]
    assert row["loss"] == pytest.approx(expected_row["loss"], rel=1e-5, abs=1e-6)
    for name, value in connector_state().items():
        torch.testing.assert_close(value, expected[name], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
