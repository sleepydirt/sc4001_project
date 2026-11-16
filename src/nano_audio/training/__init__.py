"""Auditable response-only SFT and GRPO training routines and loss functions.

Exports:
- TrainingConfig: Hyperparameter dataclass and profile presets (smoke, local_pilot, main).
- Trainer: Single-example microbatch execution engine for SFT and GRPO.
- grpo_loss: Group Relative Policy Optimization clipped surrogate loss with KL divergence regularization.
- sft_loss: Response token negative log-likelihood (NLL) loss with masking.
- make_optimizer: Parameter group optimizer builder (separate projector and decoder learning rates).
- capture_rng_state / restore_rng_state: Multi-framework RNG state capture across Python, NumPy, and PyTorch.
- save_resume_state / load_resume_state: Checkpoint serialization and restoration utilities.
"""

from .core import (
    Trainer,
    TrainingConfig,
    capture_rng_state,
    grpo_loss,
    load_resume_state,
    make_optimizer,
    restore_rng_state,
    save_resume_state,
    sft_loss,
)

__all__ = [
    "TrainingConfig",
    "Trainer",
    "grpo_loss",
    "sft_loss",
    "make_optimizer",
    "capture_rng_state",
    "restore_rng_state",
    "save_resume_state",
    "load_resume_state",
]
