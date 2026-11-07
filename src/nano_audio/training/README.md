# Training API

```python
from nano_audio.training import TrainingConfig, Trainer
cfg = TrainingConfig.for_profile('smoke', objective='grpo')
trainer = Trainer(policy, cfg, reward_fn=lambda example, rollout: score(example, rollout),
                  ledger_path=run_dir / 'compute.jsonl',
                  metadata={'data_hash': data_hash, 'schema_hash': schema_hash,
                            'parent_checkpoint': parent_id})
metrics = trainer.grpo_step(examples)
trainer.save_checkpoint(run_dir / 'checkpoint')
# Construct another Trainer with the same config/metadata, then:
trainer.load_checkpoint(run_dir / 'checkpoint')
```

`reward_fn(example, rollout)` returns binary 0/1. The rollout is the original
`generate` dictionary with `ids`, `text`, `eos`, `length_capped`. Capped outputs
receive zero reward. No re-tokenization occurs. G=4, temperature=1, top_k=0,
top_p=1; generation has only a recorded length safety cap. One optimizer pass
follows a fresh batch. The fixed reference deep-copies the whole policy,
including projector, before any training update; it is saved separately.

For SFT use `objective='sft'` and `trainer.sft_step(examples, targets=None)`.
Without `targets`, each example must have `target_text`. Explicit targets can
be strings or token-ID lists. The policy must expose `tokenizer.eos_token_id`
or `eos_token_id`; adapter `eos_token_id` takes precedence because the tokenizer
may use the original unpruned vocabulary. The trainer appends EOS if absent. Completion scoring is
responsible for the causal shift and conditioning on only the provided prefix.
Loss averages tokens per response, then examples equally.

The caller selects the effective batch (pass the full update's examples),
training pool, stage, and update loop. `effective_batch_size` is a profile
recommendation, not an implicit dataloader. Scoring uses singleton microbatches
for all configured microbatch sizes, with full-update normalization. Larger
microbatch settings are accepted but do not increase scoring concurrency.
`total_updates` controls a 5% warm-up followed by linear learning-rate decay.
The profile also describes staged budgets and checkpoint cadence for the caller.
`make_optimizer` preserves the policy's requires_grad flags and separates decoder
parameters by identity (including tied weights) from other trainable parameters.
The policy adapter/caller must freeze its encoder and configure stage-specific
trainability before constructing Trainer. CPU/MPS default to FP32; CUDA supports
explicit FP16 plus scaling or hardware-checked BF16.

Supply hashes/provenance in `metadata`; resume validates these and the full config.
Checkpoints contain optimizer, scheduler, scaler, Python/Torch/NumPy/CUDA RNG,
counters, sampler position, precision, parameter manifest and reference hash.
The caller must restore its dataloader at `trainer.sampler_position`. Checkpoints
are intended at optimizer boundaries. Incomplete rollout groups are discarded;
a resumed update samples a fresh group from the restored RNG. Local checkpoints
are trusted pickle inputs. Finite failures write a `.failure.pt` diagnostic next
to the ledger and raise without silently clamping the objective.

Ledger `sampled_token_surprisal` is a sampled cross-entropy diagnostic, not full
vocabulary entropy. Equal-reward groups can have zero policy-gradient signal;
the ledger reports their rate. Tiny fixture tests demonstrate infrastructure
correctness only and are not evidence of speech understanding.

For `SamsonePolicy` on CUDA, pass `precision='fp16'` explicitly: its internal
FP16 autocast requires the Trainer gradient scaler even with FP32 master weights.
Trainer rejects an incompatible precision setting for this policy. Ledger
`learning_rates` records rates actually used for the update;
`next_learning_rates` records scheduler rates after it. A smoke schedule with
`total_updates=1` has a positive first-update LR and zero next LR. For a genuine
two-update checkpoint continuity check, use `total_updates=2` from the start;
changing this config on resume is intentionally rejected.

Run the no-download fixture checks with:

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/test_training.py -q
```

The marked real-policy test is opt-in and offline. Provide an existing JSON
example with `target_text` and the public conditioning fields required by the
policy, including a local `audio_path` for audio training:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
NANO_AUDIO_TRAINING_INTEGRATION_EXAMPLE=/absolute/path/to/example.json \
PYTHONPATH=src .venv/bin/python -m pytest tests/test_training.py -m integration -q
```

It uses installed assets only, performs two SFT updates with nonzero LR,
saves after the first, then restores into a fresh optimizer/scheduler/scaler and
compares the repeated second loss and connector weights. The default fixture
checks additionally verify exact CPU full-model two-step resume continuity.
