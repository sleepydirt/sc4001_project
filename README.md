# Nano speech-to-tool learning dynamics

Research code for **Learning dynamics of supervised and reinforcement post-training for nano speech-to-tool models**. The source specification is [research_instructions.md](research_instructions.md). This execution is a **minimal smoke test**, as requested after reading the proposal. A smoke result verifies mechanics; it does not establish post-training improvement.

Use Python 3.12 and `uv`. The tested GPU environment uses Torch/Torchaudio 2.9.1 CUDA 12.8 and Transformers 4.57.6; exact installed versions are in `constraints-linux-py312.txt`. `uv.lock` resolves the package dependency graph; the constraints file additionally includes test/upstream runtime dependencies. No system Python is modified.

```bash
bash scripts/bootstrap.sh
# bootstrap fetches/verifies the upstream revisions in artifacts/assets/manifest.json.
.venv/bin/nano-audio inspect-env
.venv/bin/nano-audio resolve-models
.venv/bin/nano-audio download --profile smoke
.venv/bin/nano-audio smoke-text
.venv/bin/nano-audio smoke-audio
.venv/bin/nano-audio prepare-data --profile smoke --device cuda
.venv/bin/nano-audio audit-data --profile smoke
.venv/bin/nano-audio train --run S_P1 --profile smoke
.venv/bin/nano-audio train --run S_P_SFT --profile smoke
.venv/bin/nano-audio train --run S_P_GRPO --profile smoke
.venv/bin/nano-audio evaluate --profile smoke
.venv/bin/nano-audio report --profile smoke
```

The default resumable orchestrator runs `local_pilot`. For this request, use **explicit `--profile smoke`**. Local pilot and main profiles are provided for later execution, not measured outcomes of this run. Main training uses final test schemas only during explicit evaluation, with three conditional continuation seeds. Stage/checkpoint identities prevent reusing different data/configuration silently. Failed stages retain errors and retry commands.

```bash
.venv/bin/nano-audio run-study --profile smoke
.venv/bin/pytest -q
NANO_AUDIO_INTEGRATION=1 .venv/bin/pytest -q tests/test_policy.py
# Later, on a suitable server; not submitted automatically:
sbatch scripts/v100.slurm
```

Samsone training uses the released Whisper-tiny encoder, native 50-token pooler, nonlinear 384→576 projector, separator parameter and officially pruned SmolLM2 decoder. Fresh initialization selects the original text-pretrained embedding/output rows with the exact same vocabulary map and loads original Whisper encoder weights. Whole state shapes, map and tying are asserted equal. Encoder features are cached before the trainable projector. Native inference remains separate from research prompts. FP32 master weights and FP16 autocast/scaling are used on this GPU.

GRPO samples four unconstrained completions per prompt with temperature one and no top-k/top-p restriction. Only complete semantic success earns reward. A frozen complete shared-initialization reference includes its own projector. All-zero-reward groups provide **no reward learning signal**; inspect the saved rollout/learning ledgers. Continued SFT and GRPO have the same parent checkpoint.

Artifacts contain real prompts/token IDs/predictions, exact hashes, data audits, gradient evidence and checkpoints. See `artifacts/reports/smoke/research_report.md` for measured outcomes, external blockers and implementation limits. Human-speech results, published benchmark reproduction and full experiments must not be inferred from synthetic or infrastructure checks.

Mellow research uses BF16 on the verified RTX 4070 Super because its FP16 decoder scoring was nonfinite; on a CUDA GPU without BF16 support it uses FP32. Native Mellow inference remains FP32 with its original sampling protocol. Compatibility details and failed precision diagnostics are retained under `artifacts/reproduction/`. The main server environment must be checked independently.

The eight-example diagnostic is available as `nano-audio overfit --profile local_pilot`. It is deliberately skipped by `run-study --profile smoke`. Grounding/cascade checks use `nano-audio evaluate --profile smoke --suite controls`. Audio2Tool consistency checks use `nano-audio audit-external --input rows.jsonl --catalog fixed_domain_catalogs.json --output audit.json`; this is an audit, not a substituted official benchmark score.

A subsequent bounded exact-reward GRPO development attempt is recorded in [the GRPO continuation report](artifacts/grpo_attempt_20261107/overall_report.md). It resolved the all-zero-reward cold start with stronger shared SFT, logged every sampled generation, and compared GRPO with continued SFT from identical parents. Those results are separate from the original one-update smoke.
