# Server handoff

> **GRPO follow-up (7 November 2026):** The subsequent one-hour-budget attempt established real binary-reward GRPO gradients and a small development gain after stronger shared SFT. On the four fixed familiar-schema development probes, GRPO improved1/4→2/4 and tied continued SFT at2/4. Development-only schema success stayed0/4. Clean/silence/mismatch controls on eight dev recordings scored4/8,0/8,0/8. These are tiny constructed-speech development results, not proof of an RL advantage or a completed main study. See [the newer GRPO report](artifacts/grpo_attempt_20261107/overall_report.md) and [generation log](artifacts/grpo_attempt_20261107/strong_init/generations.log).

The measurements below describe the **earlier one-update smoke**, not the subsequent continuation.

The **minimal smoke pipeline works**, with one optimizer update per requested training stage. This is not a completed implementation/validation of the full research specification. Read `artifacts/reports/smoke/research_report.md` and its remaining-integration list before scheduling the main study.

Measured on RTX 4070 Super 12GB, Python 3.12.6, Torch 2.9.1+cu128:

- Native Samsone and Mellow inference ran. Samsone native/common output matched on its fixed example.
- Published Samsone, published Mellow v0 and fresh matched Samsone all completed their staged SFT and GRPO smoke paths.
- Primary fixed training-probe response NLL: 5.6541 → 4.7574 → 3.5277 across two supervised updates. These are teacher-forced diagnostics.
- All three GRPO groups had zero rewards/advantages. This smoke establishes no RL learning benefit.
- Held-out constructed-speech exact success remained zero. The human-speech primary outcome is unavailable.
- Resume restored optimizer/scaler/RNG/sampler state and identical log probabilities without extra training.
- Samsone uses FP16 with FP32 master weights and initial loss scale128. Mellow FP16 scoring was nonfinite; verified BF16 works on this GPU. On a non-BF16 V100 its research path falls back to FP32.

## Setup and first server checks

Copy source/configs/scripts, `uv.lock`, `constraints-linux-py312.txt`, and the artifact manifests. Paths in the tested constraints are relocation-safe. Upstream clones are reconstructed/verified by `scripts/fetch_sources.py`. Download command resolves the pinned model assets into the new machine's cache.

```bash
bash scripts/bootstrap.sh
.venv/bin/nano-audio inspect-env --output artifacts/environment-server.json
.venv/bin/nano-audio download --profile smoke
.venv/bin/nano-audio smoke-text
.venv/bin/nano-audio smoke-audio
.venv/bin/pytest -q
```

For a V100, verify `sm_70` is present in `torch.cuda.get_arch_list()` and run a real backward pass. A CUDA13-capable driver does not establish Volta support in the Torch wheel. Install a tested Volta-capable CUDA12.x Torch/Torchaudio pair in the isolated environment if needed, and freeze that environment separately; do not overwrite a working server installation.

## Next learning run

The user explicitly chose minimal smoke for this execution. The 200-update eight-example diagnostic, 64-update warm-up and 32-update continuations were not run.

```bash
.venv/bin/nano-audio prepare-data --profile local_pilot
.venv/bin/nano-audio audit-data --profile local_pilot
.venv/bin/nano-audio overfit --profile local_pilot --model samsone_134m_published
.venv/bin/nano-audio train --run S_P1 --profile local_pilot --seed 17
.venv/bin/nano-audio train --run S_P_SFT --profile local_pilot --seed 17
.venv/bin/nano-audio train --run S_P_GRPO --profile local_pilot --seed 17
```

If valid/correct sampled calls remain rare, diagnose and extend the **shared training/development warm-up**, version its common checkpoint, and restart both continuation branches. Do not add a format reward or insert gold targets into rollout prompts. Do not claim RL improvement from all-zero advantages or optimizer weight decay.

## Remaining work before full main study

- Integrate local SLURP audio acquisition/annotation mappings into `prepare-data`, with quotas, duplicate exclusions and the declared real/synthetic mixture. Current automatic preparation is explicitly `constructed_only`; human annotations are pinned separately.
- Complete turnkey benchmark downloads and official scorer integration for each paper claim. The current native adapter/claim ledger is not published-score reproduction.
- Complete Audio2Tool native scoring and audited lowercase transfer bindings. Its public tool/label inconsistencies must be excluded or explicitly resolved independently of predictions.
- Complete compute-matched SFT extension, full three-seed paired summary/interval orchestration, and the held-out domain-schema suite.
- Main schema/data/settings must be frozen before final test evaluation. Main continuation seeds are conditional on family initialization seed17.
- Mellow smoke uses a fixed seeded training distractor for the absent second audio; document/version varying per-utterance selection before larger experiments.

`configs/experiments/study.yaml`, the main profile and `scripts/v100.slurm` provide the run matrix and launch interface. The Slurm script requests one GPU,20 CPUs,64GB RAM and six hours; it was not submitted. Treat full main execution as **not yet acceptance-ready** until the integration items above are resolved.
