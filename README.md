# Learning Dynamics of Nano Speech Language Models on Tool Calling

Supervised and reinforcement post-training of Samsone-134M and Mellow on Audio2Tool. This repository contains all training code and environment dependencies needed to replicate all the results in the paper. Note that the environment is configured to work on a NVIDIA Tesla V100.

## Setup

Use Linux, Python 3.12, Git LFS and an NVIDIA GPU. Model downloads and dataset preparation require network access and substantial disk space.

```bash
uv sync --frozen --extra mellow --extra dev
uv pip install --python .venv/bin/python -r requirements-report.txt wandb==0.23.0
source .venv/bin/activate
export AUDIO2TOOL_STUDY="$PWD/artifacts/replication_data"
python -m nano_learning.workflow prepare
```

Preparation downloads the two models and Audio2Tool, verifies audio, profiles model interfaces, applies registry/tokenization/audio-length filters, and constructs grouped train/validation/test splits. The protocol caps context at 8,192 tokens and completions at 512 tokens. Keep `AUDIO2TOOL_STUDY` set to the same directory for subsequent commands.

## Test and train

```bash
python -m nano_learning.workflow test
python -m nano_learning.workflow train --runs artifacts/replication_runs
```

The test command enables the real-model integration tests. Training runs six 640-update warmups, selects each model's step-640 parent, runs independent +64 SFT/binary-GRPO/typed-GRPO branches, then the +1024 comparisons and Samsone SFT extension to +4096. Mellow has SFT only at +1024. 

Repeat the training command to resume interrupted runs. W&B defaults to offline logging; set `WANDB_MODE=online` and authenticate with `wandb login` to enable online tracking.

Individual stages remain available through `python -m nano_learning.training --help`.

## Evaluate and report

```bash
python -m nano_learning.workflow evaluate --runs artifacts/replication_runs
```

This evaluates completed endpoints on full validation, prints macro/pooled exact accuracy, argument F1 and schema validity, and writes `artifacts/replication_runs/results.csv`. Each run retains predictions, training logs and validation measurements. Accuracy values in the CSV are fractions.

For a specific checkpoint or held-out test evaluation, use `python -m nano_learning.inference --help`. Freeze checkpoint selection with `python -m nano_learning.lock_evaluation --help` before test evaluation.

The paper notebook can regenerate the original study's figures from its saved local results:

```bash
python -m nano_learning.notebook
```

New-run results are reported by the workflow command above; the historical notebook uses the original study layout. Models, raw data, environments and training outputs are Git-ignored.

## Replicate individual paper tables

Run Setup first and keep `AUDIO2TOOL_STUDY` set. Commands below use the same `artifacts/replication_runs` directory, so completed training stages can be reused. New training reproduces the protocol; stochastic scores and hardware-dependent times may differ from the paper. Exported CSV metrics are fractions; multiply by 100 for percentage columns.

### Section 4.2 — Dataset (Tables 2–4)

The source overview, filtered cohort and grouped splits correspond to `tab:audio2tool-hours`, `tab:audio2tool-filtered` and `tab:audio2tool-grouped-split`.

```bash
python -m nano_learning.workflow prepare
python -m nano_learning.paper_tables dataset
```

Outputs: `paper/replication/tables/dataset.csv`, `splits.csv` and matching LaTeX fragments. Dataset counts distinguish references, distinct WAVs and request groups.

### Section 5.1 — Design (Table 5: trainable parameters)

```bash
python -m nano_learning.paper_tables parameters
```

Output: `paper/replication/parameters.json`, containing each model's parameter manifest. This loads the prepared models without training.

### Section 5.2 — Supervised warm-up (Tables 6–7)

```bash
python -m nano_learning.workflow warmup
python -m nano_learning.paper_tables dataset
python -m nano_learning.paper_tables warmup
```

The first command runs both models at 10%, 30% and 100% training pools for 640 updates and selects the step-640 parents. Table 6 pool counts are written during preparation to `$AUDIO2TOOL_STUDY/train_subset_coverage.json`; Table 7 endpoint scores are exported to `paper/replication/warmup.csv`. Full-validation measurements at updates 256, 512 and 640 remain in each run's `validation/` directory.

### Section 5.3 — Learning dynamics of SFT against RL (Table 8: +64 pilot)

Run the warm-up commands above first.

```bash
python -m nano_learning.workflow pilot64
python -m nano_learning.paper_tables pilot64
```

The export command freezes the parent and three branch endpoints for each model, evaluates them on the held-out test split, and writes `paper/replication/pilot64.csv` plus predictions. An existing test selection must contain those checkpoints; use a separate prepared study directory for an independent experiment.

### Section 5.3.1 — Extensions (Table 9: +1024 and SFT +4096)

Run the warm-up commands first. The +64 stage is not a checkpoint prerequisite for these independent continuations.

```bash
python -m nano_learning.workflow continuation1024
python -m nano_learning.workflow sft4096
python -m nano_learning.paper_tables extensions
```

Output: `paper/replication/extensions.csv` with full-validation endpoint metrics. Training durations are the sum of `elapsed_seconds` in each run's `learning.jsonl`; the +4096 total includes both the +1024 SFT and subsequent 3072-update segments. These timings exclude loading, evaluation and checkpoint I/O and depend on the GPU used.

Table 1 in Section 2.3 contains analytical weight-memory estimates, rather than measured training results. Its ideal storage calculation is parameter count × bytes per parameter; peak inference memory requires a separate device measurement.
