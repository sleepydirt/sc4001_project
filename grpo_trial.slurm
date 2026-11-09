#!/bin/bash -l
#SBATCH --job-name=grpo_trial
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail
cd "${NANO_AUDIO_ROOT:-$SLURM_SUBMIT_DIR}"

module load slurm anaconda cuda/13.0
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/sc4001"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

set -u
run="${1:-S_P_GRPO}"
case "$run" in
    S_P_GRPO|S_P_SFT) default_updates=512 ;;
    S_P1) default_updates=64 ;;
    *) echo "Usage: sbatch $0 [S_P1|S_P_GRPO|S_P_SFT] [updates]" >&2; exit 2 ;;
esac
updates="${2:-$default_updates}"

if [[ ! -s artifacts/data/local_pilot/examples.jsonl || ! -s artifacts/data/local_pilot/audit.json ]]; then
    echo "Missing local_pilot examples or audit. Submit from the prepared project root." >&2
    exit 1
fi

if [[ "$run" != S_P1 && ! -e artifacts/runs/local_pilot/S_P1/seed_17/checkpoint/policy ]]; then
    echo "Missing shared S_P1 checkpoint. Run: sbatch grpo_trial.slurm S_P1" >&2
    exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-team88_sc4001}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$PWD/artifacts/wandb/$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1
mkdir -p "$WANDB_DIR"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

import torch
import wandb
from torchcodec.decoders import AudioDecoder

audit = json.loads(Path("artifacts/data/local_pilot/audit.json").read_text())
if audit["status"] != "passed" or audit["groups"] != 448:
    raise SystemExit("Expected a passed local_pilot audit with 448 groups.")

x = torch.ones(16, device="cuda", dtype=torch.float16, requires_grad=True)
x.square().sum().backward()
torch.cuda.synchronize()
print(f"GPU: {torch.cuda.get_device_name(0)} | Torch: {torch.__version__} | W&B: {wandb.__version__}")
PY

# Replace the batch shell so its early USR1 signal reaches the trainer.
exec .venv/bin/nano-audio train \
    --run "$run" --profile local_pilot --seed 17 --device cuda --updates "$updates"

