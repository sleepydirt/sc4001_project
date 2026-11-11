#!/bin/bash -l
#SBATCH --job-name=samsone-semi
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
export PYTHONUNBUFFERED=1

set -u
if (( $# < 2 )); then
    echo "Usage: sbatch semi_pilot/run.slurm STAGE EXPERIMENT [stage flags]" >&2
    exit 2
fi
stage="$1"
experiment="$2"
shift 2
case "$stage" in
    prepare|overfit|warmup|gate|sft|grpo|test) ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac

export WANDB_PROJECT="${WANDB_PROJECT:-team88_sc4001}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$PWD/artifacts/wandb/${SLURM_JOB_ID:-local}}"

.venv/bin/python - <<'PY'
import torch
from torchcodec.decoders import AudioDecoder

device = torch.cuda.get_device_name(0)
if "V100" in device and "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit("The Torch wheel must include sm_70 for a V100.")
x = torch.ones(16, device="cuda", dtype=torch.float16, requires_grad=True)
x.square().sum().backward()
torch.cuda.synchronize()
print(f"GPU: {device} | Torch: {torch.__version__}", flush=True)
PY

# exec delivers Slurm's early signal directly to the training process.
case "$stage" in
    prepare)
        exec .venv/bin/python semi_pilot/data.py prepare --experiment "$experiment" --device cuda "$@"
        ;;
    overfit|warmup|gate|sft|grpo)
        exec .venv/bin/python semi_pilot/wandb_log.py job "$stage" "$experiment" --device cuda "$@"
        ;;
    test)
        exec .venv/bin/python semi_pilot/run.py "$stage" --experiment "$experiment" --device cuda "$@"
        ;;
esac
