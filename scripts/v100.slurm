#!/bin/bash
#SBATCH --job-name=nano-audio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
cd "${NANO_AUDIO_ROOT:-$SLURM_SUBMIT_DIR}"
# Use a tested CUDA 12.x build that includes sm_70; driver banner is insufficient.
.venv/bin/python -c 'import torch; assert "sm_70" in torch.cuda.get_arch_list(), "Install a Volta-capable Torch CUDA12 build"'
exec .venv/bin/nano-audio run-study --profile main --device cuda
