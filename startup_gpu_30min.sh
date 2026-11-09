#!/usr/bin/env bash

# When run as a script, request a GPU and use this file to initialize Bash.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    script=$(realpath -- "${BASH_SOURCE[0]}") || exit 1
    exec srun \
        --partition=UGGPU-TC1 \
        --gres=gpu:1 \
        --ntasks=1 \
        --time=01:30:00 \
        --pty bash --rcfile "$script" -i
fi

# This section runs inside the interactive shell on the compute node.
if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh || exit 1
fi

module load slurm uv anaconda cuda/13.0 || exit 1
source /tc1apps/anaconda3/etc/profile.d/conda.sh || exit 1
conda activate "$HOME/.conda/envs/sc4001" || exit 1

# Keep the Conda interpreter ahead of the cluster Python modules.
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
hash -r

printf 'Node: %s\nPython: %s\n' "$(hostname)" "$(command -v python)"
