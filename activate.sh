#!/bin/bash
# Activate the correct conda environment for GC-MS Consistency project.
# The 'seas' environment has all required packages (numpy, torch, sklearn, etc.)
#
# Usage: source activate.sh

GCMS_CONDA_ENV="${GCMS_CONDA_ENV:-seas}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found; initialize Conda before sourcing activate.sh" >&2
    return 1 2>/dev/null || exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${GCMS_CONDA_ENV}"

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "GC-MS Consistency environment activated: $(command -v python)"
echo "Python: $(python --version)"
