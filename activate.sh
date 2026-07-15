#!/bin/bash
# Activate the correct conda environment for GC-MS Consistency project.
# The 'seas' environment has all required packages (numpy, torch, sklearn, etc.)
#
# Usage: source activate.sh

conda activate seas
echo "GC-MS Consistency environment activated: $(which python)"
echo "Python: $(python --version)"
