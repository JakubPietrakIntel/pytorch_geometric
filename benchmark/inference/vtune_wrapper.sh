#!/bin/sh

echo "TEST"
conda activate pyg
conda list

PYTHON=$(which python)
echo $PYTHON
$PYTHON -u inference_benchmark.py