#!/bin/sh

RUN_DIR="${HOME}/pyg-jpietrak/pytorch_geometric/benchmark/inference"
CONDA_ENV="pyg-jpietrak"
PYTHON=$(which python)

MODEL='gcn'
DATASET='ogbn-products'
NUM_WORKERS=2
NUM_LAYERS=3
NUM_HIDDEN_CHANNELS=128
HETERO_NEIGHBORS=5
WARMUP=1
AFF=0
ST=1
BATCH_SIZE=1024

RUN_CMD="python -u inference_benchmark.py --models ${MODEL} --datasets ${DATASET} --num-workers ${NUM_WORKERS} --num-layers ${NUM_LAYERS} --num-hidden-channels ${NUM_HIDDEN_CHANNELS} --hetero-num-neighbors ${HETERO_NEIGHBORS} --warmup ${WARMUP} --cpu_affinity ${AFF} --use-sparse-tensor ${ST} --eval-batch-sizes ${BATCH_SIZE}"

echo -n '===== VTune Being Used =====: '; vtune --version
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

#export ...

cd $RUN_DIR
vtune -collect uarch-exploration -- ${RUN_CMD} # Run VTune
echo "Results can be found in ${PWD}”
