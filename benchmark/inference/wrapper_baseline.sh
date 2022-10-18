#!/bin/sh

# CPU SPECS - PHYSICAL CORES ONLY
CORES=$(lscpu | grep 'Core(s)' | cut -f2 -d':')
SOCKETS=$(lscpu | grep 'Socket(s)' | cut -f2 -d':')
TOTAL_CORES=$((SOCKETS * CORES))
echo "TOTAL_CORES:" $TOTAL_CORES

PYTHON=$(which python)

# Hyperthreading
declare -a HT=(0 1)
# OMP config variables
declare -a USE_OMP=(0 1)
declare -a OMP_SCHEDULE=(0 1)
declare -a OMP_PROC_BIND=(0 1)
# Affinitization - use together with GOMP_CPU_AFFINITY
declare -a AFFINITY=(0 1)

# loop variables
declare -a SPARSE_TENSOR=(0 1)
declare -a NUM_WORKERS=(0 1 2 4 8 16)
declare -a MODELS=('gcn') # 'gat' 'rgcn')

# inputs for the script
BATCH_SIZE="512 1024 2048 4096 8192"
NUM_HIDDEN_CHANNELS="64 128 256"
NUM_LAYERS="2 3"
HETERO_NEIGHBORS=5
WARMUP=1
# for each model run benchmark in 4 configs: NO_HT+NO_AFF, NO_HT+AFF, HT+NO_AFF, HT+AFF
for model in ${MODELS[@]}; do
    for omp in ${USE_OMP[@]}; do
        log_dir="logs/no_omp"
        mkdir -p "${log_dir}"
        if [ $omp = 0 ]; then
            for st in ${SPARSE_TENSOR[@]}; do
                
                for ht in ${HT[@]}; do
                    if [ $ht = 1 ]; then
                        echo on > /sys/devices/system/cpu/smt/control
                    else
                        echo off > /sys/devices/system/cpu/smt/control
                    fi
                    WORKERS="0 1 2 4 8 16"
                    unset GOMP_CPU_AFFINITY
                    unset OMP_NUM_THREADS
                    unset OMP_SCHEDULE
                    unset OMP_PROC_BIND
                    log="${log_dir}/BASELINE_${model}HT${ht}ST${st}.log"

                    echo "OMP:" $omp
                    echo "HYPERTHREADING:" $(cat /sys/devices/system/cpu/smt/active)
                    echo "SPARSE_TENSOR:" $st
                    echo "LOG: " $log
                    $PYTHON -u inference_benchmark.py --models $model --num-workers $WORKERS --num-layers $NUM_LAYERS --num-hidden-channels $NUM_HIDDEN_CHANNELS --hetero-num-neighbors $HETERO_NEIGHBORS --eval-batch-sizes $BATCH_SIZE --warmup $WARMUP --cpu_affinity $aff --use-sparse-tensor $st | tee $log
                done
            done    
        fi
    done
done