#!/bin/sh

# CPU SPECS - PHYSICAL CORES ONLY
CORES=$(lscpu | grep 'Core(s)' | cut -f2 -d':')
SOCKETS=$(lscpu | grep 'Socket(s)' | cut -f2 -d':')
PHYSICAL_CORES=$((SOCKETS * CORES))
LOGICAL_CORES=$((PHYSICAL_CORES * 2))

echo "PHYSICAL CORES:" $PHYSICAL_CORES
echo "LOGICAL CORES:" $LOGICAL_CORES

PYTHON=$(which python)

declare -a HYPERTHREADING=(1)
declare -a OMP_PROC_BIND=(0 1)
#declare -a USE_LOGICAL_CORES=(0)
# CPU Affinitization
declare -a COMPUTE_AFFINITY=(0 1 2 3 4) #0-none 1-all avaialable 2-ommit first 3 cores 3-single CPU 4-single CPU with ommit
declare -a DATALOADER_AFFINITY=(1)
declare -a NUM_WORKERS=(1 2 4 8 16)

#declare -a MODELS=('gcn') # 'gat' 'rgcn')



# inputs for the script
MODEL='gcn'
DATASET='ogbn-products'
BATCH_SIZE="512 1024 2048 4096 8192"
NUM_HIDDEN_CHANNELS="128"
NUM_LAYERS="2 3"
HETERO_NEIGHBORS=5
WARMUP=1
SPARSE_TENSOR=1
DL_AFFINITY=1
iter=0

for ht in ${HYPERTHREADING[@]}; do
    if [ $ht = 1 ]; then
        echo on > /sys/devices/system/cpu/smt/control
    else
        echo off > /sys/devices/system/cpu/smt/control
    fi
    for ob in ${OMP_PROC_BIND[@]}; do
        for nw in ${NUM_WORKERS[@]}; do
            for caff in ${COMPUTE_AFFINITY[@]}; do
                if [ $nw = 0 ] && [ $DL_AFFINITY = 1 ]; then
                    continue
                fi
                iter=$((iter + 1)) # count runs
                if [ $caff != 0 ]; then
                    if [ $caff = 1 ]; then
                        # compute aff on all cores
                        lower=0
                        upper=$((PHYSICAL_CORES - nw - 1))
                    elif [ $caff = 2 ]; then
                        # compute aff on all excluding first 3
                        lower=3
                        upper=$((PHYSICAL_CORES - nw - 1))
                    elif [ $caff = 3 ]; then
                        # single CPU
                        lower=0
                        upper=$(((PHYSICAL_CORES / 2) - 1))
                    elif [ $caff = 4 ]; then
                        lower=3
                        upper=$(((PHYSICAL_CORES / 2) - 1))
                    fi

                    export GOMP_CPU_AFFINITY="$(echo $lower-$upper)"
                    export OMP_NUM_THREADS=$((upper - lower + 1))

                    if [ $ob = 1 ]; then
                        export OMP_PROC_BIND=CLOSE
                    else
                        unset OMP_PROC_BIND
                    fi

                else
                    # no compute aff
                    unset GOMP_CPU_AFFINITY
                    unset OMP_NUM_THREADS
                    unset OMP_PROC_BIND
                fi
                mkdir -p logs
                log="logs/${iter}_${MODEL}_${DATASET}.log"
                touch $log
                echo "----------------------"
                echo """ Iteration: $iter
                LOG: $log
                MODEL: $MODEL
                DATASET: $DATASET
                SPARSE_TENSOR: $SPARSE_TENSOR
                BATCH_SIZE: $BATCH_SIZE
                LAYER_SIZE: $NUM_LAYERS
                NUM_WORKERS: $nw
                HYPERTHREADING: $(cat /sys/devices/system/cpu/smt/active)
                DL AFFINITY: $DL_AFFINITY
                OMP_NUM_THREADS: $(echo $OMP_NUM_THREADS)
                GOMP_CPU_AFFINITY: $(echo $GOMP_CPU_AFFINITY)
                OMP_PROC_BIND: $(echo $OMP_PROC_BIND)
                """ | tee -a $log

                $PYTHON -u inference_benchmark.py --models $MODEL --datasets $DATASET --num-layers $NUM_LAYERS --num-hidden-channels $NUM_HIDDEN_CHANNELS --hetero-num-neighbors $HETERO_NEIGHBORS --warmup $WARMUP --use-sparse-tensor $SPARSE_TENSOR --eval-batch-sizes $BATCH_SIZE --cpu-affinity $DL_AFFINITY --num-workers $nw | tee -a $log
                # --loader-cores $lc --compute-cores $cc
            done
        done  
    done
done
echo "BENCHMARK FINISHED"
