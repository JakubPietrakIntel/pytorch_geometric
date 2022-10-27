#!/bin/sh

# CPU SPECS - PHYSICAL CORES ONLY
CORES=$(lscpu | grep 'Core(s)' | cut -f2 -d':')
SOCKETS=$(lscpu | grep 'Socket(s)' | cut -f2 -d':')
PHYSICAL_CORES=$((SOCKETS * CORES))
LOGICAL_CORES=$((PHYSICAL_CORES * 2))

echo "PHYSICAL CORES:" $PHYSICAL_CORES
echo "LOGICAL CORES WITH HT:" $LOGICAL_CORES

PYTHON=$(which python)

declare -a HT=(1)
declare -a OMP_SCHEDULE=(0 1)
declare -a OMP_PROC_BIND=(0 1)
declare -a USE_LOGICAL_CORES=(0)

# CPU Affinitization
declare -a COMPUTE_AFFINITY=(0 1 2 3 4) #0-none 1-all avaialable 2-ommit first 3 cores 3-single CPU 4-single CPU with ommit
declare -a DATALOADER_AFFINITY=(1)


# loop variables
declare -a SPARSE_TENSOR=(1)
declare -a NUM_WORKERS=(1 2 4 8 16)
#declare -a MODELS=('gcn') # 'gat' 'rgcn')



# inputs for the script
MODEL='gcn'
DATASET='Reddit'
BATCH_SIZE="512 1024 2048 4096 8192"
NUM_HIDDEN_CHANNELS="128"
NUM_LAYERS="2 3"
HETERO_NEIGHBORS=5
WARMUP=1
iter=0

for ht in ${HT[@]}; do
    if [ $ht = 1 ]; then
        echo on > /sys/devices/system/cpu/smt/control
    else
        echo off > /sys/devices/system/cpu/smt/control
    fi
    for nw in ${NUM_WORKERS[@]}; do
        for caff in ${COMPUTE_AFFINITY[@]}; do
            for dlaff in ${DATALOADER_AFFINITY[@]}; do
                for bind in ${OMP_PROC_BIND[@]}; do
                    for st in ${SPARSE_TENSOR[@]}; do
                        if [ $aff = 1 ] && [ $nw = 0 ]; then
                            continue
                        fi
                        iter=iter+1 # count runs

                        if [ $caff = 1 ]; then
                            # compute aff on all cores
                            lower = 0
                            upper = $((PHYSICAL_CORES - nw - 1))
                        elif [ $caff = 2 ]; then
                            # compute aff on all excluding first 3
                            lower = 3
                            upper = $((PHYSICAL_CORES - nw - 1))
                        elif [ $caff = 3 ]; then
                            # single CPU
                            lower = 0
                            upper = $(((PHYSICAL_CORES / 2) - 1))
                        elif [ $caff = 4 ]; then
                            lower = 3
                            upper = $(((PHYSICAL_CORES / 2) - 1))
                        fi

                        if [ $caff != 0 ]; then
                            export GOMP_CPU_AFFINITY="$(echo $lower-$upper)"
                            export OMP_NUM_THREADS=$((upper - lower + 1))
                        else
                            # no compute aff
                            unset GOMP_CPU_AFFINITY
                            unset OMP_NUM_THREADS
                        fi
                        
                        if [ $bind = 1 ]; then
                            export OMP_PROC_BIND=CLOSE
                        else
                            unset OMP_PROC_BIND
                        fi

                        log="${log_dir}/${iter}_${MODEL}_${DATASET}.log"
                        #touch $log
                        echo "Iteration:" $iter
                        echo "LOG: " $log
                        echo "HYPERTHREADING:" $(cat /sys/devices/system/cpu/smt/active) | tee -a log
                        echo "DL AFFINITY:" $dlaff | tee -a log
                        echo "OMP_NUM_THREADS: " $(echo $OMP_NUM_THREADS) | tee -a log
                        echo "GOMP_CPU_AFFINITY: " $(echo $GOMP_CPU_AFFINITY) | tee -a log
                        echo "OMP_PROC_BIND:" $bind | tee -a log
                        echo "nw: " $nw | tee -a log
                        echo "MODEL: " $MODEL | tee -a log
                        echo "DATASET: " $DATASET | tee -a log
                        echo "SPARSE_TENSOR" $st | tee -a log
                        echo "BATCH_SIZE: " $batch_size | tee -a log
                        echo "LAYER_SIZE: " $NUM_LAYERS  | tee -a log

                        #$PYTHON -u inference_benchmark.py --models $MODEL --datasets $DATASET --num-layers $NUM_LAYERS --num-hidden-channels $NUM_HIDDEN_CHANNELS --hetero-num-neighbors $HETERO_NEIGHBORS --warmup $WARMUP --eval-batch-sizes $BATCH_SIZE --cpu-affinity $dlaff --use-sparse-tensor $st --num-workers $nw --loader-cores $lc --compute-cores $cc | tee -a $log
                    done
                done
            done  
        done
    done
done
