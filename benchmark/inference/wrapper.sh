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
# OMP config variables
declare -a USE_OMP=(0 1)
#declare -a OMP_SCHEDULE=(0 1)
declare -a OMP_PROC_BIND=(0)
declare -a USE_LOGICAL_CORES=(0)
declare -a SEPARATION = (0 1)
# CPU Affinitization
declare -a GOMP_CPU_AFFINITY=(0 1)

# loop variables
declare -a SPARSE_TENSOR=(0 1)
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
    for aff in ${GOMP_CPU_AFFINITY[@]}; do
        for lgc in ${USE_LOGICAL_CORES[@]}; do
            for bind in ${OMP_PROC_BIND[@]}; do
                for st in ${SPARSE_TENSOR[@]}; do
                    for sep in ${SEPARATION[@]}; do
                        if [ $aff = 1 ] && [ $nr_workers = 0 ]; then
                            continue
                        fi
                        iter=iter+1 # count runs

                        if [ $lgc = 1 ]; then
                            export NUMEXPR_MAX_THREADS=$LOGICAL_CORES
                            export OMP_NUM_THREADS=$((LOGICAL_CORES - nr_workers))
                        else
                            if [ $sep = 1 ]; then
                                export OMP_NUM_THREADS=$CORES
                            else 
                                export OMP_NUM_THREADS=$((PHYSICAL_CORES - nr_workers))
                            fi
                        fi

                        if [ $aff = 1 ]; then
                            if [ $sep = 1 ]; then
                                lower=$((PHYSICAL_CORES / 2))
                            else
                                lower=$nr_workers
                            fi
                            upper=$((PHYSICAL_CORES - 1))
                            export GOMP_CPU_AFFINITY="$(echo $lower-$upper)"
                        else
                            unset GOMP_CPU_AFFINITY
                        fi
                        
                        if [ $bind = 1 ]; then
                            export OMP_PROC_BIND=CLOSE
                        else
                            unset OMP_PROC_BIND
                        fi

                        log="${log_dir}/${iter}_${MODEL}_${DATASET}.log"
                        touch $log
                        echo "Iteration:" $iter
                        echo "LOG: " $log
                        echo "HYPERTHREADING:" $(cat /sys/devices/system/cpu/smt/active) | tee -a log
                        echo "AFFINITY:" $aff | tee -a log
                        echo "OMP_NUM_THREADS: " $(echo $OMP_NUM_THREADS) | tee -a log
                        echo "GOMP_CPU_AFFINITY: " $(echo $GOMP_CPU_AFFINITY) | tee -a log
                        echo "OMP_PROC_BIND:" $bind | tee -a log
                        echo "NR_WORKERS: " $nr_workers | tee -a log
                        echo "MODEL: " $MODEL | tee -a log
                        echo "DATASET: " $DATASET | tee -a log
                        echo "SPARSE_TENSOR" $st | tee -a log
                        echo "BATCH_SIZE: " $batch_size | tee -a log
                        echo "LAYER_SIZE: " $NUM_LAYERS  | tee -a log

                    
                        #$PYTHON -u inference_benchmark.py --models $MODEL --datasets $DATASET --num-workers $NUM_WORKERS --num-layers $NUM_LAYERS --num-hidden-channels $NUM_HIDDEN_CHANNELS --hetero-num-neighbors $HETERO_NEIGHBORS --warmup $WARMUP --cpu_affinity $aff --use-sparse-tensor $st --eval-batch-sizes $BATCH_SIZE | tee -a $log
                    done
                done
            done  
        done
    done
done