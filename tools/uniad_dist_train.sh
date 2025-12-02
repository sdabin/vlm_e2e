#!/usr/bin/env bash

T=`date +%m%d%H%M`
DATE_STR=`date +%Y%m%d_%H%M`
# -------------------------------------------------- #
# Usually you only need to customize these variables #
# Usage: ./uniad_dist_train.sh <config> <gpu_indices> #
# Example: ./uniad_dist_train.sh config.py 0,1,2,3   #
#          ./uniad_dist_train.sh config.py 4,5       #
CFG=$1                                               #
GPU_IDS=$2                                           #
# -------------------------------------------------- #

# GPU 인덱스를 CUDA_VISIBLE_DEVICES로 설정
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# 콤마로 구분된 GPU 개수 계산
GPUS=$(echo ${GPU_IDS} | tr ',' '\n' | wc -l)
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=`expr $GPUS / $GPUS_PER_NODE`

# 랜덤 포트 생성 (29500-29999 범위)
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 500))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

BASE_WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")
WORK_DIR=${BASE_WORK_DIR}_${DATE_STR}/

# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

echo "Using GPUs: ${GPU_IDS} (${GPUS} GPUs)"
echo "Work directory: ${WORK_DIR}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    $(dirname "$0")/train.py \
    $CFG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.$T