#!/usr/bin/env bash

T=`date +%m%d%H%M`
DATE_STR=`date +%Y%m%d_%H%M`
# -------------------------------------------------- #
# Usually you only need to customize these variables #
# Usage: ./uniad_dist_train.sh <config> <train_gpus> [vlm_gpus] #
#                                                    #
# Examples:                                          #
#   # 학습 GPU 0,1 / VLM 자동 선택 (2 또는 3)        #
#   ./uniad_dist_train.sh config.py 0,1 2,3          #
#                                                    #
#   # 학습 GPU 2 / VLM GPU 0,1 중 자동 선택          #
#   ./uniad_dist_train.sh config.py 2 0,1            #
#                                                    #
#   # VLM 없이 학습만 (기존 방식)                    #
#   ./uniad_dist_train.sh config.py 0,1              #
# -------------------------------------------------- #
CFG=$1                                               #
TRAIN_GPUS=$2                                        #
VLM_GPUS=${3:-""}                                    #
# -------------------------------------------------- #

# 학습 GPU 개수 계산
GPUS=$(echo ${TRAIN_GPUS} | tr ',' '\n' | wc -l)

# VLM GPU가 지정된 경우 CUDA_VISIBLE_DEVICES에 포함
if [ -n "${VLM_GPUS}" ]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS},${VLM_GPUS}"
    echo "Training GPUs: ${TRAIN_GPUS} (${GPUS} GPUs)"
    echo "VLM GPUs: ${VLM_GPUS} (auto-select from these)"
else
    export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}
    echo "Using GPUs: ${TRAIN_GPUS} (${GPUS} GPUs)"
    echo "VLM: disabled or CPU fallback"
fi

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

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
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
    --launcher pytorch ${@:4} \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.$T