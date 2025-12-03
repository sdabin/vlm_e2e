#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
# Usage: ./uniad_dist_eval.sh <config> <checkpoint> <eval_gpus> [vlm_gpus] [extra_args] #
#                                                    #
# Examples:                                          #
#   # 평가 GPU 0,1 / VLM GPU 2,3                     #
#   ./uniad_dist_eval.sh config.py ckpt.pth 0,1 2,3  #
#                                                    #
#   # 평가 GPU 0 / VLM GPU 1                         #
#   ./uniad_dist_eval.sh config.py ckpt.pth 0 1      #
#                                                    #
#   # 100개 샘플만 평가 (quick test)                 #
#   ./uniad_dist_eval.sh config.py ckpt.pth 0 1 --max-samples 100 #
#                                                    #
#   # 결과 파일 위치 지정                            #
#   ./uniad_dist_eval.sh config.py ckpt.pth 0 1 --out /path/to/results.pkl #
# -------------------------------------------------- #
CFG=$1                                               #
CKPT=$2                                              #
EVAL_GPUS=$3                                         #
VLM_GPUS=${4:-""}                                    #
# -------------------------------------------------- #

# 평가 GPU 개수 계산
GPUS=$(echo ${EVAL_GPUS} | tr ',' '\n' | wc -l)

# VLM GPU가 지정된 경우 CUDA_VISIBLE_DEVICES에 포함
if [ -n "${VLM_GPUS}" ]; then
    export CUDA_VISIBLE_DEVICES="${EVAL_GPUS},${VLM_GPUS}"
    echo "Eval GPUs: ${EVAL_GPUS} (${GPUS} GPUs)"
    echo "VLM GPUs: ${VLM_GPUS}"
else
    export CUDA_VISIBLE_DEVICES=${EVAL_GPUS}
    echo "Using GPUs: ${EVAL_GPUS} (${GPUS} GPUs)"
    echo "VLM: disabled or CPU fallback"
fi

GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

# 랜덤 포트 생성 (29500-29999 범위)
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 500))}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Work directory: ${WORK_DIR}"

# 기본 결과 파일 경로 (--out이 extra_args에 없으면 사용)
DEFAULT_OUT="${WORK_DIR}results_${T}.pkl"

# extra_args에 --out이 있는지 확인
EXTRA_ARGS="${@:5}"
if [[ "$EXTRA_ARGS" == *"--out"* ]]; then
    OUT_ARG=""
else
    OUT_ARG="--out ${DEFAULT_OUT}"
fi

echo "Results will be saved to: ${OUT_ARG:-'(specified in extra args)'}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --launcher pytorch ${EXTRA_ARGS} \
    ${OUT_ARG} \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/eval.$T