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

# 기본 결과 파일 경로: checkpoint와 같은 폴더에 저장
CKPT_DIR=$(dirname "${CKPT}")
CKPT_NAME=$(basename "${CKPT}" .pth)

# extra_args 파싱
EXTRA_ARGS="${@:5}"

# --max-samples 값 추출
MAX_SAMPLES=""
if [[ "$EXTRA_ARGS" =~ --max-samples[[:space:]]+([0-9]+) ]]; then
    MAX_SAMPLES="${BASH_REMATCH[1]}"
fi

# 테스트 타입에 따른 파일명 설정
if [ -n "$MAX_SAMPLES" ]; then
    TEST_TYPE="quick_test_${MAX_SAMPLES}"
else
    TEST_TYPE="complete_test"
fi

DEFAULT_OUT="${CKPT_DIR}/${CKPT_NAME}_${TEST_TYPE}.pkl"

# extra_args에 --out이 있는지 확인
if [[ "$EXTRA_ARGS" == *"--out"* ]]; then
    OUT_ARG=""
    echo "Results will be saved to: (specified in extra args)"
else
    OUT_ARG="--out ${DEFAULT_OUT}"
    echo "Results will be saved to: ${DEFAULT_OUT}"
fi

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