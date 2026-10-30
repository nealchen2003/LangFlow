#!/bin/bash
# Train LangFlow or a baseline for 1M steps with a global batch size of 512.
# Usage: bash scripts/train.sh <lm1b|owt> <ar|mdlm|sedd|udlm|duo|langflow> [hydra overrides...]
# BATCH_SIZE is the per-GPU batch size; gradients are accumulated to 512.
source "$(dirname "$0")/_common.sh" "$1" "$2"
shift 2
[[ ${DATASET} == lm1b ]] && BATCH_SIZE=${BATCH_SIZE:-64} || BATCH_SIZE=${BATCH_SIZE:-16}

python -u main.py ${CONFIG} \
  ${ALGO_ARGS} ${DATA_ARGS} ${DUO_TRAIN_ARGS} \
  model=small \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${BATCH_SIZE} \
  wandb.name=${ALGO}-${DATASET} \
  "$@"
