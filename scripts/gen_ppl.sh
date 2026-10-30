#!/bin/bash
# Generate 1024 samples and report Gen. PPL (GPT-2 Large) and entropy.
# Usage: bash scripts/gen_ppl.sh <lm1b|owt> <algo> <checkpoint> [hydra overrides...]
# <checkpoint>: a .ckpt file; for LangFlow also the released HuggingFace weights
# (model.safetensors, a directory or a Hub repo id).
# LangFlow samplers: sampling.predictor=ode|sde, sampling.temperature=<T>,
# sampling.sde_offset=<offset>, sampling.sde_mode=elf|edm.
source "$(dirname "$0")/_common.sh" "$1" "$2"
CHECKPOINT=$3
shift 3
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}

python -u main.py ${CONFIG} \
  mode=sample_eval \
  ${ALGO_ARGS} ${DATA_ARGS} \
  model=small \
  eval.checkpoint_path=${CHECKPOINT} \
  loader.eval_batch_size=${EVAL_BATCH_SIZE} \
  sampling.steps=${STEPS} \
  sampling.num_sample_batches=$((1024 / EVAL_BATCH_SIZE)) \
  eval.generated_samples_path=${PWD}/samples_${ALGO}_${DATASET}.json \
  wandb.mode=disabled \
  "$@"
