#!/bin/bash
# Perplexity on the validation set (upper bound for diffusion models).
# Usage: bash scripts/eval_ppl.sh <lm1b|owt> <algo> <checkpoint> [hydra overrides...]
# LangFlow: eval.ppl_bound=ode (default) | sde; for the ODE bound,
# eval.self_cond=attached (default) | detached | off and
# eval.ode_steps / eval.ode_method (default: 128-step heun2).
# Uses all visible GPUs; add trainer.num_nodes=<N> for multi-node runs.
source "$(dirname "$0")/_common.sh" "$1" "$2"
CHECKPOINT=$3
shift 3
BATCH_SIZE=${BATCH_SIZE:-16}

python -u main.py ${CONFIG} \
  mode=ppl_eval \
  ${ALGO_ARGS} ${DATA_ARGS} \
  model=small \
  eval.checkpoint_path=${CHECKPOINT} \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${BATCH_SIZE} \
  wandb.mode=disabled \
  "$@"
