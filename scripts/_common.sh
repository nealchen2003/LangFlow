# Shared settings for the scripts in this directory.
# Usage: source scripts/_common.sh <lm1b|owt> <ar|mdlm|sedd|udlm|duo|langflow>
DATASET=$1
ALGO=$2

case ${DATASET} in
  lm1b) DATA_ARGS="data=lm1b-wrap model.length=128"; STEPS=128 ;;
  owt)  DATA_ARGS="data=openwebtext-split model.length=1024"; STEPS=1024 ;;
  *) echo "Unknown dataset '${DATASET}' (lm1b | owt)"; exit 1 ;;
esac

case ${ALGO} in
  ar)       ALGO_ARGS="algo=ar" ;;
  mdlm)     ALGO_ARGS="algo=mdlm sampling.predictor=ancestral_cache" ;;
  sedd)     ALGO_ARGS="algo=sedd sampling.predictor=analytic" ;;
  udlm)     ALGO_ARGS="algo=udlm sampling.predictor=ancestral" ;;
  duo)      ALGO_ARGS="algo=duo sampling.predictor=ancestral" ;;
  langflow) ALGO_ARGS="algo=langflow" ;;
  *) echo "Unknown algorithm '${ALGO}' (ar | mdlm | sedd | udlm | duo | langflow)"; exit 1 ;;
esac

# Paper settings for LangFlow (preset configs) and the Duo curriculum.
if [[ ${ALGO} == langflow ]]; then
  [[ ${DATASET} == lm1b ]] && CONFIG="-cn langflow-lm1b" || CONFIG="-cn langflow-owt"
  DATA_ARGS=""
fi
if [[ ${ALGO} == duo ]]; then
  [[ ${DATASET} == lm1b ]] && GAMMA="algo.gamma_min=-3.5 algo.gamma_max=-1.75" \
                           || GAMMA="algo.gamma_min=-3.55 algo.gamma_max=-1.85"
  DUO_TRAIN_ARGS="algo.gumbel_tau_log10_start=-3.0 algo.gumbel_tau_log10_end=-3.0 \
    algo.curriculum_start=0 algo.curriculum_end=500000 ${GAMMA}"
fi
