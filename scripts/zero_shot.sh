#!/bin/bash
# Zero-shot perplexities of a model trained on OpenWebText.
# Usage: bash scripts/zero_shot.sh <algo> <checkpoint> [hydra overrides...]
ALGO=$1
CHECKPOINT=$2
shift 2

for data in ag_news scientific_papers_pubmed scientific_papers_arxiv lambada \
            wikitext2 wikitext103 ptb lm1b-gpt2; do
  echo "${data}"
  bash "$(dirname "$0")/eval_ppl.sh" owt ${ALGO} ${CHECKPOINT} \
    data=${data} data.insert_valid_eos=False loader.eval_global_batch_size=128 "$@"
done
