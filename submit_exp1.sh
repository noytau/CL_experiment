#!/usr/bin/env bash
# submit_exp1.sh — Experiment 1: train CTC head on frozen data2vec (LibriSpeech clean)
# Usage: bash submit_exp1.sh
# Monitor: runai logs -f cl-exp1-head-training

set -euo pipefail

JOB_NAME="cl-exp1-head-training"
IMAGE="noyhassid/spectralfm-lean:v6"
CODE_DIR="/storage/noy/CL_experiment"
CKPT_DIR="/storage/noy/CL_experiment/checkpoints/exp1"
HF_HOME="/storage/noy/hf_cache"

runai submit "${JOB_NAME}" \
  --project raja \
  --image "${IMAGE}" \
  --gpu-memory 40G \
  --node-pools faculty,raja \
  --existing-pvc "claimname=storage,path=/storage" \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  -e HF_HOME="${HF_HOME}" \
  -- bash -c "
    if [ ! -d ${CODE_DIR} ]; then
      git clone https://github.com/noytau/CL_experiment.git ${CODE_DIR};
    else
      cd ${CODE_DIR} && git pull;
    fi &&
    cd ${CODE_DIR} &&
    conda run -n cl_exp python ${CODE_DIR}/train_head.py \
      --n_train    500 \
      --n_val      100 \
      --batch_size 8   \
      --n_epochs   20  \
      --lr         1e-3 \
      --save_dir   ${CKPT_DIR} \
      --hf_home    ${HF_HOME}
  "

echo ""
echo "Submitted: ${JOB_NAME}"
echo "Monitor:   runai logs -f ${JOB_NAME}"
echo "Status:    runai list jobs"
