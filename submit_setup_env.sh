#!/usr/bin/env bash
# submit_setup_env.sh — One-time job to create the cl_exp conda environment on the cluster.
# Run once before any training jobs.
# Monitor: runai logs -f cl-setup-env

set -euo pipefail

JOB_NAME="cl-setup-env"
IMAGE="noyhassid/spectralfm-lean:v6"
CODE_DIR="/storage/noy/CL_experiment"

runai delete job "${JOB_NAME}" -p raja 2>/dev/null || true

runai submit "${JOB_NAME}" \
  --project raja \
  --image "${IMAGE}" \
  --cpu 4 \
  --memory 16G \
  --node-pools faculty,raja \
  --existing-pvc "claimname=storage,path=/storage" \
  -- bash -c "
    if [ ! -d ${CODE_DIR} ]; then
      git clone https://github.com/noytau/CL_experiment.git ${CODE_DIR};
    else
      cd ${CODE_DIR} && git pull;
    fi &&
    conda env remove -n cl_exp -y 2>/dev/null || true &&
    conda env create -f ${CODE_DIR}/environment.yml &&
    conda run -n cl_exp python -c 'import torch, transformers, datasets, jiwer, torchcodec; print(\"Environment OK\")'
  "

echo ""
echo "Submitted: ${JOB_NAME}"
echo "Monitor:   runai logs -f ${JOB_NAME}"
