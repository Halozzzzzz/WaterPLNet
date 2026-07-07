#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-0,1}"
NUM_GPUS=$(python - <<'PY'
import os
print(len([x for x in os.environ.get("GPUS", "0,1").split(",") if x.strip()]))
PY
)

export CUDA_VISIBLE_DEVICES="${GPUS}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT:-29500}" \
  run/train.py \
  --model_name WaterPLNet \
  --backbone resnet18 \
  "$@"
