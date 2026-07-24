#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export VISTA_LLAVA_MODEL_PATH="${VISTA_LLAVA_MODEL_PATH:-/data/sun_yuxi/models/llava-1.5-7b-hf}"
COCO_VAL2014_PATH="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}/val2014"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python pope_eval.py \
  --model "llava-1.5" \
  --data-path "$COCO_VAL2014_PATH" \
  --pope-type "random" \
  --vsv \
  --vsv-lambda 0.01 \
  --logits-aug \
  --logits-alpha 0.3

# Read a generated result file:
# python pope_ans.py --ans_file "path/to/result.jsonl"
