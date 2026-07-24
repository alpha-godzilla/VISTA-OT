#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export VISTA_LLAVA_MODEL_PATH="${VISTA_LLAVA_MODEL_PATH:-/data/sun_yuxi/models/llava-1.5-7b-hf}"
COCO_VAL2014_PATH="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}/val2014"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python chair_eval.py \
  --model "llava-1.5" \
  --data-path "$COCO_VAL2014_PATH" \
  --vsv \
  --vsv-lambda 0.17 \
  --logits-aug \
  --logits-alpha 0.3

# Read a generated result file:
# python chair_ans.py \
#   --cap_file "path/to/result.jsonl" \
#   --coco_path "${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}/annotations"
