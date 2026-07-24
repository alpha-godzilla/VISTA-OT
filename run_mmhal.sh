#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export VISTA_LLAVA_MODEL_PATH="${VISTA_LLAVA_MODEL_PATH:-/data/sun_yuxi/models/llava-1.5-7b-hf}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python mmhal_eval.py \
  --model "llava-1.5" \
  --input-path "$SCRIPT_DIR/MMHal-Bench/response_template.json" \
  --data-path "$SCRIPT_DIR/MMHal-Bench/images" \
  --vsv \
  --vsv-lambda 0.1 \
  --logits-aug \
  --logits-alpha 0.3

# Read a generated result file:
# python mmhal_ans.py --response "path/to/result.jsonl"
