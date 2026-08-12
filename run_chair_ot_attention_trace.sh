#!/usr/bin/env bash
# Diagnose whether attention-OT under-allocates visual mass to COCO objects.
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 COCO_IMAGE_ID" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_ID="$1"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-llava-1.5}"
VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
LOGITS_ALPHA="${LOGITS_ALPHA:-0.3}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
TRACE_DIR="${TRACE_DIR:-$SCRIPT_DIR/exp_results/attention_ot_trace_${IMAGE_ID}}"
EXP_FOLDER="${EXP_FOLDER:-attention_ot_trace_${IMAGE_ID}}"

mkdir -p "$TRACE_DIR"
IDS_FILE="$TRACE_DIR/image_ids.txt"
printf '%s\n' "$IMAGE_ID" > "$IDS_FILE"

cd "$SCRIPT_DIR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" chair_eval.py \
  --exp_folder "$EXP_FOLDER" --model "$MODEL" --data-path "$VISTA_COCO_ROOT/val2014" \
  --subset-size 1 --subset-ids-file "$IDS_FILE" --seed 2024 --max-new-tokens 512 \
  --vsv --vsv-lambda "$VSV_LAMBDA" --logits-aug --logits-layers "$LOGITS_LAYERS" --logits-alpha "$LOGITS_ALPHA" \
  --use-ot-bary-sla --ot-attention-visual-marginal --ot-topk 16 \
  --ot-sinkhorn-iters 50 --ot-sinkhorn-tolerance 0.001 --ot-epsilon 0.05 \
  --ot-layer-temperature 0.06 --ot-attention-power 0.75 --ot-attention-uniform-mix 0.02 \
  --ot-log-stats --ot-attention-trace

TRACE_JSONL="$(find "$SCRIPT_DIR/exp_results/$EXP_FOLDER/$MODEL" -name '*_ot_stats.jsonl' -print -quit)"
[[ -n "$TRACE_JSONL" ]] || { echo "OT trace JSONL was not written" >&2; exit 1; }
"$PYTHON_BIN" scripts/analyze_attention_ot_trace.py \
  --trace-jsonl "$TRACE_JSONL" --annotations "$VISTA_COCO_ROOT/annotations/instances_val2014.json" \
  --image-id "$IMAGE_ID" --output-json "$TRACE_DIR/coverage.json" \
  --output-markdown "$TRACE_DIR/coverage.md"
echo "Diagnosis: $TRACE_DIR/coverage.md"
