#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${MODE:-ot}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXP_FOLDER="${EXP_FOLDER:-chair_ot_bary}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
LOGITS_ALPHA="${LOGITS_ALPHA:-0.3}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"

export VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
export NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"

if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then
  for candidate in \
    /data/sun_yuxi/models/llava-v1.5-7b \
    /data/sun_yuxi/models/llava-1.5-7b-hf; do
    if [[ -d "$candidate" ]]; then
      export VISTA_LLAVA_MODEL_PATH="$candidate"
      break
    fi
  done
fi

if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then
  echo "Set VISTA_LLAVA_MODEL_PATH to the LLaVA-1.5 model directory." >&2
  exit 1
fi
if [[ ! -f "$VISTA_LLAVA_MODEL_PATH/config.json" ]]; then
  echo "Model config not found: $VISTA_LLAVA_MODEL_PATH/config.json" >&2
  exit 1
fi
if [[ ! -d "$VISTA_COCO_ROOT/val2014" ]]; then
  echo "COCO val2014 directory not found: $VISTA_COCO_ROOT/val2014" >&2
  exit 1
fi

common_args=(
  --exp_folder "$EXP_FOLDER"
  --model llava-1.5
  --data-path "$VISTA_COCO_ROOT/val2014"
  --subset-size "$SUBSET_SIZE"
  --vsv
  --vsv-lambda "$VSV_LAMBDA"
  --logits-aug
  --logits-layers "$LOGITS_LAYERS"
  --logits-alpha "$LOGITS_ALPHA"
)

case "$MODE" in
  original)
    method_args=()
    ;;
  ot)
    method_args=(
      --use-ot-bary-sla
      --ot-topk "${OT_TOPK:-8}"
      --ot-visual-tokens "${OT_VISUAL_TOKENS:-36}"
      --ot-sinkhorn-iters "${OT_SINKHORN_ITERS:-3}"
      --ot-epsilon "${OT_EPSILON:-0.05}"
    )
    if [[ "${OT_LOG_STATS:-1}" == "1" ]]; then
      method_args+=(--ot-log-stats)
    fi
    ;;
  uniform)
    method_args=(
      --use-ot-bary-sla
      --ot-force-uniform
      --ot-topk "${OT_TOPK:-8}"
      --ot-visual-tokens "${OT_VISUAL_TOKENS:-36}"
      --ot-sinkhorn-iters "${OT_SINKHORN_ITERS:-3}"
      --ot-epsilon "${OT_EPSILON:-0.05}"
    )
    ;;
  *)
    echo "MODE must be one of: original, ot, uniform; got: $MODE" >&2
    exit 1
    ;;
esac

echo "Mode: $MODE"
echo "Model: $VISTA_LLAVA_MODEL_PATH"
echo "COCO: $VISTA_COCO_ROOT"
echo "SLA layers: $LOGITS_LAYERS (inclusive)"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
python chair_eval.py "${common_args[@]}" "${method_args[@]}"
