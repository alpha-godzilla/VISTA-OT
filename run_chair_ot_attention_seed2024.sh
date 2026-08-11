#!/usr/bin/env bash
set -Eeuo pipefail

# Paired, one-seed recommended run for the attention-derived visual marginal.
# GPU 6 runs the matched VISTA baseline; GPU 7 runs OT-attention. Both consume
# exactly the same seed-2024, 500-image CHAIR manifest.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2024}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
BASELINE_GPU="${BASELINE_GPU:-6}"
ATTENTION_GPU="${ATTENTION_GPU:-7}"
EXP_FOLDER="${EXP_FOLDER:-chair_ot_attention_seed2024}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
LOGITS_ALPHA="${LOGITS_ALPHA:-0.3}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"

# Recommended first configuration. Attention OT uses all original visual
# tokens; OT_VISUAL_TOKENS is intentionally absent from this path.
OT_TOPK="${OT_TOPK:-16}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-50}"
OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"
OT_EPSILON="${OT_EPSILON:-0.05}"
OT_LAYER_TEMPERATURE="${OT_LAYER_TEMPERATURE:-0.2}"
OT_ATTENTION_POWER="${OT_ATTENTION_POWER:-0.5}"
OT_ATTENTION_UNIFORM_MIX="${OT_ATTENTION_UNIFORM_MIX:-0.02}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

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

if [[ ! -f "${VISTA_LLAVA_MODEL_PATH:-}/config.json" ]]; then
  echo "Set VISTA_LLAVA_MODEL_PATH to a LLaVA-1.5 model directory." >&2
  exit 1
fi
if [[ ! -d "$VISTA_COCO_ROOT/val2014" ]]; then
  echo "COCO val2014 directory not found: $VISTA_COCO_ROOT/val2014" >&2
  exit 1
fi

RUN_DIR="exp_results/$EXP_FOLDER"
MANIFEST="$RUN_DIR/chair_seed${SEED}_${SUBSET_SIZE}.txt"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$MANIFEST" ]]; then
  "$PYTHON_BIN" scripts/make_chair_seed_manifest.py \
    --data-path "$VISTA_COCO_ROOT/val2014" \
    --seed "$SEED" --subset-size "$SUBSET_SIZE" --output "$MANIFEST"
fi

common_args=(
  --exp_folder "$EXP_FOLDER" --model llava-1.5
  --data-path "$VISTA_COCO_ROOT/val2014"
  --subset-size "$SUBSET_SIZE" --subset-ids-file "$MANIFEST" --seed "$SEED"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --vsv --vsv-lambda "$VSV_LAMBDA"
  --logits-aug --logits-layers "$LOGITS_LAYERS" --logits-alpha "$LOGITS_ALPHA"
)

echo "Seed $SEED: baseline on GPU $BASELINE_GPU, OT-attention on GPU $ATTENTION_GPU"
echo "OT-attention: M=$OT_TOPK K=unpooled eps=$OT_EPSILON max_iters=$OT_SINKHORN_ITERS tol=$OT_SINKHORN_TOLERANCE beta=$OT_ATTENTION_POWER rho=$OT_ATTENTION_UNIFORM_MIX"

CUDA_VISIBLE_DEVICES="$BASELINE_GPU" "$PYTHON_BIN" chair_eval.py \
  "${common_args[@]}" > "$LOG_DIR/vista_seed${SEED}.log" 2>&1 &
baseline_pid=$!

CUDA_VISIBLE_DEVICES="$ATTENTION_GPU" "$PYTHON_BIN" chair_eval.py \
  "${common_args[@]}" \
  --use-ot-bary-sla --ot-attention-visual-marginal \
  --ot-topk "$OT_TOPK" \
  --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" --ot-epsilon "$OT_EPSILON" \
  --ot-sinkhorn-tolerance "$OT_SINKHORN_TOLERANCE" \
  --ot-layer-temperature "$OT_LAYER_TEMPERATURE" \
  --ot-attention-power "$OT_ATTENTION_POWER" \
  --ot-attention-uniform-mix "$OT_ATTENTION_UNIFORM_MIX" \
  --ot-log-stats > "$LOG_DIR/ot_attention_seed${SEED}.log" 2>&1 &
attention_pid=$!

wait "$baseline_pid"
wait "$attention_pid"

echo "Completed. Logs: $LOG_DIR"
