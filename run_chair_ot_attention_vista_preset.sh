#!/usr/bin/env bash
# Recommended eight-GPU VISTA + unpooled attention-OT CHAIR configuration.
# Override any exported value on the command line when needed.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the two original VISTA controls and their CHAIR reference values.
export VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
export LOGITS_ALPHA="${LOGITS_ALPHA:-0.3}"
export LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"

# Search only the new attention-OT controls.
export SEEDS="${SEEDS:-2024}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
export LAYER_TEMPERATURES="${LAYER_TEMPERATURES:-0.03 0.06 0.1 0.2 0.4 0.8}"
export ATTENTION_POWERS="${ATTENTION_POWERS:-0.25 0.5 0.75 1.0 1.5}"
export UNIFORM_MIXES="${UNIFORM_MIXES:-0.02}"
export OT_TOPK="${OT_TOPK:-16}"
export OT_EPSILON="${OT_EPSILON:-0.05}"
export OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-50}"
export OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"

exec bash "$SCRIPT_DIR/run_chair_ot_attention_grid.sh"
