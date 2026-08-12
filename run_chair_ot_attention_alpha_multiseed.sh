#!/usr/bin/env bash
# Matched VISTA and attention-OT search over VISTA's original logits_alpha.
# For every seed and alpha, this runs one unmodified VISTA baseline (no OT)
# and one attention-OT run on exactly the same image subset.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# VISTA controls: alpha is deliberately searched for both methods.
export VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
export LOGITS_ALPHA="${LOGITS_ALPHA:-0.15 0.20 0.25 0.30 0.35}"
export LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
export SEEDS="${SEEDS:-1994 2024 3407 42 1234}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"

# Fixed attention-OT configuration under comparison.
export LAYER_TEMPERATURES="${LAYER_TEMPERATURES:-0.06}"
export ATTENTION_POWERS="${ATTENTION_POWERS:-0.75}"
export UNIFORM_MIXES="${UNIFORM_MIXES:-0.02}"
export OT_TOPK="${OT_TOPK:-16}"
export OT_EPSILON="${OT_EPSILON:-0.05}"
export OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-50}"
export OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"

# Keep alpha-search outputs separate from prior experiments.
export SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_attention_alpha_multiseed}"
export VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-chair_ot_attention_alpha_multiseed_vista}"
export OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_ot_attention_alpha_multiseed_otattn}"

exec bash "$SCRIPT_DIR/run_chair_ot_attention_grid.sh"
