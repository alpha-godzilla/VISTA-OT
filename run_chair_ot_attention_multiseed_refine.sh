#!/usr/bin/env bash
# Five-seed refinement around the promising unpooled attention-OT settings.
# Existing VISTA settings are kept unchanged; only the attention-OT controls
# are searched. Every variable may be overridden by the remote environment.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Original VISTA controls.
export VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
export LOGITS_ALPHA="${LOGITS_ALPHA:-0.3}"
export LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"

# Matched five-seed evaluation on the four nearby attention-OT settings.
export SEEDS="${SEEDS:-1994 2024 3407 42 1234}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
export LAYER_TEMPERATURES="${LAYER_TEMPERATURES:-0.06 0.1}"
export ATTENTION_POWERS="${ATTENTION_POWERS:-0.75 1.0}"
export UNIFORM_MIXES="${UNIFORM_MIXES:-0.02}"
export OT_TOPK="${OT_TOPK:-16}"
export OT_EPSILON="${OT_EPSILON:-0.05}"
export OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-50}"
export OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"

# Keep its manifests, logs, and outputs separate from the broad seed-2024 run.
export SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_attention_multiseed_refine}"
export VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-chair_ot_attention_multiseed_refine_vista}"
export OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_ot_attention_multiseed_refine_otattn}"

exec bash "$SCRIPT_DIR/run_chair_ot_attention_grid.sh"
