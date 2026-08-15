#!/usr/bin/env bash
# Clean five-seed test: fixed-alpha controls followed by adaptive-alpha tuning.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SEEDS="${SEEDS:-1994 2024 3407 42 1234}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
export VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
export LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
export LAYER_TEMPERATURES="${LAYER_TEMPERATURES:-0.06}"
export ATTENTION_POWERS="${ATTENTION_POWERS:-0.75}"
export UNIFORM_MIXES="${UNIFORM_MIXES:-0.02}"
export OT_TOPK="${OT_TOPK:-16}"
export OT_EPSILON="${OT_EPSILON:-0.05}"
export OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-50}"
export OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"

# Stage 1: determine what fixed alpha alone can achieve for both methods.
export LOGITS_ALPHA="${FIXED_LOGITS_ALPHAS:-0.2 0.25 0.3}"
export SWEEP_DIR="${FIXED_ALPHA_SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_adaptive_refine_fixed_alpha}"
export VISTA_EXP_FOLDER="${FIXED_ALPHA_VISTA_EXP_FOLDER:-chair_ot_adaptive_refine_fixed_vista}"
export OT_EXP_FOLDER="${FIXED_ALPHA_OT_EXP_FOLDER:-chair_ot_adaptive_refine_fixed_otattn}"
bash "$SCRIPT_DIR/run_chair_ot_attention_alpha_multiseed.sh"

# Stage 2: adaptive alpha uses base alpha=0.3 and no coverage module.
export LOGITS_ALPHA="${ADAPTIVE_BASE_LOGITS_ALPHA:-0.3}"
export ADAPTIVE_MIN_RATIOS="${ADAPTIVE_MIN_RATIOS:-0 0.1 0.2 0.25 0.3 0.4}"
export RUN_COVERAGE=0
export SWEEP_DIR="${ADAPTIVE_SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_adaptive_refine_adaptive}"
export VISTA_EXP_FOLDER="${ADAPTIVE_VISTA_EXP_FOLDER:-chair_ot_adaptive_refine_adaptive_vista}"
export OT_EXP_FOLDER="${ADAPTIVE_OT_EXP_FOLDER:-chair_ot_adaptive_refine_adaptive_otattn}"
bash "$SCRIPT_DIR/run_chair_ot_attention_module_ablation.sh"
