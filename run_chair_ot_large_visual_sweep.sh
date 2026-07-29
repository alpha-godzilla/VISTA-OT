#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TOPKS="${TOPKS:-8 16 32 64}"
export VISUAL_TOKENS="${VISUAL_TOKENS:-100 196 324 576}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"
export SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_large_visual_sweep}"
export SUMMARY_BASENAME="${SUMMARY_BASENAME:-chair_ot_large_visual_summary}"

exec bash "$SCRIPT_DIR/run_chair_ot_topk_visual_sweep.sh"
