#!/usr/bin/env bash
# Strict held-out sweep of VISTA/SLA logits_alpha for UOT-CRC.
# Seed 1994 is development-only; final tables contain only seeds 2024 and 3407.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a LOGITS_ALPHAS <<< "${LOGITS_ALPHAS:-0.15 0.20 0.25 0.30 0.35}"
read -r -a HELDOUT_SEEDS <<< "${HELDOUT_SEEDS:-2024 3407}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1994}"

ROOT_DIR="${ROOT_DIR:-$SCRIPT_DIR/exp_results/chair_ot_uot_crc_alpha_heldout}"
VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-chair_ot_uot_crc_alpha_heldout_vista}"
OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_ot_uot_crc_alpha_heldout_otattn}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/exp_results/chair_ot_uot_crc_alpha_heldout_outputs}"
RECOVERY_RHO="${RECOVERY_RHO:-0.25}"
RECALL_CANDIDATE_TOPK="${RECALL_CANDIDATE_TOPK:-32}"

if (( ${#LOGITS_ALPHAS[@]} == 0 || ${#HELDOUT_SEEDS[@]} == 0 || ${#GPUS[@]} == 0 )); then
  echo "LOGITS_ALPHAS, HELDOUT_SEEDS, and GPU_IDS must be non-empty." >&2
  exit 1
fi
for seed in "${HELDOUT_SEEDS[@]}"; do
  if [[ "$seed" == "$CALIBRATION_SEED" ]]; then
    echo "CALIBRATION_SEED must not be included in HELDOUT_SEEDS." >&2
    exit 1
  fi
done

canonical_float() { "$PYTHON_BIN" -c 'import sys; print(float(sys.argv[1]))' "$1"; }
all_seeds="$CALIBRATION_SEED ${HELDOUT_SEEDS[*]}"
gpu_ids="${GPUS[*]}"
mkdir -p "$ROOT_DIR" "$OUTPUT_ROOT"
declare -a SUMMARY_ENTRIES=()

for raw_alpha in "${LOGITS_ALPHAS[@]}"; do
  alpha="$(canonical_float "$raw_alpha")"
  alpha_dir="$ROOT_DIR/alpha_${alpha}"
  stage1_dir="$alpha_dir/stage1"
  uot_dir="$alpha_dir/uot"
  echo "=== logits_alpha=$alpha: paired stage-1 generation ==="
  PYTHON_BIN="$PYTHON_BIN" SEEDS="$all_seeds" GPU_IDS="$gpu_ids" \
    LOGITS_ALPHA="$alpha" RECOVERY_RHOS="$RECOVERY_RHO" \
    RECALL_CANDIDATE_TOPK="$RECALL_CANDIDATE_TOPK" \
    INCLUDE_ADAPTIVE_BASELINE=0 SWEEP_DIR="$stage1_dir" \
    VISTA_EXP_FOLDER="$VISTA_EXP_FOLDER" OT_EXP_FOLDER="$OT_EXP_FOLDER" \
    bash "$SCRIPT_DIR/run_chair_ot_recall_recovery_ablation.sh"

  rho="$(canonical_float "$RECOVERY_RHO")"
  echo "=== logits_alpha=$alpha: UOT-CRC calibration and held-out application ==="
  PYTHON_BIN="$PYTHON_BIN" SEEDS="$all_seeds" GPU_IDS="$gpu_ids" \
    CALIBRATION_SEED="$CALIBRATION_SEED" SOURCE_MANIFEST="$stage1_dir/manifest.tsv" \
    OT_METHOD=recall_recovery OT_SETTING="rho${rho}_k${RECALL_CANDIDATE_TOPK}" \
    SWEEP_DIR="$uot_dir" OUTPUT_DIR="$OUTPUT_ROOT/alpha_${alpha}" \
    RUN_COUNTERFACTUAL=0 \
    bash "$SCRIPT_DIR/run_chair_ot_uot_risk_control.sh"

  SUMMARY_ENTRIES+=(--entry "$alpha=$uot_dir/uot_crc/manifest.tsv")
done

heldout_args=()
for seed in "${HELDOUT_SEEDS[@]}"; do heldout_args+=("$seed"); done
"$PYTHON_BIN" scripts/summarize_uot_crc_alpha_heldout.py \
  "${SUMMARY_ENTRIES[@]}" --heldout-seeds "${heldout_args[@]}" \
  --calibration-seed "$CALIBRATION_SEED" \
  --by-seed-csv "$ROOT_DIR/heldout_by_seed.csv" \
  --summary-csv "$ROOT_DIR/heldout_summary.csv" \
  --markdown "$ROOT_DIR/heldout_summary.md"

echo "Held-out alpha sweep complete: $ROOT_DIR/heldout_summary.md"
