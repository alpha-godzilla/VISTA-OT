#!/usr/bin/env bash
# V2 recall recovery: wider proposals -> candidate UOT -> scalar CRC gate.
# Existing local-verifier scripts and result directories are intentionally untouched.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a SEEDS <<< "${SEEDS:-1994 2024 3407}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"

MODEL="${MODEL:-llava-1.5}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1994}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$SCRIPT_DIR/exp_results/chair_ot_recall_recovery_ablation/manifest.tsv}"
OT_METHOD="${OT_METHOD:-recall_recovery}"
OT_SETTING="${OT_SETTING:-rho0.25_k32}"
SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_uot_risk_control}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/exp_results/chair_ot_uot_risk_control_outputs/$MODEL}"

VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
REGION_TOPKS="${REGION_TOPKS:-8,16,32}"
UOT_RELAXATIONS="${UOT_RELAXATIONS:-0.1,0.2,0.5}"
OT_EPSILON="${OT_EPSILON:-0.05}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-50}"
OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"
OT_LAYER_TEMPERATURE="${OT_LAYER_TEMPERATURE:-0.06}"
OT_ATTENTION_POWER="${OT_ATTENTION_POWER:-0.75}"
OT_UNIFORM_MIX="${OT_UNIFORM_MIX:-0.02}"

MAX_CANDIDATES="${MAX_CANDIDATES:-12}"
MAX_ADDITIONS="${MAX_ADDITIONS:-1}"
TUNE_FRACTION="${TUNE_FRACTION:-0.5}"
SPLIT_SALT="${SPLIT_SALT:-2024}"
RISK_BUDGET="${RISK_BUDGET:-0.01}"
TUNE_PRECISION_FLOOR="${TUNE_PRECISION_FLOOR:-0.90}"
RUN_COUNTERFACTUAL="${RUN_COUNTERFACTUAL:-1}"
COUNTERFACTUAL_NOISE_STD="${COUNTERFACTUAL_NOISE_STD:-0.20}"

export VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
export NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"
if [[ -z "${HF_HOME:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" && -d /data/sun_yuxi/huggingface ]]; then
  export HF_HOME=/data/sun_yuxi/huggingface
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then
  for candidate in /data/sun_yuxi/models/llava-v1.5-7b /data/sun_yuxi/models/llava-1.5-7b-hf /home/ljc/code/models/llava-v1.5-7b; do
    if [[ -f "$candidate/config.json" ]]; then
      export VISTA_LLAVA_MODEL_PATH="$candidate"
      break
    fi
  done
fi
if [[ ! -f "$SOURCE_MANIFEST" ]]; then
  echo "Missing source manifest: $SOURCE_MANIFEST" >&2
  echo "Finish run_chair_ot_recall_recovery_ablation.sh first." >&2
  exit 1
fi
if [[ ! -d "$VISTA_COCO_ROOT/val2014" || ! -f "${VISTA_LLAVA_MODEL_PATH:-}/config.json" ]]; then
  echo "Set VISTA_COCO_ROOT and VISTA_LLAVA_MODEL_PATH before running." >&2
  exit 1
fi
if (( ${#SEEDS[@]} == 0 || ${#GPUS[@]} == 0 )); then
  echo "SEEDS and GPU_IDS must be non-empty." >&2
  exit 1
fi
if [[ "$MAX_ADDITIONS" != 1 ]]; then
  echo "V2 risk control requires MAX_ADDITIONS=1." >&2
  exit 1
fi

mkdir -p "$SWEEP_DIR/logs" "$SWEEP_DIR/scores" "$OUTPUT_DIR"
WORK_MANIFEST="$SWEEP_DIR/candidate_work_v2.jsonl"
seed_args=()
for seed in "${SEEDS[@]}"; do seed_args+=("$seed"); done

"$PYTHON_BIN" scripts/build_ot_candidate_work.py \
  --manifest "$SOURCE_MANIFEST" --output "$WORK_MANIFEST" \
  --seeds "${seed_args[@]}" --ot-method "$OT_METHOD" --ot-setting "$OT_SETTING" \
  --candidate-version v2 --max-candidates "$MAX_CANDIDATES" \
  > "$SWEEP_DIR/candidate_extraction_v2.log"
"$PYTHON_BIN" scripts/audit_ot_candidate_extraction.py \
  --source-manifest "$SOURCE_MANIFEST" --work-manifest "$WORK_MANIFEST" \
  --output-json "$SWEEP_DIR/candidate_extraction_audit.json" \
  --output-markdown "$SWEEP_DIR/candidate_extraction_audit.md" \
  --seeds "${seed_args[@]}" --ot-method "$OT_METHOD" --ot-setting "$OT_SETTING"

noise_std=0
if [[ "$RUN_COUNTERFACTUAL" == 1 ]]; then noise_std="$COUNTERFACTUAL_NOISE_STD"; fi
echo "UOT verifier: 8-way candidate shards on GPUs ${GPUS[*]} (counterfactual=$RUN_COUNTERFACTUAL)"
declare -a PIDS=() SCORE_FILES=()
for ((worker=0; worker<${#GPUS[@]}; worker+=1)); do
  gpu="${GPUS[$worker]}"
  score_file="$SWEEP_DIR/scores/shard_${worker}_of_${#GPUS[@]}.jsonl"
  SCORE_FILES+=("$score_file")
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_ot_candidate_verifier.py \
    --model "$MODEL" --data-path "$VISTA_COCO_ROOT/val2014" \
    --work-manifest "$WORK_MANIFEST" --output "$score_file" \
    --shard-index "$worker" --num-shards "${#GPUS[@]}" \
    --seed "$CALIBRATION_SEED" --vsv --vsv-lambda "$VSV_LAMBDA" \
    --logits-layers "$LOGITS_LAYERS" --region-topks "$REGION_TOPKS" \
    --uot-marginal-relaxations "$UOT_RELAXATIONS" \
    --counterfactual-noise-std "$noise_std" \
    --ot-epsilon "$OT_EPSILON" --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" \
    --ot-sinkhorn-tolerance "$OT_SINKHORN_TOLERANCE" \
    --ot-layer-temperature "$OT_LAYER_TEMPERATURE" \
    --ot-attention-power "$OT_ATTENTION_POWER" \
    --ot-attention-uniform-mix "$OT_UNIFORM_MIX" \
    > "$SWEEP_DIR/logs/uot_verifier_shard_${worker}.log" 2>&1 &
  PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
if (( failed != 0 )); then
  echo "UOT candidate scoring failed; see $SWEEP_DIR/logs" >&2
  exit 1
fi

is_complete_chair() {
  "$PYTHON_BIN" scripts/validate_chair_output.py \
    --path "$1" --expected-images "$SUBSET_SIZE" > /dev/null 2>&1
}

run_mode() {
  local mode="$1"
  local tag="$2"
  local mode_dir="$SWEEP_DIR/$tag"
  local manifest="$mode_dir/manifest.tsv"
  mkdir -p "$mode_dir"
  "$PYTHON_BIN" scripts/calibrate_apply_ot_uot_risk.py \
    --source-manifest "$SOURCE_MANIFEST" --work-manifest "$WORK_MANIFEST" \
    --scores "${SCORE_FILES[@]}" --output-dir "$OUTPUT_DIR" \
    --output-manifest "$manifest" --output-tag "$tag" \
    --report-json "$mode_dir/report.json" \
    --report-markdown "$mode_dir/report.md" --mode "$mode" \
    --seeds "${seed_args[@]}" --calibration-seed "$CALIBRATION_SEED" \
    --tune-fraction "$TUNE_FRACTION" --split-salt "$SPLIT_SALT" \
    --risk-budget "$RISK_BUDGET" \
    --tune-precision-floor "$TUNE_PRECISION_FLOOR" \
    --max-additions "$MAX_ADDITIONS" \
    --ot-method "$OT_METHOD" --ot-setting "$OT_SETTING"

  while IFS=$'\t' read -r method setting seed gpu ids result chair_json gate_passed crc_bound; do
    [[ "$method" == method ]] && continue
    if [[ "$method" != uot_crc && "$method" != uot_contrast_crc ]]; then
      continue
    fi
    if ! is_complete_chair "$chair_json" || [[ "$chair_json" -ot "$result" ]]; then
      "$PYTHON_BIN" chair_ans.py --cap_file "$result" \
        --coco_path "$VISTA_COCO_ROOT/annotations" \
        --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$chair_json" \
        > "$SWEEP_DIR/logs/chair_${tag}_seed${seed}.log" 2>&1
    fi
  done < "$manifest"
  "$PYTHON_BIN" scripts/summarize_chair_ot_attention_modules.py \
    --manifest "$manifest" --csv "$mode_dir/summary.csv" \
    --markdown "$mode_dir/summary.md"
}

echo "Stage 1: UOT-only scalar verifier and independent risk calibration"
run_mode uot uot_crc
if [[ "$RUN_COUNTERFACTUAL" == 1 ]]; then
  echo "Stage 2: UOT + counterfactual contrast ablation"
  run_mode contrast uot_contrast_crc
fi

echo "Candidate extraction audit: $SWEEP_DIR/candidate_extraction_audit.md"
echo "UOT-only report: $SWEEP_DIR/uot_crc/report.md"
if [[ "$RUN_COUNTERFACTUAL" == 1 ]]; then
  echo "Contrast report: $SWEEP_DIR/uot_contrast_crc/report.md"
fi
