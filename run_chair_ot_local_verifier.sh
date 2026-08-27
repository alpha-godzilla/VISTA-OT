#!/usr/bin/env bash
# Candidate-conditioned local OT verifier: score on 8 GPUs, calibrate, append, evaluate.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a SEEDS <<< "${SEEDS:-1994 2024 3407}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a VISTA_LOGITS_ALPHAS <<< "${VISTA_LOGITS_ALPHAS:-0.15 0.20 0.25 0.30 0.35}"
read -r -a OURS_PRECISION_FLOORS <<< "${OURS_PRECISION_FLOORS:-0.90 0.92 0.94 0.95 0.96 0.98}"

MODEL="${MODEL:-llava-1.5}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1994}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$SCRIPT_DIR/exp_results/chair_ot_recall_recovery_ablation/manifest.tsv}"
OT_METHOD="${OT_METHOD:-recall_recovery}"
OT_SETTING="${OT_SETTING:-rho0.25_k32}"
SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_local_verifier}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/exp_results/chair_ot_local_verifier_outputs/$MODEL}"
VISTA_ALPHA_EXP_FOLDER="${VISTA_ALPHA_EXP_FOLDER:-chair_ot_attention_alpha_multiseed_vista}"

VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
REGION_TOPKS="${REGION_TOPKS:-8,16,32}"
OT_EPSILON="${OT_EPSILON:-0.05}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-50}"
OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"
OT_LAYER_TEMPERATURE="${OT_LAYER_TEMPERATURE:-0.06}"
OT_ATTENTION_POWER="${OT_ATTENTION_POWER:-0.75}"
OT_UNIFORM_MIX="${OT_UNIFORM_MIX:-0.02}"
MINIMUM_TPR="${MINIMUM_TPR:-0.30}"
MAX_ADDITIONS="${MAX_ADDITIONS:-2}"
F1_TOLERANCE="${F1_TOLERANCE:-0.005}"

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
if (( ${#SEEDS[@]} == 0 || ${#GPUS[@]} == 0 || ${#VISTA_LOGITS_ALPHAS[@]} == 0 || ${#OURS_PRECISION_FLOORS[@]} == 0 )); then
  echo "SEEDS, GPU_IDS, VISTA_LOGITS_ALPHAS, and OURS_PRECISION_FLOORS must be non-empty." >&2
  exit 1
fi
if [[ ! "$SUBSET_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUBSET_SIZE must be a positive integer." >&2
  exit 1
fi

mkdir -p "$SWEEP_DIR/logs" "$SWEEP_DIR/scores" "$OUTPUT_DIR"
WORK_MANIFEST="$SWEEP_DIR/candidate_work.jsonl"

seed_args=()
for seed in "${SEEDS[@]}"; do seed_args+=("$seed"); done
"$PYTHON_BIN" scripts/build_ot_candidate_work.py \
  --manifest "$SOURCE_MANIFEST" --output "$WORK_MANIFEST" \
  --seeds "${seed_args[@]}" --ot-method "$OT_METHOD" --ot-setting "$OT_SETTING" \
  > "$SWEEP_DIR/candidate_extraction.log"

echo "Local OT verifier: scoring candidate shards on GPUs ${GPUS[*]}"
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
    --ot-epsilon "$OT_EPSILON" --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" \
    --ot-sinkhorn-tolerance "$OT_SINKHORN_TOLERANCE" \
    --ot-layer-temperature "$OT_LAYER_TEMPERATURE" \
    --ot-attention-power "$OT_ATTENTION_POWER" \
    --ot-attention-uniform-mix "$OT_UNIFORM_MIX" \
    > "$SWEEP_DIR/logs/verifier_shard_${worker}.log" 2>&1 &
  PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
if (( failed != 0 )); then
  echo "Candidate scoring failed; see $SWEEP_DIR/logs" >&2
  exit 1
fi

# The expensive candidate features above are shared by every offline gate.
declare -a OURS_MANIFESTS=()
for raw_floor in "${OURS_PRECISION_FLOORS[@]}"; do
  floor="$($PYTHON_BIN -c 'import sys; print(float(sys.argv[1]))' "$raw_floor")"
  gate_tag="p${floor}_m${MAX_ADDITIONS}"
  gate_dir="$SWEEP_DIR/gates/$gate_tag"
  gate_manifest="$gate_dir/manifest.tsv"
  mkdir -p "$gate_dir"
  "$PYTHON_BIN" scripts/calibrate_apply_ot_candidate_verifier.py \
    --source-manifest "$SOURCE_MANIFEST" --work-manifest "$WORK_MANIFEST" \
    --scores "${SCORE_FILES[@]}" --output-dir "$OUTPUT_DIR" \
    --output-tag "$gate_tag" --output-manifest "$gate_manifest" \
    --report-json "$gate_dir/verifier_report.json" \
    --report-markdown "$gate_dir/verifier_report.md" \
    --seeds "${seed_args[@]}" --calibration-seed "$CALIBRATION_SEED" \
    --ot-method "$OT_METHOD" --ot-setting "$OT_SETTING" \
    --precision-floor "$floor" --minimum-tpr "$MINIMUM_TPR" \
    --max-additions "$MAX_ADDITIONS" --allow-failed-gate
  OURS_MANIFESTS+=("$gate_manifest")

  while IFS=$'\t' read -r method setting seed gpu ids result chair_json gate_passed calibration_precision calibration_tpr; do
    [[ "$method" == method || "$method" != local_verifier ]] && continue
    if [[ ! -f "$chair_json" || "$chair_json" -ot "$result" ]]; then
      "$PYTHON_BIN" chair_ans.py --cap_file "$result" \
        --coco_path "$VISTA_COCO_ROOT/annotations" \
        --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$chair_json" \
        > "$SWEEP_DIR/logs/chair_${method}_${gate_tag}_seed${seed}.log" 2>&1
    fi
  done < "$gate_manifest"
done

# Run only VISTA's original alpha axis. Alpha=0.3 reuses the source baseline;
# other completed alpha results are also reused from the prior alpha folder.
VISTA_ALPHA_MANIFEST="$SWEEP_DIR/vista_alpha_manifest.tsv"
printf 'method\tseed\tlogits_alpha\tgpu\tids_file\tresult_jsonl\tchair_json\n' > "$VISTA_ALPHA_MANIFEST"
declare -a VISTA_JOB_SEEDS=() VISTA_JOB_ALPHAS=() VISTA_JOB_IDS=() VISTA_JOB_RESULTS=()
vista_pending=0

lookup_source() {
  local method="$1"
  local setting="$2"
  local seed="$3"
  local column="$4"
  awk -F '\t' -v method="$method" -v setting="$setting" -v seed="$seed" -v column="$column" \
    'NR > 1 && $1 == method && $2 == setting && $3 == seed { print $column; found=1; exit } END { if (!found) exit 1 }' \
    "$SOURCE_MANIFEST"
}
vista_result_path() {
  local seed="$1"
  local alpha="$2"
  printf '%s/exp_results/%s/%s/seed%s_vsv_lambda_%s_logaug_loglayer_%s_logalpha_%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$VISTA_ALPHA_EXP_FOLDER" "$MODEL" "$seed" "$VSV_LAMBDA" \
    "$LOGITS_LAYERS" "$alpha" "$MAX_NEW_TOKENS"
}
is_complete_result() { [[ -f "$1" && "$(wc -l < "$1")" -eq "$SUBSET_SIZE" ]]; }

for seed in "${SEEDS[@]}"; do
  ids="$(lookup_source "$OT_METHOD" "$OT_SETTING" "$seed" 5)"
  for raw_alpha in "${VISTA_LOGITS_ALPHAS[@]}"; do
    alpha="$($PYTHON_BIN -c 'import sys; print(float(sys.argv[1]))' "$raw_alpha")"
    if [[ "$alpha" == 0.3 ]]; then
      result="$(lookup_source vista original "$seed" 6)"
      chair_json="$(lookup_source vista original "$seed" 7)"
      gpu=-1
    else
      result="$(vista_result_path "$seed" "$alpha")"
      chair_json="${result%.jsonl}_chair.json"
      if is_complete_result "$result"; then
        gpu=-1
      else
        gpu="${GPUS[$((vista_pending % ${#GPUS[@]}))]}"
        VISTA_JOB_SEEDS+=("$seed")
        VISTA_JOB_ALPHAS+=("$alpha")
        VISTA_JOB_IDS+=("$ids")
        VISTA_JOB_RESULTS+=("$result")
        ((vista_pending += 1))
      fi
    fi
    printf 'vista\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$seed" "$alpha" "$gpu" "$ids" "$result" "$chair_json" >> "$VISTA_ALPHA_MANIFEST"
  done
done

run_vista_job() {
  local index="$1"
  local gpu="$2"
  local seed="${VISTA_JOB_SEEDS[$index]}"
  local alpha="${VISTA_JOB_ALPHAS[$index]}"
  local ids="${VISTA_JOB_IDS[$index]}"
  local result="${VISTA_JOB_RESULTS[$index]}"
  local backup
  local log_file="$SWEEP_DIR/logs/vista_seed${seed}_alpha${alpha}.log"
  if [[ -f "$result" ]]; then
    backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"
    mv "$result" "$backup"
  fi
  echo "[GPU $gpu] start VISTA seed=$seed logits_alpha=$alpha"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_eval.py \
    --exp_folder "$VISTA_ALPHA_EXP_FOLDER" --model "$MODEL" \
    --data-path "$VISTA_COCO_ROOT/val2014" --subset-size "$SUBSET_SIZE" \
    --subset-ids-file "$ids" --seed "$seed" --max-new-tokens "$MAX_NEW_TOKENS" \
    --vsv --vsv-lambda "$VSV_LAMBDA" --logits-aug \
    --logits-layers "$LOGITS_LAYERS" --logits-alpha "$alpha" \
    > "$log_file" 2>&1
  is_complete_result "$result" || { echo "Incomplete VISTA result: $result" >&2; return 1; }
}
run_vista_worker() {
  local worker="$1"
  local gpu="${GPUS[$worker]}"
  local index
  for ((index=worker; index<${#VISTA_JOB_SEEDS[@]}; index+=${#GPUS[@]})); do
    run_vista_job "$index" "$gpu"
  done
}

echo "VISTA matched-F1 alpha sweep: ${#VISTA_JOB_SEEDS[@]} pending jobs"
declare -a VISTA_PIDS=()
for ((worker=0; worker<${#GPUS[@]} && worker<${#VISTA_JOB_SEEDS[@]}; worker+=1)); do
  run_vista_worker "$worker" & VISTA_PIDS+=("$!")
done
vista_failed=0
for pid in "${VISTA_PIDS[@]}"; do wait "$pid" || vista_failed=1; done
if (( vista_failed != 0 )); then
  echo "VISTA alpha generation failed; see $SWEEP_DIR/logs" >&2
  exit 1
fi

while IFS=$'\t' read -r method seed alpha gpu ids result chair_json; do
  [[ "$method" == method ]] && continue
  if [[ ! -f "$chair_json" || "$chair_json" -ot "$result" ]]; then
    "$PYTHON_BIN" chair_ans.py --cap_file "$result" \
      --coco_path "$VISTA_COCO_ROOT/annotations" \
      --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$chair_json" \
      > "$SWEEP_DIR/logs/chair_vista_seed${seed}_alpha${alpha}.log" 2>&1
  fi
done < "$VISTA_ALPHA_MANIFEST"

"$PYTHON_BIN" scripts/summarize_ot_local_verifier_matched_f1.py \
  --vista-manifest "$VISTA_ALPHA_MANIFEST" \
  --ours-manifests "${OURS_MANIFESTS[@]}" \
  --calibration-seed "$CALIBRATION_SEED" --f1-tolerance "$F1_TOLERANCE" \
  --csv "$SWEEP_DIR/matched_f1_all_pairs.csv" \
  --markdown "$SWEEP_DIR/matched_f1.md"

echo "Per-gate reports: $SWEEP_DIR/gates"
echo "Matched-F1 comparison: $SWEEP_DIR/matched_f1.md"
