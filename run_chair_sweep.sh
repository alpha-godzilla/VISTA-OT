#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Gamma is the name used for SLA strength in this sweep. In VISTA's CLI the
# same value is passed through --logits-alpha.
read -r -a GAMMAS <<< "${GAMMAS:-0.1 0.2 0.3 0.4}"
read -r -a LAMBDAS <<< "${LAMBDAS:-0.13 0.14 0.15 0.16 0.17 0.18}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5}"

MODEL="${MODEL:-llava-1.5}"
EXP_FOLDER="${EXP_FOLDER:-chair_eval}"
SEED="${SEED:-1994}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"

export VISTA_LLAVA_MODEL_PATH="${VISTA_LLAVA_MODEL_PATH:-/data/sun_yuxi/models/llava-1.5-7b-hf}"
export VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
export NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"

COCO_VAL2014_PATH="$VISTA_COCO_ROOT/val2014"
COCO_ANNOTATIONS_PATH="$VISTA_COCO_ROOT/annotations"
CHAIR_CACHE="${CHAIR_CACHE:-$VISTA_COCO_ROOT/chair.pkl}"
RESULT_DIR="$SCRIPT_DIR/exp_results/$EXP_FOLDER/$MODEL"
SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_sweep_gamma_lambda}"
LOG_DIR="$SWEEP_DIR/logs"
MANIFEST="$SWEEP_DIR/manifest.tsv"
SUMMARY_CSV="$SWEEP_DIR/chair_sweep_summary.csv"
SUMMARY_MD="$SWEEP_DIR/chair_sweep_summary.md"

if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi

if ! [[ "$SUBSET_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUBSET_SIZE must be a positive integer; got: $SUBSET_SIZE" >&2
  exit 1
fi

for path in \
  "$VISTA_LLAVA_MODEL_PATH" \
  "$COCO_VAL2014_PATH" \
  "$COCO_ANNOTATIONS_PATH"; do
  if [[ ! -d "$path" ]]; then
    echo "Required directory not found: $path" >&2
    exit 1
  fi
done

for name in \
  instances_train2014.json \
  instances_val2014.json \
  captions_train2014.json \
  captions_val2014.json; do
  if [[ ! -f "$COCO_ANNOTATIONS_PATH/$name" ]]; then
    echo "Required COCO annotation not found: $COCO_ANNOTATIONS_PATH/$name" >&2
    exit 1
  fi
done

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$NLTK_DATA"

result_path() {
  local gamma="$1"
  local lambda="$2"
  printf '%s/seed%s_vsv_lambda_%s_logaug_loglayer_%s_logalpha_%s_greedy_max_new_tokens_%s.jsonl' \
    "$RESULT_DIR" "$SEED" "$lambda" "$LOGITS_LAYERS" "$gamma" "$MAX_NEW_TOKENS"
}

is_complete_result() {
  local result="$1"
  [[ -f "$result" ]] && [[ "$(wc -l < "$result")" -eq "$SUBSET_SIZE" ]]
}

declare -a JOB_GAMMAS=()
declare -a JOB_LAMBDAS=()
declare -a JOB_RESULTS=()

job_index=0
{
  printf 'gamma\tlambda\tgpu\tresult_jsonl\tchair_json\n'
  for gamma in "${GAMMAS[@]}"; do
    for lambda in "${LAMBDAS[@]}"; do
      gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
      result="$(result_path "$gamma" "$lambda")"
      chair_json="${result%.jsonl}_chair.json"
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$gamma" "$lambda" "$gpu" "$result" "$chair_json"
      JOB_GAMMAS+=("$gamma")
      JOB_LAMBDAS+=("$lambda")
      JOB_RESULTS+=("$result")
      ((job_index += 1))
    done
  done
} > "$MANIFEST"

run_gpu_worker() {
  local worker_index="$1"
  local gpu="${GPUS[$worker_index]}"
  local index gamma lambda result log backup

  for ((index=worker_index; index<${#JOB_GAMMAS[@]}; index+=${#GPUS[@]})); do
    gamma="${JOB_GAMMAS[$index]}"
    lambda="${JOB_LAMBDAS[$index]}"
    result="${JOB_RESULTS[$index]}"
    log="$LOG_DIR/gamma_${gamma}_lambda_${lambda}.log"

    if is_complete_result "$result"; then
      echo "[GPU $gpu] skip complete result: gamma=$gamma lambda=$lambda"
      continue
    fi

    if [[ -f "$result" ]]; then
      backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"
      mv "$result" "$backup"
      echo "[GPU $gpu] preserved partial result as: $backup"
    fi

    echo "[GPU $gpu] start gamma=$gamma lambda=$lambda (log: $log)"
    if ! CUDA_VISIBLE_DEVICES="$gpu" python chair_eval.py \
      --exp_folder "$EXP_FOLDER" \
      --model "$MODEL" \
      --data-path "$COCO_VAL2014_PATH" \
      --subset-size "$SUBSET_SIZE" \
      --vsv \
      --vsv-lambda "$lambda" \
      --logits-aug \
      --logits-layers "$LOGITS_LAYERS" \
      --logits-alpha "$gamma" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --seed "$SEED" \
      > "$log" 2>&1; then
      echo "[GPU $gpu] failed gamma=$gamma lambda=$lambda; see $log" >&2
      return 1
    fi

    if ! is_complete_result "$result"; then
      echo "[GPU $gpu] incomplete result after generation: $result" >&2
      return 1
    fi
    echo "[GPU $gpu] finished gamma=$gamma lambda=$lambda"
  done
}

declare -a WORKER_PIDS=()

stop_workers() {
  if (( ${#WORKER_PIDS[@]} > 0 )); then
    kill "${WORKER_PIDS[@]}" 2>/dev/null || true
  fi
}
trap stop_workers INT TERM

echo "Launching ${#JOB_GAMMAS[@]} runs on GPUs: ${GPUS[*]}"
for ((worker=0; worker<${#GPUS[@]}; worker+=1)); do
  run_gpu_worker "$worker" &
  WORKER_PIDS+=("$!")
done

generation_failed=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    generation_failed=1
  fi
done
WORKER_PIDS=()

if (( generation_failed != 0 )); then
  echo "At least one generation worker failed. Fix the logged error and rerun; completed results will be skipped." >&2
  exit 1
fi

echo "All generations finished. Running CHAIR evaluation serially..."
for ((index=0; index<${#JOB_GAMMAS[@]}; index+=1)); do
  gamma="${JOB_GAMMAS[$index]}"
  lambda="${JOB_LAMBDAS[$index]}"
  result="${JOB_RESULTS[$index]}"
  chair_json="${result%.jsonl}_chair.json"
  eval_log="$LOG_DIR/gamma_${gamma}_lambda_${lambda}_chair.log"

  if ! python chair_ans.py \
    --cap_file "$result" \
    --coco_path "$COCO_ANNOTATIONS_PATH" \
    --cache "$CHAIR_CACHE" \
    --save_path "$chair_json" \
    > "$eval_log" 2>&1; then
    echo "CHAIR evaluation failed for gamma=$gamma lambda=$lambda; see $eval_log" >&2
    exit 1
  fi
  echo "Evaluated gamma=$gamma lambda=$lambda"
done

python scripts/summarize_chair_sweep.py \
  --manifest "$MANIFEST" \
  --csv "$SUMMARY_CSV" \
  --markdown "$SUMMARY_MD"

echo "Sweep complete."
echo "Markdown summary: $SUMMARY_MD"
echo "CSV summary:      $SUMMARY_CSV"
