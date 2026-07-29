#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Refine around the first sweep's best point (topk=16, visual_tokens=64).
# The three completed 64-token anchors (topk=8,16,32) are reused, leaving
# exactly 24 new jobs by default.
read -r -a TOPKS <<< "${TOPKS:-8 10 12 14 16 18 20 24 32}"
read -r -a VISUAL_TOKENS <<< "${VISUAL_TOKENS:-49 64 81}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5}"

MODEL="${MODEL:-llava-1.5}"
EXP_FOLDER="${EXP_FOLDER:-chair_ot_topk_visual}"
SEED="${SEED:-1994}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
SUBSET_IDS_FILE="${SUBSET_IDS_FILE:-/data/sun_yuxi/datasets/coco/splits/chair_seed1994_500.txt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
GAMMA="${GAMMA:-0.3}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-3}"
OT_EPSILON="${OT_EPSILON:-0.05}"

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

COCO_VAL2014_PATH="$VISTA_COCO_ROOT/val2014"
COCO_ANNOTATIONS_PATH="$VISTA_COCO_ROOT/annotations"
CHAIR_CACHE="${CHAIR_CACHE:-$VISTA_COCO_ROOT/chair.pkl}"
RESULT_DIR="$SCRIPT_DIR/exp_results/$EXP_FOLDER/$MODEL"
SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_topk_visual_sweep}"
LOG_DIR="$SWEEP_DIR/logs"
MANIFEST="$SWEEP_DIR/manifest.tsv"
SUMMARY_CSV="$SWEEP_DIR/chair_ot_topk_visual_summary.csv"
SUMMARY_MD="$SWEEP_DIR/chair_ot_topk_visual_summary.md"

if (( ${#TOPKS[@]} == 0 || ${#VISUAL_TOKENS[@]} == 0 )); then
  echo "TOPKS and VISUAL_TOKENS must each contain at least one value." >&2
  exit 1
fi
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi
if ! [[ "$SUBSET_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUBSET_SIZE must be a positive integer; got: $SUBSET_SIZE" >&2
  exit 1
fi
if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then
  echo "Set VISTA_LLAVA_MODEL_PATH to the LLaVA-1.5 model directory." >&2
  exit 1
fi
if [[ ! -f "$SUBSET_IDS_FILE" ]]; then
  echo "Fixed CHAIR image ID file not found: $SUBSET_IDS_FILE" >&2
  exit 1
fi

fixed_id_count="$(
  awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' \
    "$SUBSET_IDS_FILE"
)"
if [[ "$fixed_id_count" -ne "$SUBSET_SIZE" ]]; then
  echo "Expected $SUBSET_SIZE fixed image IDs, found $fixed_id_count in $SUBSET_IDS_FILE" >&2
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
  local topk="$1"
  local visual_tokens="$2"
  printf '%s/seed%s_vsv_lambda_%s_logaug_loglayer_%s_logalpha_%s_otbary_m%s_k%s_it%s_eps%s_greedy_max_new_tokens_%s.jsonl' \
    "$RESULT_DIR" \
    "$SEED" \
    "$VSV_LAMBDA" \
    "$LOGITS_LAYERS" \
    "$GAMMA" \
    "$topk" \
    "$visual_tokens" \
    "$OT_SINKHORN_ITERS" \
    "$OT_EPSILON" \
    "$MAX_NEW_TOKENS"
}

is_complete_result() {
  local result="$1"
  [[ -f "$result" ]] && [[ "$(wc -l < "$result")" -eq "$SUBSET_SIZE" ]]
}

declare -a JOB_TOPKS=()
declare -a JOB_VISUAL_TOKENS=()
declare -a JOB_RESULTS=()
declare -a PENDING_INDICES=()

job_index=0
pending_index=0
{
  printf 'topk\tvisual_tokens\tgpu\tresult_jsonl\tchair_json\n'
  for topk in "${TOPKS[@]}"; do
    for visual_tokens in "${VISUAL_TOKENS[@]}"; do
      result="$(result_path "$topk" "$visual_tokens")"
      chair_json="${result%.jsonl}_chair.json"
      if is_complete_result "$result"; then
        gpu="-1"
      else
        gpu="${GPUS[$((pending_index % ${#GPUS[@]}))]}"
        PENDING_INDICES+=("$job_index")
        ((pending_index += 1))
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$topk" "$visual_tokens" "$gpu" "$result" "$chair_json"
      JOB_TOPKS+=("$topk")
      JOB_VISUAL_TOKENS+=("$visual_tokens")
      JOB_RESULTS+=("$result")
      ((job_index += 1))
    done
  done
} > "$MANIFEST"

run_gpu_worker() {
  local worker_index="$1"
  local gpu="${GPUS[$worker_index]}"
  local pending_position index topk visual_tokens result stats log backup

  for ((pending_position=worker_index; pending_position<${#PENDING_INDICES[@]}; pending_position+=${#GPUS[@]})); do
    index="${PENDING_INDICES[$pending_position]}"
    topk="${JOB_TOPKS[$index]}"
    visual_tokens="${JOB_VISUAL_TOKENS[$index]}"
    result="${JOB_RESULTS[$index]}"
    stats="${result%.jsonl}_ot_stats.jsonl"
    log="$LOG_DIR/topk_${topk}_visual_${visual_tokens}.log"

    if is_complete_result "$result"; then
      echo "[GPU $gpu] skip complete result: topk=$topk visual_tokens=$visual_tokens"
      continue
    fi

    if [[ -f "$result" ]]; then
      backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"
      mv "$result" "$backup"
      echo "[GPU $gpu] preserved partial result as: $backup"
    fi
    if [[ -f "$stats" ]]; then
      backup="${stats}.partial.$(date +%Y%m%d_%H%M%S)"
      mv "$stats" "$backup"
    fi

    echo "[GPU $gpu] start topk=$topk visual_tokens=$visual_tokens (log: $log)"
    if ! CUDA_VISIBLE_DEVICES="$gpu" python chair_eval.py \
      --exp_folder "$EXP_FOLDER" \
      --model "$MODEL" \
      --data-path "$COCO_VAL2014_PATH" \
      --subset-size "$SUBSET_SIZE" \
      --subset-ids-file "$SUBSET_IDS_FILE" \
      --seed "$SEED" \
      --vsv \
      --vsv-lambda "$VSV_LAMBDA" \
      --logits-aug \
      --logits-layers "$LOGITS_LAYERS" \
      --logits-alpha "$GAMMA" \
      --use-ot-bary-sla \
      --ot-topk "$topk" \
      --ot-visual-tokens "$visual_tokens" \
      --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" \
      --ot-epsilon "$OT_EPSILON" \
      --ot-log-stats \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      > "$log" 2>&1; then
      echo "[GPU $gpu] failed topk=$topk visual_tokens=$visual_tokens; see $log" >&2
      return 1
    fi

    if ! is_complete_result "$result"; then
      echo "[GPU $gpu] incomplete result after generation: $result" >&2
      return 1
    fi
    echo "[GPU $gpu] finished topk=$topk visual_tokens=$visual_tokens"
  done
}

declare -a WORKER_PIDS=()

stop_workers() {
  if (( ${#WORKER_PIDS[@]} > 0 )); then
    kill "${WORKER_PIDS[@]}" 2>/dev/null || true
  fi
}
trap stop_workers INT TERM

echo "Fixed gamma=$GAMMA lambda=$VSV_LAMBDA"
echo "Fixed image IDs: $SUBSET_IDS_FILE ($fixed_id_count images)"
echo "Grid entries: ${#JOB_TOPKS[@]}; reused: $((${#JOB_TOPKS[@]} - ${#PENDING_INDICES[@]})); pending: ${#PENDING_INDICES[@]}"
echo "Launching pending runs on GPUs: ${GPUS[*]}"
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
  echo "At least one generation worker failed. Fix the logged error and rerun." >&2
  exit 1
fi

echo "All generations finished. Running CHAIR evaluation serially..."
for ((index=0; index<${#JOB_TOPKS[@]}; index+=1)); do
  topk="${JOB_TOPKS[$index]}"
  visual_tokens="${JOB_VISUAL_TOKENS[$index]}"
  result="${JOB_RESULTS[$index]}"
  chair_json="${result%.jsonl}_chair.json"
  eval_log="$LOG_DIR/topk_${topk}_visual_${visual_tokens}_chair.log"

  if [[ -f "$chair_json" ]] && [[ "$chair_json" -nt "$result" ]]; then
    echo "Reuse CHAIR evaluation: topk=$topk visual_tokens=$visual_tokens"
    continue
  fi

  if ! python chair_ans.py \
    --cap_file "$result" \
    --coco_path "$COCO_ANNOTATIONS_PATH" \
    --cache "$CHAIR_CACHE" \
    --save_path "$chair_json" \
    > "$eval_log" 2>&1; then
    echo "CHAIR evaluation failed for topk=$topk visual_tokens=$visual_tokens; see $eval_log" >&2
    exit 1
  fi
  echo "Evaluated topk=$topk visual_tokens=$visual_tokens"
done

python scripts/summarize_chair_ot_sweep.py \
  --manifest "$MANIFEST" \
  --csv "$SUMMARY_CSV" \
  --markdown "$SUMMARY_MD" \
  --gamma "$GAMMA" \
  --vsv-lambda "$VSV_LAMBDA"

echo "Sweep complete."
echo "Markdown summary: $SUMMARY_MD"
echo "CSV summary:      $SUMMARY_CSV"
