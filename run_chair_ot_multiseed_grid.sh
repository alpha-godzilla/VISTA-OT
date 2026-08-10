#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

read -r -a SEEDS <<< "${SEEDS:-1994 2024 3407 42 1234}"
read -r -a TOPKS <<< "${TOPKS:-4 16 32}"
read -r -a VISUAL_TOKENS <<< "${VISUAL_TOKENS:-16 64 81}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5}"

MODEL="${MODEL:-llava-1.5}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
GAMMA="${GAMMA:-0.3}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-3}"
OT_EPSILON="${OT_EPSILON:-0.05}"
OT_LAYER_TEMPERATURE="${OT_LAYER_TEMPERATURE:-0.1}"

GRID_DIR="${GRID_DIR:-$SCRIPT_DIR/exp_results/chair_ot_multiseed_grid_vdust_tlogit}"
VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-chair_vista_ot_seed_compare_vista}"
OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_ot_multiseed_grid_vdust_tlogit}"

export VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
export NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"
if [[ -z "${HF_HOME:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" ]] && [[ -d /data/sun_yuxi/huggingface ]]; then
  export HF_HOME=/data/sun_yuxi/huggingface
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

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

if (( ${#SEEDS[@]} == 0 || ${#TOPKS[@]} == 0 || ${#VISUAL_TOKENS[@]} == 0 || ${#GPUS[@]} == 0 )); then
  echo "SEEDS, TOPKS, VISUAL_TOKENS, and GPU_IDS must not be empty." >&2
  exit 1
fi
if ! [[ "$SUBSET_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUBSET_SIZE must be a positive integer; got: $SUBSET_SIZE" >&2
  exit 1
fi
if [[ ! -d "$VISTA_COCO_ROOT/val2014" ]]; then
  echo "COCO val2014 directory not found: $VISTA_COCO_ROOT/val2014" >&2
  exit 1
fi
if [[ ! -f "${VISTA_LLAVA_MODEL_PATH:-}/config.json" ]]; then
  echo "Set VISTA_LLAVA_MODEL_PATH to the LLaVA-1.5 model directory." >&2
  exit 1
fi

mkdir -p "$GRID_DIR/manifests" "$GRID_DIR/logs"
MANIFEST="$GRID_DIR/manifest.tsv"

base_stem() {
  local seed="$1"
  printf 'seed%s_vsv_lambda_%s_logaug_loglayer_%s_logalpha_%s' \
    "$seed" "$VSV_LAMBDA" "$LOGITS_LAYERS" "$GAMMA"
}

vista_result_path() {
  local seed="$1"
  printf '%s/exp_results/%s/%s/%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$VISTA_EXP_FOLDER" "$MODEL" "$(base_stem "$seed")" "$MAX_NEW_TOKENS"
}

ot_result_path() {
  local seed="$1" topk="$2" visual_tokens="$3"
  printf '%s/exp_results/%s/%s/%s_otbary_vdust_tlogit_m%s_k%s_it%s_eps%s_ltemp%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$OT_EXP_FOLDER" "$MODEL" "$(base_stem "$seed")" \
    "$topk" "$visual_tokens" "$OT_SINKHORN_ITERS" "$OT_EPSILON" "$OT_LAYER_TEMPERATURE" "$MAX_NEW_TOKENS"
}

is_complete_result() {
  local result="$1"
  [[ -f "$result" ]] && [[ "$(wc -l < "$result")" -eq "$SUBSET_SIZE" ]]
}

prepare_seed_ids() {
  local seed="$1" destination source
  destination="$GRID_DIR/manifests/seed_${seed}_ids.txt"
  if [[ -f "$destination" ]] && [[ "$(wc -l < "$destination")" -eq "$SUBSET_SIZE" ]]; then
    return
  fi

  for source in \
    "$SCRIPT_DIR/exp_results/chair_vista_ot_seed_compare/manifests/seed_${seed}_ids.txt" \
    "$SCRIPT_DIR/exp_results/chair_vista_ot_seed_compare_vdust_tlogit/manifests/seed_${seed}_ids.txt"; do
    if [[ -f "$source" ]] && [[ "$(wc -l < "$source")" -eq "$SUBSET_SIZE" ]]; then
      cp "$source" "$destination"
      echo "Reused fixed image IDs for seed=$seed from $source"
      return
    fi
  done

  python scripts/make_chair_seed_manifest.py \
    --data-path "$VISTA_COCO_ROOT/val2014" \
    --seed "$seed" \
    --subset-size "$SUBSET_SIZE" \
    --output "$destination"
}

for seed in "${SEEDS[@]}"; do
  prepare_seed_ids "$seed"
done

declare -a JOB_METHODS=()
declare -a JOB_SEEDS=()
declare -a JOB_TOPKS=()
declare -a JOB_VISUAL_TOKENS=()
declare -a JOB_IDS_FILES=()
declare -a JOB_RESULTS=()

pending_count=0
printf 'method\tseed\ttopk\tvisual_tokens\tgpu\tids_file\tresult_jsonl\tchair_json\n' > "$MANIFEST"
for seed in "${SEEDS[@]}"; do
  ids_file="$GRID_DIR/manifests/seed_${seed}_ids.txt"
  vista_result="$(vista_result_path "$seed")"
  if is_complete_result "$vista_result"; then
    gpu=-1
  else
    gpu="${GPUS[$((pending_count % ${#GPUS[@]}))]}"
    JOB_METHODS+=(vista)
    JOB_SEEDS+=("$seed")
    JOB_TOPKS+=(0)
    JOB_VISUAL_TOKENS+=(0)
    JOB_IDS_FILES+=("$ids_file")
    JOB_RESULTS+=("$vista_result")
    ((pending_count += 1))
  fi
  printf 'vista\t%s\t0\t0\t%s\t%s\t%s\t%s\n' \
    "$seed" "$gpu" "$ids_file" "$vista_result" "${vista_result%.jsonl}_chair.json" >> "$MANIFEST"

  for topk in "${TOPKS[@]}"; do
    for visual_tokens in "${VISUAL_TOKENS[@]}"; do
      ot_result="$(ot_result_path "$seed" "$topk" "$visual_tokens")"
      if is_complete_result "$ot_result"; then
        gpu=-1
      else
        gpu="${GPUS[$((pending_count % ${#GPUS[@]}))]}"
        JOB_METHODS+=(ot)
        JOB_SEEDS+=("$seed")
        JOB_TOPKS+=("$topk")
        JOB_VISUAL_TOKENS+=("$visual_tokens")
        JOB_IDS_FILES+=("$ids_file")
        JOB_RESULTS+=("$ot_result")
        ((pending_count += 1))
      fi
      printf 'ot\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$seed" "$topk" "$visual_tokens" "$gpu" "$ids_file" "$ot_result" "${ot_result%.jsonl}_chair.json" >> "$MANIFEST"
    done
  done
done

run_job() {
  local index="$1" gpu="$2"
  local method="${JOB_METHODS[$index]}"
  local seed="${JOB_SEEDS[$index]}"
  local topk="${JOB_TOPKS[$index]}"
  local visual_tokens="${JOB_VISUAL_TOKENS[$index]}"
  local ids_file="${JOB_IDS_FILES[$index]}"
  local result="${JOB_RESULTS[$index]}"
  local exp_folder="$VISTA_EXP_FOLDER"
  local log_file="$GRID_DIR/logs/${method}_seed${seed}_m${topk}_k${visual_tokens}.log"
  local backup stats_file
  local -a method_args=(
    --vsv
    --vsv-lambda "$VSV_LAMBDA"
    --logits-aug
    --logits-layers "$LOGITS_LAYERS"
    --logits-alpha "$GAMMA"
  )

  if [[ "$method" == "ot" ]]; then
    exp_folder="$OT_EXP_FOLDER"
    method_args+=(
      --use-ot-bary-sla
      --ot-topk "$topk"
      --ot-visual-tokens "$visual_tokens"
      --ot-sinkhorn-iters "$OT_SINKHORN_ITERS"
      --ot-epsilon "$OT_EPSILON"
      --ot-layer-temperature "$OT_LAYER_TEMPERATURE"
      --ot-log-stats
    )
  fi

  if [[ -f "$result" ]]; then
    backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"
    mv "$result" "$backup"
    echo "[GPU $gpu] preserved partial result as: $backup"
  fi
  stats_file="${result%.jsonl}_ot_stats.jsonl"
  if [[ -f "$stats_file" ]]; then
    backup="${stats_file}.partial.$(date +%Y%m%d_%H%M%S)"
    mv "$stats_file" "$backup"
  fi

  echo "[GPU $gpu] start method=$method seed=$seed topk=$topk visual_tokens=$visual_tokens"
  CUDA_VISIBLE_DEVICES="$gpu" python chair_eval.py \
    --exp_folder "$exp_folder" \
    --model "$MODEL" \
    --data-path "$VISTA_COCO_ROOT/val2014" \
    --subset-size "$SUBSET_SIZE" \
    --subset-ids-file "$ids_file" \
    --seed "$seed" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${method_args[@]}" > "$log_file" 2>&1

  if ! is_complete_result "$result"; then
    echo "Incomplete result: $result" >&2
    return 1
  fi
}

run_gpu_worker() {
  local worker_index="$1"
  local gpu="${GPUS[$worker_index]}"
  local index
  for ((index=worker_index; index<${#JOB_METHODS[@]}; index+=${#GPUS[@]})); do
    run_job "$index" "$gpu"
  done
}

echo "Seeds: ${SEEDS[*]}"
echo "Grid: topk={${TOPKS[*]}} visual_tokens={${VISUAL_TOKENS[*]}}"
echo "Fixed lambda=$VSV_LAMBDA gamma=$GAMMA iters=$OT_SINKHORN_ITERS epsilon=$OT_EPSILON layer_temperature=$OT_LAYER_TEMPERATURE"
echo "Pending generation jobs: ${#JOB_METHODS[@]}"

declare -a WORKER_PIDS=()
for ((worker=0; worker<${#GPUS[@]} && worker<${#JOB_METHODS[@]}; worker+=1)); do
  run_gpu_worker "$worker" &
  WORKER_PIDS+=("$!")
done

generation_failed=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    generation_failed=1
  fi
done
if (( generation_failed != 0 )); then
  echo "At least one generation failed; see $GRID_DIR/logs" >&2
  exit 1
fi

while IFS=$'\t' read -r method seed topk visual_tokens gpu ids_file result chair_json; do
  [[ "$method" == "method" ]] && continue
  if [[ ! -f "$chair_json" ]] || [[ "$chair_json" -ot "$result" ]]; then
    python chair_ans.py \
      --cap_file "$result" \
      --coco_path "$VISTA_COCO_ROOT/annotations" \
      --cache "$VISTA_COCO_ROOT/chair.pkl" \
      --save_path "$chair_json" \
      > "$GRID_DIR/logs/chair_${method}_seed${seed}_m${topk}_k${visual_tokens}.log" 2>&1
  fi
done < "$MANIFEST"

python scripts/summarize_chair_ot_multiseed_grid.py \
  --manifest "$MANIFEST" \
  --per-seed-csv "$GRID_DIR/per_seed.csv" \
  --aggregate-csv "$GRID_DIR/aggregate.csv" \
  --markdown "$GRID_DIR/summary.md" \
  --gamma "$GAMMA" \
  --vsv-lambda "$VSV_LAMBDA"

echo "Grid complete: $GRID_DIR/aggregate.csv"
