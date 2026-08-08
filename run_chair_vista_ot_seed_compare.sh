#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Greedy decoding is deterministic. These seeds therefore measure robustness
# across independently seeded, paired COCO validation subsets.
read -r -a SEEDS <<< "${SEEDS:-1994 2024 3407 42 1234}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5}"

MODEL="${MODEL:-llava-1.5}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
GAMMA="${GAMMA:-0.3}"

# Best F1 configuration in the completed top-k / visual-token sweep.
OT_TOPK="${OT_TOPK:-32}"
OT_VISUAL_TOKENS="${OT_VISUAL_TOKENS:-81}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-3}"
OT_EPSILON="${OT_EPSILON:-0.05}"

COMPARE_DIR="${COMPARE_DIR:-$SCRIPT_DIR/exp_results/chair_vista_ot_seed_compare}"
VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-chair_vista_ot_seed_compare_vista}"
OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_vista_ot_seed_compare_ot}"

export VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
export NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"

# The compute nodes have no Hub access. Prefer the shared data-disk cache that
# already contains LLaVA's CLIP vision tower, while preserving explicit user
# cache settings when they are provided.
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

if (( ${#SEEDS[@]} == 0 || ${#GPUS[@]} == 0 )); then
  echo "SEEDS and GPU_IDS must not be empty." >&2
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

mkdir -p "$COMPARE_DIR/manifests" "$COMPARE_DIR/logs"
MANIFEST="$COMPARE_DIR/manifest.tsv"

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
  local seed="$1"
  printf '%s/exp_results/%s/%s/%s_otbary_m%s_k%s_it%s_eps%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$OT_EXP_FOLDER" "$MODEL" "$(base_stem "$seed")" \
    "$OT_TOPK" "$OT_VISUAL_TOKENS" "$OT_SINKHORN_ITERS" "$OT_EPSILON" "$MAX_NEW_TOKENS"
}

is_complete_result() {
  local result="$1"
  [[ -f "$result" ]] && [[ "$(wc -l < "$result")" -eq "$SUBSET_SIZE" ]]
}

printf 'seed\tids_file\tvista_result_jsonl\tot_result_jsonl\tvista_chair_json\tot_chair_json\n' > "$MANIFEST"
for seed in "${SEEDS[@]}"; do
  ids_file="$COMPARE_DIR/manifests/seed_${seed}_ids.txt"
  if [[ ! -f "$ids_file" ]] || [[ "$(wc -l < "$ids_file")" -ne "$SUBSET_SIZE" ]]; then
    python scripts/make_chair_seed_manifest.py \
      --data-path "$VISTA_COCO_ROOT/val2014" \
      --seed "$seed" \
      --subset-size "$SUBSET_SIZE" \
      --output "$ids_file"
  fi
  vista_result="$(vista_result_path "$seed")"
  ot_result="$(ot_result_path "$seed")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$seed" "$ids_file" "$vista_result" "$ot_result" \
    "${vista_result%.jsonl}_chair.json" "${ot_result%.jsonl}_chair.json" >> "$MANIFEST"
done

run_one() {
  local seed="$1" gpu="$2" ids_file="$3" mode="$4"
  local exp_folder="$VISTA_EXP_FOLDER"
  local result="$(vista_result_path "$seed")"
  local log_file="$COMPARE_DIR/logs/seed_${seed}_${mode}.log"
  local -a method_args=(
    --vsv
    --vsv-lambda "$VSV_LAMBDA"
    --logits-aug
    --logits-layers "$LOGITS_LAYERS"
    --logits-alpha "$GAMMA"
  )

  if [[ "$mode" == "ot" ]]; then
    exp_folder="$OT_EXP_FOLDER"
    result="$(ot_result_path "$seed")"
    method_args+=(
      --use-ot-bary-sla
      --ot-topk "$OT_TOPK"
      --ot-visual-tokens "$OT_VISUAL_TOKENS"
      --ot-sinkhorn-iters "$OT_SINKHORN_ITERS"
      --ot-epsilon "$OT_EPSILON"
      --ot-log-stats
    )
  fi

  if is_complete_result "$result"; then
    echo "[GPU $gpu] skip complete seed=$seed mode=$mode"
    return
  fi
  if [[ -f "$result" ]]; then
    echo "Partial result exists and will not be overwritten: $result" >&2
    return 1
  fi

  echo "[GPU $gpu] start seed=$seed mode=$mode"
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

run_seed_worker() {
  local worker_index="$1"
  local seed gpu ids_file
  for ((index=worker_index; index<${#SEEDS[@]}; index+=${#GPUS[@]})); do
    seed="${SEEDS[$index]}"
    gpu="${GPUS[$worker_index]}"
    ids_file="$COMPARE_DIR/manifests/seed_${seed}_ids.txt"
    run_one "$seed" "$gpu" "$ids_file" vista
    run_one "$seed" "$gpu" "$ids_file" ot
  done
}

echo "Seeds: ${SEEDS[*]}"
echo "VISTA: lambda=$VSV_LAMBDA, gamma=$GAMMA, layers=$LOGITS_LAYERS"
echo "VISTA-OT: topk=$OT_TOPK, visual_tokens=$OT_VISUAL_TOKENS, iters=$OT_SINKHORN_ITERS, epsilon=$OT_EPSILON"
echo "Each VISTA/OT pair shares the same seed-specific val2014 image IDs."

declare -a WORKER_PIDS=()
for ((worker=0; worker<${#GPUS[@]} && worker<${#SEEDS[@]}; worker+=1)); do
  run_seed_worker "$worker" &
  WORKER_PIDS+=("$!")
done

generation_failed=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    generation_failed=1
  fi
done
if (( generation_failed != 0 )); then
  echo "At least one generation failed; see $COMPARE_DIR/logs" >&2
  exit 1
fi

while IFS=$'\t' read -r seed ids_file vista_result ot_result vista_chair ot_chair; do
  [[ "$seed" == "seed" ]] && continue
  if [[ ! -f "$vista_chair" ]] || [[ "$vista_chair" -ot "$vista_result" ]]; then
    python chair_ans.py --cap_file "$vista_result" --coco_path "$VISTA_COCO_ROOT/annotations" --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$vista_chair"
  fi
  if [[ ! -f "$ot_chair" ]] || [[ "$ot_chair" -ot "$ot_result" ]]; then
    python chair_ans.py --cap_file "$ot_result" --coco_path "$VISTA_COCO_ROOT/annotations" --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$ot_chair"
  fi
done < "$MANIFEST"

python scripts/summarize_chair_seed_compare.py \
  --manifest "$MANIFEST" \
  --csv "$COMPARE_DIR/summary.csv" \
  --markdown "$COMPARE_DIR/summary.md"

echo "Comparison complete: $COMPARE_DIR/summary.csv"
