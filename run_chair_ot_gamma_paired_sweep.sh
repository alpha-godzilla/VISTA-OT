#!/usr/bin/env bash
# Paired VISTA / OT ablation over the SLA mixing coefficient (gamma).
# Every VISTA/OT pair at a given seed and gamma uses exactly the same images.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

read -r -a SEEDS <<< "${SEEDS:-1994 2024}"
read -r -a GAMMAS <<< "${GAMMAS:-0.1 0.3 0.5 0.7}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"

MODEL="${MODEL:-llava-1.5}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
OT_TOPK="${OT_TOPK:-16}"
OT_VISUAL_TOKENS="${OT_VISUAL_TOKENS:-64}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-3}"
OT_EPSILON="${OT_EPSILON:-0.05}"
OT_LAYER_TEMPERATURE="${OT_LAYER_TEMPERATURE:-0.1}"

SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_gamma_paired_sweep}"
VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-chair_ot_gamma_paired_sweep_vista}"
OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_ot_gamma_paired_sweep_ot}"

export VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
export NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"
if [[ -z "${HF_HOME:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" ]] && [[ -d /data/sun_yuxi/huggingface ]]; then
  export HF_HOME=/data/sun_yuxi/huggingface
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then
  for candidate in /data/sun_yuxi/models/llava-v1.5-7b /data/sun_yuxi/models/llava-1.5-7b-hf; do
    if [[ -d "$candidate" ]]; then
      export VISTA_LLAVA_MODEL_PATH="$candidate"
      break
    fi
  done
fi

if (( ${#SEEDS[@]} == 0 || ${#GAMMAS[@]} == 0 || ${#GPUS[@]} == 0 )); then
  echo "SEEDS, GAMMAS, and GPU_IDS must not be empty." >&2
  exit 1
fi
if [[ ! "$SUBSET_SIZE" =~ ^[1-9][0-9]*$ ]]; then
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

mkdir -p "$SWEEP_DIR/manifests" "$SWEEP_DIR/logs"
MANIFEST="$SWEEP_DIR/manifest.tsv"

base_stem() {
  local seed="$1" gamma="$2"
  printf 'seed%s_vsv_lambda_%s_logaug_loglayer_%s_logalpha_%s' \
    "$seed" "$VSV_LAMBDA" "$LOGITS_LAYERS" "$gamma"
}

vista_result_path() {
  local seed="$1" gamma="$2"
  printf '%s/exp_results/%s/%s/%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$VISTA_EXP_FOLDER" "$MODEL" "$(base_stem "$seed" "$gamma")" "$MAX_NEW_TOKENS"
}

ot_result_path() {
  local seed="$1" gamma="$2"
  printf '%s/exp_results/%s/%s/%s_otbary_vdust_tlogit_m%s_k%s_it%s_eps%s_ltemp%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$OT_EXP_FOLDER" "$MODEL" "$(base_stem "$seed" "$gamma")" \
    "$OT_TOPK" "$OT_VISUAL_TOKENS" "$OT_SINKHORN_ITERS" "$OT_EPSILON" "$OT_LAYER_TEMPERATURE" "$MAX_NEW_TOKENS"
}

is_complete_result() {
  local result="$1"
  [[ -f "$result" ]] && [[ "$(wc -l < "$result")" -eq "$SUBSET_SIZE" ]]
}

printf 'method\tseed\tgamma\tgpu\tids_file\tresult_jsonl\tchair_json\n' > "$MANIFEST"
declare -a JOB_METHODS=() JOB_SEEDS=() JOB_GAMMAS=() JOB_IDS_FILES=() JOB_RESULTS=()
pending_count=0
for seed in "${SEEDS[@]}"; do
  ids_file="$SWEEP_DIR/manifests/seed_${seed}_ids.txt"
  if [[ ! -f "$ids_file" ]] || [[ "$(wc -l < "$ids_file")" -ne "$SUBSET_SIZE" ]]; then
    python scripts/make_chair_seed_manifest.py --data-path "$VISTA_COCO_ROOT/val2014" --seed "$seed" --subset-size "$SUBSET_SIZE" --output "$ids_file"
  fi
  for gamma in "${GAMMAS[@]}"; do
    for method in vista ot; do
      if [[ "$method" == vista ]]; then result="$(vista_result_path "$seed" "$gamma")"; else result="$(ot_result_path "$seed" "$gamma")"; fi
      if is_complete_result "$result"; then gpu=-1; else
        gpu="${GPUS[$((pending_count % ${#GPUS[@]}))]}"
        JOB_METHODS+=("$method"); JOB_SEEDS+=("$seed"); JOB_GAMMAS+=("$gamma"); JOB_IDS_FILES+=("$ids_file"); JOB_RESULTS+=("$result")
        ((pending_count += 1))
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$method" "$seed" "$gamma" "$gpu" "$ids_file" "$result" "${result%.jsonl}_chair.json" >> "$MANIFEST"
    done
  done
done

run_job() {
  local index="$1"
  local gpu="$2"
  local method="${JOB_METHODS[$index]}"
  local seed="${JOB_SEEDS[$index]}"
  local gamma="${JOB_GAMMAS[$index]}"
  local ids_file="${JOB_IDS_FILES[$index]}"
  local result="${JOB_RESULTS[$index]}"
  local exp_folder="$VISTA_EXP_FOLDER"
  local log_file="$SWEEP_DIR/logs/${method}_seed${seed}_gamma${gamma}.log" backup stats_file
  local -a method_args=(--vsv --vsv-lambda "$VSV_LAMBDA" --logits-aug --logits-layers "$LOGITS_LAYERS" --logits-alpha "$gamma")
  if [[ "$method" == ot ]]; then
    exp_folder="$OT_EXP_FOLDER"
    method_args+=(--use-ot-bary-sla --ot-topk "$OT_TOPK" --ot-visual-tokens "$OT_VISUAL_TOKENS" --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" --ot-epsilon "$OT_EPSILON" --ot-layer-temperature "$OT_LAYER_TEMPERATURE" --ot-log-stats)
  fi
  if [[ -f "$result" ]]; then backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"; mv "$result" "$backup"; fi
  stats_file="${result%.jsonl}_ot_stats.jsonl"
  if [[ -f "$stats_file" ]]; then backup="${stats_file}.partial.$(date +%Y%m%d_%H%M%S)"; mv "$stats_file" "$backup"; fi
  echo "[GPU $gpu] start method=$method seed=$seed gamma=$gamma"
  CUDA_VISIBLE_DEVICES="$gpu" python chair_eval.py --exp_folder "$exp_folder" --model "$MODEL" --data-path "$VISTA_COCO_ROOT/val2014" --subset-size "$SUBSET_SIZE" --subset-ids-file "$ids_file" --seed "$seed" --max-new-tokens "$MAX_NEW_TOKENS" "${method_args[@]}" > "$log_file" 2>&1
  is_complete_result "$result" || { echo "Incomplete result: $result" >&2; return 1; }
}

run_gpu_worker() {
  local worker_index="$1"
  local gpu="${GPUS[$worker_index]}"
  local index
  for ((index=worker_index; index<${#JOB_METHODS[@]}; index+=${#GPUS[@]})); do run_job "$index" "$gpu"; done
}

echo "Seeds: ${SEEDS[*]}; gammas: ${GAMMAS[*]}; GPUs: ${GPUS[*]}"
echo "OT: topk=$OT_TOPK visual_tokens=$OT_VISUAL_TOKENS iters=$OT_SINKHORN_ITERS epsilon=$OT_EPSILON layer_temperature=$OT_LAYER_TEMPERATURE"
echo "Pending generation jobs: ${#JOB_METHODS[@]}"
declare -a WORKER_PIDS=()
for ((worker=0; worker<${#GPUS[@]} && worker<${#JOB_METHODS[@]}; worker+=1)); do run_gpu_worker "$worker" & WORKER_PIDS+=("$!"); done
generation_failed=0
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || generation_failed=1; done
if (( generation_failed != 0 )); then echo "At least one generation failed; see $SWEEP_DIR/logs" >&2; exit 1; fi

while IFS=$'\t' read -r method seed gamma gpu ids_file result chair_json; do
  [[ "$method" == method ]] && continue
  if [[ ! -f "$chair_json" ]] || [[ "$chair_json" -ot "$result" ]]; then
    python chair_ans.py --cap_file "$result" --coco_path "$VISTA_COCO_ROOT/annotations" --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$chair_json" > "$SWEEP_DIR/logs/chair_${method}_seed${seed}_gamma${gamma}.log" 2>&1
  fi
done < "$MANIFEST"

python scripts/summarize_chair_ot_gamma_paired_sweep.py --manifest "$MANIFEST" --csv "$SWEEP_DIR/summary.csv" --markdown "$SWEEP_DIR/summary.md" --ot-topk "$OT_TOPK" --ot-visual-tokens "$OT_VISUAL_TOKENS" --vsv-lambda "$VSV_LAMBDA"
echo "Sweep complete: $SWEEP_DIR/summary.csv"
