#!/usr/bin/env bash
# Paired eight-GPU search for unpooled, layer-aligned attention OT.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a SEEDS <<< "${SEEDS:-2024}"
read -r -a GAMMAS <<< "${GAMMAS:-0.3 0.5}"
read -r -a LAYER_TEMPERATURES <<< "${LAYER_TEMPERATURES:-0.1 0.2}"
read -r -a ATTENTION_POWERS <<< "${ATTENTION_POWERS:-0.5 1.0}"
read -r -a UNIFORM_MIXES <<< "${UNIFORM_MIXES:-0.0 0.02}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"

MODEL="${MODEL:-llava-1.5}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
OT_TOPK="${OT_TOPK:-16}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-50}"
OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"
OT_EPSILON="${OT_EPSILON:-0.05}"

SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_attention_grid}"
VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-chair_ot_attention_grid_vista}"
OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_ot_attention_grid_otattn}"

export VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
export NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"
if [[ -z "${HF_HOME:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" && -d /data/sun_yuxi/huggingface ]]; then
  export HF_HOME=/data/sun_yuxi/huggingface
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then
  for candidate in \
    /data/sun_yuxi/models/llava-v1.5-7b \
    /data/sun_yuxi/models/llava-1.5-7b-hf \
    /home/ljc/code/models/llava-v1.5-7b; do
    if [[ -f "$candidate/config.json" ]]; then
      export VISTA_LLAVA_MODEL_PATH="$candidate"
      break
    fi
  done
fi
if [[ ! -d "$VISTA_COCO_ROOT/val2014" && -d /home/ljc/code/data/val2014 ]]; then
  export VISTA_COCO_ROOT=/home/ljc/code/data
fi

if (( ${#SEEDS[@]} == 0 || ${#GAMMAS[@]} == 0 || ${#LAYER_TEMPERATURES[@]} == 0 || ${#ATTENTION_POWERS[@]} == 0 || ${#UNIFORM_MIXES[@]} == 0 || ${#GPUS[@]} == 0 )); then
  echo "All search arrays and GPU_IDS must be non-empty." >&2
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
    "$SCRIPT_DIR" "$VISTA_EXP_FOLDER" "$MODEL" \
    "$(base_stem "$seed" "$gamma")" "$MAX_NEW_TOKENS"
}

ot_result_path() {
  local seed="$1" gamma="$2" layer_temperature="$3" power="$4" mix="$5"
  printf '%s/exp_results/%s/%s/%s_otattn_nodust_layerhid_lmhead_tlogit_m%s_kunpooled_it%s_tol%s_eps%s_ltemp%s_apow%s_amix%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$OT_EXP_FOLDER" "$MODEL" \
    "$(base_stem "$seed" "$gamma")" "$OT_TOPK" "$OT_SINKHORN_ITERS" \
    "$OT_SINKHORN_TOLERANCE" "$OT_EPSILON" "$layer_temperature" "$power" "$mix" "$MAX_NEW_TOKENS"
}

is_complete_result() {
  [[ -f "$1" ]] && [[ "$(wc -l < "$1")" -eq "$SUBSET_SIZE" ]]
}

printf 'method\tseed\tgamma\tlayer_temperature\tattention_power\tuniform_mix\tgpu\tids_file\tresult_jsonl\tchair_json\tstats_jsonl\n' > "$MANIFEST"
declare -a JOB_METHODS=() JOB_SEEDS=() JOB_GAMMAS=() JOB_LAYER_TEMPERATURES=() JOB_POWERS=() JOB_MIXES=() JOB_IDS_FILES=() JOB_RESULTS=()
pending_count=0

enqueue() {
  local method="$1" seed="$2" gamma="$3" layer_temperature="$4" power="$5" mix="$6" ids_file="$7" result="$8" gpu
  if is_complete_result "$result"; then
    gpu=-1
  else
    gpu="${GPUS[$((pending_count % ${#GPUS[@]}))]}"
    JOB_METHODS+=("$method"); JOB_SEEDS+=("$seed"); JOB_GAMMAS+=("$gamma")
    JOB_LAYER_TEMPERATURES+=("$layer_temperature"); JOB_POWERS+=("$power"); JOB_MIXES+=("$mix")
    JOB_IDS_FILES+=("$ids_file"); JOB_RESULTS+=("$result")
    ((pending_count += 1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$method" "$seed" "$gamma" "$layer_temperature" "$power" "$mix" "$gpu" "$ids_file" "$result" "${result%.jsonl}_chair.json" "${result%.jsonl}_ot_stats.jsonl" >> "$MANIFEST"
}

for seed in "${SEEDS[@]}"; do
  ids_file="$SWEEP_DIR/manifests/seed_${seed}_ids.txt"
  if [[ ! -f "$ids_file" ]] || [[ "$(wc -l < "$ids_file")" -ne "$SUBSET_SIZE" ]]; then
    "$PYTHON_BIN" scripts/make_chair_seed_manifest.py \
      --data-path "$VISTA_COCO_ROOT/val2014" --seed "$seed" \
      --subset-size "$SUBSET_SIZE" --output "$ids_file"
  fi
  for gamma in "${GAMMAS[@]}"; do
    enqueue vista "$seed" "$gamma" baseline baseline baseline "$ids_file" "$(vista_result_path "$seed" "$gamma")"
    for layer_temperature in "${LAYER_TEMPERATURES[@]}"; do
      for power in "${ATTENTION_POWERS[@]}"; do
        for mix in "${UNIFORM_MIXES[@]}"; do
          enqueue ot "$seed" "$gamma" "$layer_temperature" "$power" "$mix" "$ids_file" "$(ot_result_path "$seed" "$gamma" "$layer_temperature" "$power" "$mix")"
        done
      done
    done
  done
done

run_job() {
  local index="$1" gpu="$2"
  local method="${JOB_METHODS[$index]}" seed="${JOB_SEEDS[$index]}" gamma="${JOB_GAMMAS[$index]}"
  local layer_temperature="${JOB_LAYER_TEMPERATURES[$index]}" power="${JOB_POWERS[$index]}" mix="${JOB_MIXES[$index]}"
  local ids_file="${JOB_IDS_FILES[$index]}" result="${JOB_RESULTS[$index]}"
  local exp_folder="$VISTA_EXP_FOLDER" backup stats_file
  local log_file="$SWEEP_DIR/logs/${method}_seed${seed}_g${gamma}_lt${layer_temperature}_p${power}_mix${mix}.log"
  local -a method_args=(--vsv --vsv-lambda "$VSV_LAMBDA" --logits-aug --logits-layers "$LOGITS_LAYERS" --logits-alpha "$gamma")
  if [[ "$method" == ot ]]; then
    exp_folder="$OT_EXP_FOLDER"
    method_args+=(
      --use-ot-bary-sla --ot-attention-visual-marginal --ot-topk "$OT_TOPK"
      --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" --ot-sinkhorn-tolerance "$OT_SINKHORN_TOLERANCE"
      --ot-epsilon "$OT_EPSILON" --ot-layer-temperature "$layer_temperature"
      --ot-attention-power "$power" --ot-attention-uniform-mix "$mix" --ot-log-stats
    )
  fi
  if [[ -f "$result" ]]; then backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"; mv "$result" "$backup"; fi
  stats_file="${result%.jsonl}_ot_stats.jsonl"
  if [[ -f "$stats_file" ]]; then backup="${stats_file}.partial.$(date +%Y%m%d_%H%M%S)"; mv "$stats_file" "$backup"; fi
  echo "[GPU $gpu] start method=$method seed=$seed gamma=$gamma ltemp=$layer_temperature power=$power mix=$mix"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_eval.py \
    --exp_folder "$exp_folder" --model "$MODEL" --data-path "$VISTA_COCO_ROOT/val2014" \
    --subset-size "$SUBSET_SIZE" --subset-ids-file "$ids_file" --seed "$seed" \
    --max-new-tokens "$MAX_NEW_TOKENS" "${method_args[@]}" > "$log_file" 2>&1
  is_complete_result "$result" || { echo "Incomplete result: $result" >&2; return 1; }
}

run_worker() {
  local worker_index="$1" gpu="${GPUS[$worker_index]}" index
  for ((index=worker_index; index<${#JOB_METHODS[@]}; index+=${#GPUS[@]})); do
    run_job "$index" "$gpu"
  done
}

echo "Seeds: ${SEEDS[*]}; GPUs: ${GPUS[*]}"
echo "Grid: gamma={${GAMMAS[*]}} ltemp={${LAYER_TEMPERATURES[*]}} power={${ATTENTION_POWERS[*]}} mix={${UNIFORM_MIXES[*]}}"
echo "Fixed unpooled attention OT: topk=$OT_TOPK eps=$OT_EPSILON max_iters=$OT_SINKHORN_ITERS tol=$OT_SINKHORN_TOLERANCE"
echo "Pending generation jobs: ${#JOB_METHODS[@]}"
declare -a WORKER_PIDS=()
for ((worker=0; worker<${#GPUS[@]} && worker<${#JOB_METHODS[@]}; worker+=1)); do
  run_worker "$worker" & WORKER_PIDS+=("$!")
done
generation_failed=0
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || generation_failed=1; done
if (( generation_failed != 0 )); then
  echo "At least one generation failed; see $SWEEP_DIR/logs" >&2
  exit 1
fi

while IFS=$'\t' read -r method seed gamma layer_temperature power mix gpu ids_file result chair_json stats_jsonl; do
  [[ "$method" == method ]] && continue
  if [[ ! -f "$chair_json" ]] || [[ "$chair_json" -ot "$result" ]]; then
    "$PYTHON_BIN" chair_ans.py --cap_file "$result" --coco_path "$VISTA_COCO_ROOT/annotations" \
      --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$chair_json" \
      > "$SWEEP_DIR/logs/chair_${method}_seed${seed}_g${gamma}_lt${layer_temperature}_p${power}_mix${mix}.log" 2>&1
  fi
  if [[ "$method" == ot && ! -f "$stats_jsonl" ]]; then
    echo "Missing OT diagnostics: $stats_jsonl" >&2
    exit 1
  fi
done < "$MANIFEST"

"$PYTHON_BIN" scripts/summarize_chair_ot_attention_grid.py \
  --manifest "$MANIFEST" --summary-csv "$SWEEP_DIR/summary.csv" \
  --weight-csv "$SWEEP_DIR/layer_weight_stats.csv" --markdown "$SWEEP_DIR/summary.md"
echo "Sweep complete: $SWEEP_DIR/summary.csv"
