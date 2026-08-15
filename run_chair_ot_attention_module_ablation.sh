#!/usr/bin/env bash
# Paired ablations of coverage-aware marginals and adaptive alpha on 8 GPUs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a SEEDS <<< "${SEEDS:-1994 2024 3407}"
read -r -a COVERAGE_BETAS <<< "${COVERAGE_BETAS:-0.1 0.25 0.5}"
read -r -a ADAPTIVE_MIN_RATIOS <<< "${ADAPTIVE_MIN_RATIOS:-0 0.25 0.5}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
RUN_COVERAGE="${RUN_COVERAGE:-1}"

MODEL="${MODEL:-llava-1.5}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
LOGITS_ALPHA="${LOGITS_ALPHA:-0.3}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
OT_TOPK="${OT_TOPK:-16}"
OT_EPSILON="${OT_EPSILON:-0.05}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-50}"
OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"
OT_LAYER_TEMPERATURE="${OT_LAYER_TEMPERATURE:-0.06}"
OT_ATTENTION_POWER="${OT_ATTENTION_POWER:-0.75}"
OT_UNIFORM_MIX="${OT_UNIFORM_MIX:-0.02}"
OT_COVERAGE_EPSILON="${OT_COVERAGE_EPSILON:-0.1}"

SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_attention_module_ablation}"
VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-chair_ot_attention_module_ablation_vista}"
OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_ot_attention_module_ablation_otattn}"

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

if [[ "$RUN_COVERAGE" != 0 && "$RUN_COVERAGE" != 1 ]]; then
  echo "RUN_COVERAGE must be 0 or 1." >&2
  exit 1
fi
if (( ${#SEEDS[@]} == 0 || ${#ADAPTIVE_MIN_RATIOS[@]} == 0 || ${#GPUS[@]} == 0 )) || { [[ "$RUN_COVERAGE" == 1 ]] && (( ${#COVERAGE_BETAS[@]} == 0 )); }; then
  echo "SEEDS, module grids, and GPU_IDS must be non-empty." >&2
  exit 1
fi
if [[ ! -d "$VISTA_COCO_ROOT/val2014" || ! -f "${VISTA_LLAVA_MODEL_PATH:-}/config.json" ]]; then
  echo "Set VISTA_COCO_ROOT and VISTA_LLAVA_MODEL_PATH before running." >&2
  exit 1
fi

mkdir -p "$SWEEP_DIR/manifests" "$SWEEP_DIR/logs"
MANIFEST="$SWEEP_DIR/manifest.tsv"

base_stem() { printf 'seed%s_vsv_lambda_%s_logaug_loglayer_%s_logalpha_%s' "$1" "$VSV_LAMBDA" "$LOGITS_LAYERS" "$LOGITS_ALPHA"; }
vista_result() { printf '%s/exp_results/%s/%s/%s_greedy_max_new_tokens_%s.jsonl' "$SCRIPT_DIR" "$VISTA_EXP_FOLDER" "$MODEL" "$(base_stem "$1")" "$MAX_NEW_TOKENS"; }
ot_result() {
  local seed="$1" suffix="$2"
  printf '%s/exp_results/%s/%s/%s_otattn_nodust_layerhid_lmhead_tlogit_m%s_kunpooled_it%s_tol%s_eps%s_ltemp%s_apow%s_amix%s%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$OT_EXP_FOLDER" "$MODEL" "$(base_stem "$seed")" "$OT_TOPK" "$OT_SINKHORN_ITERS" "$OT_SINKHORN_TOLERANCE" "$OT_EPSILON" "$OT_LAYER_TEMPERATURE" "$OT_ATTENTION_POWER" "$OT_UNIFORM_MIX" "$suffix" "$MAX_NEW_TOKENS"
}
is_complete() { [[ -f "$1" && "$(wc -l < "$1")" -eq "$SUBSET_SIZE" ]]; }
canonical_python_float() { "$PYTHON_BIN" -c 'import sys; print(float(sys.argv[1]))' "$1"; }

printf 'method\tsetting\tseed\tgpu\tids_file\tresult_jsonl\tchair_json\n' > "$MANIFEST"
declare -a JOB_METHODS=() JOB_SETTINGS=() JOB_SEEDS=() JOB_IDS=() JOB_RESULTS=()
pending=0
enqueue() {
  local method="$1" setting="$2" seed="$3" ids="$4" result="$5" gpu
  if is_complete "$result"; then gpu=-1; else
    gpu="${GPUS[$((pending % ${#GPUS[@]}))]}"
    JOB_METHODS+=("$method"); JOB_SETTINGS+=("$setting"); JOB_SEEDS+=("$seed"); JOB_IDS+=("$ids"); JOB_RESULTS+=("$result")
    ((pending += 1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$method" "$setting" "$seed" "$gpu" "$ids" "$result" "${result%.jsonl}_chair.json" >> "$MANIFEST"
}

for seed in "${SEEDS[@]}"; do
  ids="$SWEEP_DIR/manifests/seed_${seed}_ids.txt"
  if [[ ! -f "$ids" || "$(wc -l < "$ids")" -ne "$SUBSET_SIZE" ]]; then
    "$PYTHON_BIN" scripts/make_chair_seed_manifest.py --data-path "$VISTA_COCO_ROOT/val2014" --seed "$seed" --subset-size "$SUBSET_SIZE" --output "$ids"
  fi
  enqueue vista original "$seed" "$ids" "$(vista_result "$seed")"
  enqueue ot_base base "$seed" "$ids" "$(ot_result "$seed" '')"
  if [[ "$RUN_COVERAGE" == 1 ]]; then
    for raw_beta in "${COVERAGE_BETAS[@]}"; do
      beta="$(canonical_python_float "$raw_beta")"
      enqueue coverage "$beta" "$seed" "$ids" "$(ot_result "$seed" "_covbeta${beta}_coveps${OT_COVERAGE_EPSILON}")"
    done
  fi
  for raw_ratio in "${ADAPTIVE_MIN_RATIOS[@]}"; do
    ratio="$(canonical_python_float "$raw_ratio")"
    enqueue adaptive_alpha "$ratio" "$seed" "$ids" "$(ot_result "$seed" "_adaptamin${ratio}")"
  done
done

run_job() {
  local index="$1" gpu="$2" method="${JOB_METHODS[$index]}" setting="${JOB_SETTINGS[$index]}" seed="${JOB_SEEDS[$index]}" ids="${JOB_IDS[$index]}" result="${JOB_RESULTS[$index]}"
  local exp_folder="$VISTA_EXP_FOLDER" backup log_file
  local -a args=(--vsv --vsv-lambda "$VSV_LAMBDA" --logits-aug --logits-layers "$LOGITS_LAYERS" --logits-alpha "$LOGITS_ALPHA")
  if [[ "$method" != vista ]]; then
    exp_folder="$OT_EXP_FOLDER"
    args+=(--use-ot-bary-sla --ot-attention-visual-marginal --ot-topk "$OT_TOPK" --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" --ot-sinkhorn-tolerance "$OT_SINKHORN_TOLERANCE" --ot-epsilon "$OT_EPSILON" --ot-layer-temperature "$OT_LAYER_TEMPERATURE" --ot-attention-power "$OT_ATTENTION_POWER" --ot-attention-uniform-mix "$OT_UNIFORM_MIX" --ot-log-stats)
    if [[ "$method" == coverage ]]; then args+=(--ot-attention-coverage-beta "$setting" --ot-attention-coverage-epsilon "$OT_COVERAGE_EPSILON"); fi
    if [[ "$method" == adaptive_alpha ]]; then args+=(--ot-adaptive-alpha --ot-adaptive-alpha-min-ratio "$setting"); fi
  fi
  if [[ -f "$result" ]]; then backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"; mv "$result" "$backup"; fi
  log_file="$SWEEP_DIR/logs/${method}_setting${setting}_seed${seed}.log"
  echo "[GPU $gpu] start method=$method setting=$setting seed=$seed"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_eval.py --exp_folder "$exp_folder" --model "$MODEL" --data-path "$VISTA_COCO_ROOT/val2014" --subset-size "$SUBSET_SIZE" --subset-ids-file "$ids" --seed "$seed" --max-new-tokens "$MAX_NEW_TOKENS" "${args[@]}" > "$log_file" 2>&1
  is_complete "$result" || { echo "Incomplete result: $result" >&2; return 1; }
}
run_worker() { local worker="$1"; local gpu="${GPUS[$worker]}"; local index; for ((index=worker; index<${#JOB_METHODS[@]}; index+=${#GPUS[@]})); do run_job "$index" "$gpu"; done; }

echo "VISTA + OT module ablation: ${#JOB_METHODS[@]} pending jobs on GPUs ${GPUS[*]}"
declare -a PIDS=()
for ((worker=0; worker<${#GPUS[@]} && worker<${#JOB_METHODS[@]}; worker+=1)); do run_worker "$worker" & PIDS+=("$!"); done
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "Generation failed; see $SWEEP_DIR/logs" >&2; exit 1; }

while IFS=$'\t' read -r method setting seed gpu ids result chair_json; do
  [[ "$method" == method ]] && continue
  if [[ ! -f "$chair_json" || "$chair_json" -ot "$result" ]]; then
    "$PYTHON_BIN" chair_ans.py --cap_file "$result" --coco_path "$VISTA_COCO_ROOT/annotations" --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$chair_json" > "$SWEEP_DIR/logs/chair_${method}_setting${setting}_seed${seed}.log" 2>&1
  fi
done < "$MANIFEST"

"$PYTHON_BIN" scripts/summarize_chair_ot_attention_modules.py --manifest "$MANIFEST" --csv "$SWEEP_DIR/summary.csv" --markdown "$SWEEP_DIR/summary.md"
echo "Ablation complete: $SWEEP_DIR/summary.md"
