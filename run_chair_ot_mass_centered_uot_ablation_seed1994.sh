#!/usr/bin/env bash
# Seed-1994 development sweep for mass-centered direction-aware UOT.
# Generation never reads CHAIR labels; CHAIR is used only after all jobs finish.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-1994}"
read -r -a CENTERED_ALPHAS <<< "${CENTERED_ALPHAS:-0.35 0.5 0.7 1.0}"
read -r -a MARGINAL_RELAXATIONS <<< "${MARGINAL_RELAXATIONS:-0.5 0.6 0.7 0.8 0.9 1.0}"
read -r -a BASELINE_ALPHAS <<< "${BASELINE_ALPHAS:-0.15 0.25 0.35}"
RAW_REFERENCE_ALPHA="${RAW_REFERENCE_ALPHA:-0.35}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"

MODEL="${MODEL:-llava-1.5}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
OT_TOPK="${OT_TOPK:-16}"
OT_EPSILON="${OT_EPSILON:-0.05}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-100}"
OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"
OT_LAYER_TEMPERATURE="${OT_LAYER_TEMPERATURE:-0.06}"
OT_ATTENTION_POWER="${OT_ATTENTION_POWER:-0.75}"
OT_UNIFORM_MIX="${OT_UNIFORM_MIX:-0.02}"

SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_mass_centered_uot_seed1994}"
VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-chair_ot_mass_centered_uot_vista}"
OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_ot_mass_centered_uot_otattn}"

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
if (( ${#CENTERED_ALPHAS[@]} == 0 || ${#MARGINAL_RELAXATIONS[@]} == 0 || ${#BASELINE_ALPHAS[@]} == 0 || ${#GPUS[@]} == 0 )); then
  echo "CENTERED_ALPHAS, MARGINAL_RELAXATIONS, BASELINE_ALPHAS, and GPU_IDS must be non-empty." >&2
  exit 1
fi
if [[ ! -d "$VISTA_COCO_ROOT/val2014" || ! -f "${VISTA_LLAVA_MODEL_PATH:-}/config.json" ]]; then
  echo "Set VISTA_COCO_ROOT and VISTA_LLAVA_MODEL_PATH before running." >&2
  exit 1
fi

mkdir -p "$SWEEP_DIR/manifests" "$SWEEP_DIR/logs"
IDS_FILE="$SWEEP_DIR/manifests/seed_${SEED}_ids.txt"
if [[ ! -f "$IDS_FILE" || "$(wc -l < "$IDS_FILE")" -ne "$SUBSET_SIZE" ]]; then
  "$PYTHON_BIN" scripts/make_chair_seed_manifest.py \
    --data-path "$VISTA_COCO_ROOT/val2014" --seed "$SEED" \
    --subset-size "$SUBSET_SIZE" --output "$IDS_FILE"
fi

canonical_float() { "$PYTHON_BIN" -c 'import sys; print(float(sys.argv[1]))' "$1"; }
for raw in "${CENTERED_ALPHAS[@]}" "${BASELINE_ALPHAS[@]}" "$RAW_REFERENCE_ALPHA"; do
  "$PYTHON_BIN" -c 'import sys; assert 0 <= float(sys.argv[1]) <= 1' "$raw" || {
    echo "All alpha values must be in [0, 1]." >&2; exit 1;
  }
done
for raw in "${MARGINAL_RELAXATIONS[@]}"; do
  "$PYTHON_BIN" -c 'import sys; assert float(sys.argv[1]) > 0' "$raw" || {
    echo "All MARGINAL_RELAXATIONS must be positive." >&2; exit 1;
  }
done

base_stem() {
  printf 'seed%s_vsv_lambda_%s_logaug_loglayer_%s_logalpha_%s' \
    "$SEED" "$VSV_LAMBDA" "$LOGITS_LAYERS" "$1"
}
vista_result() {
  printf '%s/exp_results/%s/%s/%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$VISTA_EXP_FOLDER" "$MODEL" "$(base_stem "$1")" "$MAX_NEW_TOKENS"
}
ot_result() {
  local alpha="$1" suffix="$2"
  printf '%s/exp_results/%s/%s/%s_otattn_nodust_layerhid_lmhead_tlogit_m%s_kunpooled_it%s_tol%s_eps%s_ltemp%s_apow%s_amix%s%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$OT_EXP_FOLDER" "$MODEL" "$(base_stem "$alpha")" \
    "$OT_TOPK" "$OT_SINKHORN_ITERS" "$OT_SINKHORN_TOLERANCE" \
    "$OT_EPSILON" "$OT_LAYER_TEMPERATURE" "$OT_ATTENTION_POWER" \
    "$OT_UNIFORM_MIX" "$suffix" "$MAX_NEW_TOKENS"
}
is_complete() { [[ -f "$1" && "$(wc -l < "$1")" -eq "$SUBSET_SIZE" ]]; }

MANIFEST="$SWEEP_DIR/manifest.tsv"
printf 'method\tsetting\tseed\tlogits_alpha\tmarginal_relaxation\tgate_mode\tgpu\tids_file\tresult_jsonl\tchair_json\tstats_jsonl\n' > "$MANIFEST"
declare -a JOB_METHODS=() JOB_SETTINGS=() JOB_ALPHAS=() JOB_RHOS=() JOB_MODES=() JOB_RESULTS=()
pending=0
enqueue() {
  local method="$1" setting="$2" alpha="$3" rho="$4" mode="$5" result="$6"
  local gpu stats="${result%.jsonl}_ot_stats.jsonl"
  if is_complete "$result"; then
    gpu=-1
  else
    gpu="${GPUS[$((pending % ${#GPUS[@]}))]}"
    JOB_METHODS+=("$method"); JOB_SETTINGS+=("$setting")
    JOB_ALPHAS+=("$alpha"); JOB_RHOS+=("$rho")
    JOB_MODES+=("$mode"); JOB_RESULTS+=("$result")
    ((pending += 1))
  fi
  [[ "$method" == vista ]] && stats=na
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$method" "$setting" "$SEED" "$alpha" "$rho" "$mode" "$gpu" \
    "$IDS_FILE" "$result" "${result%.jsonl}_chair.json" "$stats" >> "$MANIFEST"
}

for raw_alpha in "${BASELINE_ALPHAS[@]}"; do
  alpha="$(canonical_float "$raw_alpha")"
  enqueue vista "alpha${alpha}" "$alpha" na baseline "$(vista_result "$alpha")"
  enqueue balanced_ot "alpha${alpha}" "$alpha" na balanced "$(ot_result "$alpha" '')"
done
raw_alpha="$(canonical_float "$RAW_REFERENCE_ALPHA")"
for raw_rho in "${MARGINAL_RELAXATIONS[@]}"; do
  rho="$(canonical_float "$raw_rho")"
  enqueue direction_aware_uot_raw "alpha${raw_alpha}_rho${rho}" "$raw_alpha" "$rho" raw \
    "$(ot_result "$raw_alpha" "_uot_mrel${rho}_masslayer_dirgate_induni")"
done
for raw_alpha in "${CENTERED_ALPHAS[@]}"; do
  alpha="$(canonical_float "$raw_alpha")"
  for raw_rho in "${MARGINAL_RELAXATIONS[@]}"; do
    rho="$(canonical_float "$raw_rho")"
    enqueue direction_aware_uot_centered "alpha${alpha}_rho${rho}" "$alpha" "$rho" centered \
      "$(ot_result "$alpha" "_uot_mrel${rho}_masslayer_dirgate_induni_masscenter")"
  done
done

run_job() {
  local index="$1" gpu="$2"
  local method="${JOB_METHODS[$index]}" setting="${JOB_SETTINGS[$index]}"
  local alpha="${JOB_ALPHAS[$index]}" rho="${JOB_RHOS[$index]}"
  local mode="${JOB_MODES[$index]}" result="${JOB_RESULTS[$index]}"
  local exp_folder="$OT_EXP_FOLDER" backup log_file
  local -a args=(
    --vsv --vsv-lambda "$VSV_LAMBDA" --logits-aug
    --logits-layers "$LOGITS_LAYERS" --logits-alpha "$alpha"
  )
  if [[ "$method" == vista ]]; then
    exp_folder="$VISTA_EXP_FOLDER"
  else
    args+=(
      --use-ot-bary-sla --ot-attention-visual-marginal
      --ot-topk "$OT_TOPK" --ot-sinkhorn-iters "$OT_SINKHORN_ITERS"
      --ot-sinkhorn-tolerance "$OT_SINKHORN_TOLERANCE"
      --ot-epsilon "$OT_EPSILON" --ot-layer-temperature "$OT_LAYER_TEMPERATURE"
      --ot-attention-power "$OT_ATTENTION_POWER"
      --ot-attention-uniform-mix "$OT_UNIFORM_MIX" --ot-log-stats
    )
    if [[ "$mode" == raw || "$mode" == centered ]]; then
      args+=(
        --ot-unbalanced --ot-marginal-relaxation "$rho"
        --ot-mass-aware-layer-weights --ot-direction-aware-gating
        --ot-independent-uniform-layer-weights
      )
      [[ "$mode" == centered ]] && args+=(--ot-mass-centered-direction-gating)
    fi
  fi
  if [[ -f "$result" ]]; then
    backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"
    mv "$result" "$backup"
  fi
  log_file="$SWEEP_DIR/logs/${method}_${setting}.log"
  echo "[GPU $gpu] start method=$method setting=$setting seed=$SEED"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_eval.py \
    --exp_folder "$exp_folder" --model "$MODEL" \
    --data-path "$VISTA_COCO_ROOT/val2014" --subset-size "$SUBSET_SIZE" \
    --subset-ids-file "$IDS_FILE" --seed "$SEED" \
    --max-new-tokens "$MAX_NEW_TOKENS" "${args[@]}" > "$log_file" 2>&1
  is_complete "$result" || { echo "Incomplete result: $result" >&2; return 1; }
}
run_worker() {
  local worker="$1" gpu="${GPUS[$worker]}" index
  for ((index=worker; index<${#JOB_METHODS[@]}; index+=${#GPUS[@]})); do
    run_job "$index" "$gpu"
  done
}

echo "Mass-centered UOT ablation: ${#JOB_METHODS[@]} pending jobs on GPUs ${GPUS[*]}"
failed=0
if (( ${#JOB_METHODS[@]} > 0 )); then
  declare -a PIDS=()
  for ((worker=0; worker<${#GPUS[@]} && worker<${#JOB_METHODS[@]}; worker+=1)); do
    run_worker "$worker" & PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
fi
(( failed == 0 )) || { echo "Generation failed; see $SWEEP_DIR/logs" >&2; exit 1; }

while IFS=$'\t' read -r method setting seed alpha rho mode gpu ids result chair_json stats_jsonl; do
  [[ "$method" == method ]] && continue
  if [[ ! -f "$chair_json" || "$chair_json" -ot "$result" ]]; then
    "$PYTHON_BIN" chair_ans.py --cap_file "$result" \
      --coco_path "$VISTA_COCO_ROOT/annotations" \
      --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$chair_json" \
      > "$SWEEP_DIR/logs/chair_${method}_${setting}.log" 2>&1
  fi
done < "$MANIFEST"

"$PYTHON_BIN" scripts/summarize_mass_centered_uot_ablation.py \
  --manifest "$MANIFEST" --summary-csv "$SWEEP_DIR/summary.csv" \
  --markdown "$SWEEP_DIR/summary.md"
echo "Mass-centered UOT ablation complete: $SWEEP_DIR/summary.md"
