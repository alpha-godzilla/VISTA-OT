#!/usr/bin/env bash
# Locate which new alignment component changes OT behaviour, one at a time.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-1994}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "$GPU_IDS"
MODEL="${MODEL:-llava-1.5}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
LOGITS_ALPHA="${LOGITS_ALPHA:-0.35}"
MARGINAL_RELAXATION="${MARGINAL_RELAXATION:-0.7}"
OT_TOPK="${OT_TOPK:-16}"
OT_EPSILON="${OT_EPSILON:-0.05}"
OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-100}"
OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"
OT_LAYER_TEMPERATURE="${OT_LAYER_TEMPERATURE:-0.06}"
OT_ATTENTION_POWER="${OT_ATTENTION_POWER:-0.75}"
OT_UNIFORM_MIX="${OT_UNIFORM_MIX:-0.02}"
SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_alignment_localization_seed1994}"
VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-chair_ot_alignment_localization_vista}"
OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_ot_alignment_localization_otattn}"

export VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
export NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"
if [[ -z "${HF_HOME:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" && -d /data/sun_yuxi/huggingface ]]; then export HF_HOME=/data/sun_yuxi/huggingface; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then
  for candidate in /data/sun_yuxi/models/llava-v1.5-7b /data/sun_yuxi/models/llava-1.5-7b-hf /home/ljc/code/models/llava-v1.5-7b; do
    [[ -f "$candidate/config.json" ]] && { export VISTA_LLAVA_MODEL_PATH="$candidate"; break; }
  done
fi
[[ ${#GPUS[@]} -gt 0 ]] || { echo "GPU_IDS must be non-empty." >&2; exit 1; }
[[ -d "$VISTA_COCO_ROOT/val2014" && -f "${VISTA_LLAVA_MODEL_PATH:-}/config.json" ]] || { echo "Set VISTA_COCO_ROOT and VISTA_LLAVA_MODEL_PATH before running." >&2; exit 1; }

mkdir -p "$SWEEP_DIR/manifests" "$SWEEP_DIR/logs"
IDS_FILE="$SWEEP_DIR/manifests/seed_${SEED}_ids.txt"
if [[ ! -f "$IDS_FILE" || "$(wc -l < "$IDS_FILE")" -ne "$SUBSET_SIZE" ]]; then
  "$PYTHON_BIN" scripts/make_chair_seed_manifest.py --data-path "$VISTA_COCO_ROOT/val2014" --seed "$SEED" --subset-size "$SUBSET_SIZE" --output "$IDS_FILE"
fi
base_stem="seed${SEED}_vsv_lambda_${VSV_LAMBDA}_logaug_loglayer_${LOGITS_LAYERS}_logalpha_${LOGITS_ALPHA}"
result_for() {
  local suffix="$1"
  printf '%s/exp_results/%s/%s/%s_otattn_nodust_layerhid_lmhead_tlogit_m%s_kunpooled_it%s_tol%s_eps%s_ltemp%s_apow%s_amix%s_uot_mrel%s_masslayer_dirgate_induni%s_greedy_max_new_tokens_%s.jsonl' \
    "$SCRIPT_DIR" "$OT_EXP_FOLDER" "$MODEL" "$base_stem" "$OT_TOPK" "$OT_SINKHORN_ITERS" "$OT_SINKHORN_TOLERANCE" "$OT_EPSILON" "$OT_LAYER_TEMPERATURE" "$OT_ATTENTION_POWER" "$OT_UNIFORM_MIX" "$MARGINAL_RELAXATION" "$suffix" "$MAX_NEW_TOKENS"
}
vista_result="$SCRIPT_DIR/exp_results/$VISTA_EXP_FOLDER/$MODEL/${base_stem}_greedy_max_new_tokens_${MAX_NEW_TOKENS}.jsonl"
is_complete() { [[ -f "$1" && "$(wc -l < "$1")" -eq "$SUBSET_SIZE" ]]; }

MANIFEST="$SWEEP_DIR/manifest.tsv"
printf 'method\tsetting\tseed\tlogits_alpha\tmarginal_relaxation\tgpu\tids_file\tresult_jsonl\tchair_json\tstats_jsonl\n' > "$MANIFEST"
declare -a JOB_METHODS=() JOB_RESULTS=()
pending=0
enqueue() {
  local method="$1" result="$2" gpu stats rho
  rho="$MARGINAL_RELAXATION"; stats="${result%.jsonl}_ot_stats.jsonl"
  [[ "$method" == vista ]] && { rho=na; stats=na; }
  if is_complete "$result"; then gpu=-1; else gpu="${GPUS[$((pending % ${#GPUS[@]}))]}"; JOB_METHODS+=("$method"); JOB_RESULTS+=("$result"); ((pending += 1)); fi
  printf '%s\talpha%s_rho%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$method" "$LOGITS_ALPHA" "$rho" "$SEED" "$LOGITS_ALPHA" "$rho" "$gpu" "$IDS_FILE" "$result" "${result%.jsonl}_chair.json" "$stats" >> "$MANIFEST"
}
enqueue vista "$vista_result"
enqueue raw_direction_aware "$(result_for '')"
enqueue shared_candidate "$(result_for '_sc')"
enqueue shared_candidate_final_norm "$(result_for '_sc_fn')"
enqueue aligned_centered "$(result_for '_mc_sc_fn')"
enqueue aligned_centered_tgate "$(result_for '_mc_sc_fn_tg')"

run_job() {
  local index="$1" gpu="$2" method="${JOB_METHODS[$index]}" result="${JOB_RESULTS[$index]}" folder="$OT_EXP_FOLDER" backup
  local -a args=(--vsv --vsv-lambda "$VSV_LAMBDA" --logits-aug --logits-layers "$LOGITS_LAYERS" --logits-alpha "$LOGITS_ALPHA")
  if [[ "$method" == vista ]]; then
    folder="$VISTA_EXP_FOLDER"
  else
    args+=(--use-ot-bary-sla --ot-attention-visual-marginal --ot-topk "$OT_TOPK" --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" --ot-sinkhorn-tolerance "$OT_SINKHORN_TOLERANCE" --ot-epsilon "$OT_EPSILON" --ot-layer-temperature "$OT_LAYER_TEMPERATURE" --ot-attention-power "$OT_ATTENTION_POWER" --ot-attention-uniform-mix "$OT_UNIFORM_MIX" --ot-unbalanced --ot-marginal-relaxation "$MARGINAL_RELAXATION" --ot-mass-aware-layer-weights --ot-direction-aware-gating --ot-independent-uniform-layer-weights --ot-log-stats)
    [[ "$method" != raw_direction_aware ]] && args+=(--ot-shared-candidate-set)
    [[ "$method" == shared_candidate_final_norm || "$method" == aligned_centered || "$method" == aligned_centered_tgate ]] && args+=(--ot-final-norm-alignment)
    [[ "$method" == aligned_centered || "$method" == aligned_centered_tgate ]] && args+=(--ot-mass-centered-direction-gating)
    [[ "$method" == aligned_centered_tgate ]] && args+=(--ot-bidirectional-timestep-gate)
  fi
  [[ -f "$result" ]] && { backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"; mv "$result" "$backup"; }
  echo "[GPU $gpu] start method=$method seed=$SEED"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_eval.py --exp_folder "$folder" --model "$MODEL" --data-path "$VISTA_COCO_ROOT/val2014" --subset-size "$SUBSET_SIZE" --subset-ids-file "$IDS_FILE" --seed "$SEED" --max-new-tokens "$MAX_NEW_TOKENS" "${args[@]}" > "$SWEEP_DIR/logs/${method}.log" 2>&1
  is_complete "$result" || { echo "Incomplete result: $result" >&2; return 1; }
}
run_worker() { local worker="$1" index; for ((index=worker; index<${#JOB_METHODS[@]}; index+=${#GPUS[@]})); do run_job "$index" "${GPUS[$worker]}"; done; }
echo "OT alignment localization: ${#JOB_METHODS[@]} pending jobs on GPUs ${GPUS[*]}"
failed=0
if (( ${#JOB_METHODS[@]} > 0 )); then
  declare -a PIDS=()
  for ((worker=0; worker<${#GPUS[@]} && worker<${#JOB_METHODS[@]}; worker+=1)); do run_worker "$worker" & PIDS+=("$!"); done
  for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
fi
(( failed == 0 )) || { echo "Generation failed; see $SWEEP_DIR/logs" >&2; exit 1; }
while IFS=$'\t' read -r method setting seed alpha rho gpu ids result chair_json stats_jsonl; do
  [[ "$method" == method ]] && continue
  if [[ ! -f "$chair_json" || "$chair_json" -ot "$result" ]]; then "$PYTHON_BIN" chair_ans.py --cap_file "$result" --coco_path "$VISTA_COCO_ROOT/annotations" --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$chair_json" > "$SWEEP_DIR/logs/chair_${method}.log" 2>&1; fi
done < "$MANIFEST"
"$PYTHON_BIN" scripts/summarize_ot_alignment_localization.py --manifest "$MANIFEST" --summary-csv "$SWEEP_DIR/summary.csv" --markdown "$SWEEP_DIR/summary.md"
echo "OT alignment localization complete: $SWEEP_DIR/summary.md"
