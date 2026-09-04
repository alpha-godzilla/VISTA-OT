#!/usr/bin/env bash
# 24 new head-aware UOT jobs, reusing completed raw-UOT/VISTA seed-1994 controls.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-1994}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
WORKPOINTS=( ${WORKPOINTS:-0.25:0.5 0.30:0.4 0.35:0.6 0.40:0.6} )
read -r -a MASS_TEMPERATURES <<< "${MASS_TEMPERATURES:-0.5 1.0}"
read -r -a UOT_TEMPERATURES <<< "${UOT_TEMPERATURES:-0.05 0.10}"
read -r -a UOT_UNIFORM_MIXES <<< "${UOT_UNIFORM_MIXES:-0.05 0.15}"
MODEL="${MODEL:-llava-1.5}"; SUBSET_SIZE="${SUBSET_SIZE:-500}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VSV_LAMBDA="${VSV_LAMBDA:-0.17}"; LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
OT_TOPK="${OT_TOPK:-16}"; OT_EPSILON="${OT_EPSILON:-0.05}"; OT_SINKHORN_ITERS="${OT_SINKHORN_ITERS:-100}"; OT_SINKHORN_TOLERANCE="${OT_SINKHORN_TOLERANCE:-0.001}"; OT_LAYER_TEMPERATURE="${OT_LAYER_TEMPERATURE:-0.06}"; OT_ATTENTION_POWER="${OT_ATTENTION_POWER:-0.75}"; OT_UNIFORM_MIX="${OT_UNIFORM_MIX:-0.02}"; HEAD_TOPK="${HEAD_TOPK:-4}"; HEAD_MASS_WEIGHT="${HEAD_MASS_WEIGHT:-0.1}"
REFERENCE_SWEEP_DIR="${REFERENCE_SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_raw_direction_aware_uot_overnight_seed1994}"
REFERENCE_VISTA_FOLDER="${REFERENCE_VISTA_FOLDER:-chair_raw_direction_aware_uot_overnight_vista}"
REFERENCE_OT_FOLDER="${REFERENCE_OT_FOLDER:-chair_raw_direction_aware_uot_overnight_otattn}"
SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_head_aware_uot_seed1994}"
OT_EXP_FOLDER="${OT_EXP_FOLDER:-chair_head_aware_uot_otattn}"
export VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"; export NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"; export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"; export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ -z "${HF_HOME:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" && -d /data/sun_yuxi/huggingface ]]; then export HF_HOME=/data/sun_yuxi/huggingface; fi
if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then for candidate in /data/sun_yuxi/models/llava-v1.5-7b /data/sun_yuxi/models/llava-1.5-7b-hf /home/ljc/code/models/llava-v1.5-7b; do [[ -f "$candidate/config.json" ]] && { export VISTA_LLAVA_MODEL_PATH="$candidate"; break; }; done; fi
[[ ${#GPUS[@]} -gt 0 && -d "$VISTA_COCO_ROOT/val2014" && -f "${VISTA_LLAVA_MODEL_PATH:-}/config.json" ]] || { echo "Set GPU_IDS, VISTA_COCO_ROOT, and VISTA_LLAVA_MODEL_PATH." >&2; exit 1; }
IDS_FILE="$REFERENCE_SWEEP_DIR/manifests/seed_${SEED}_ids.txt"
[[ -f "$IDS_FILE" && "$(wc -l < "$IDS_FILE")" -eq "$SUBSET_SIZE" ]] || { echo "Missing fixed reference manifest: $IDS_FILE" >&2; exit 1; }
mkdir -p "$SWEEP_DIR/logs"
canonical() { "$PYTHON_BIN" -c 'import sys; print(float(sys.argv[1]))' "$1"; }
base_stem() { printf 'seed%s_vsv_lambda_%s_logaug_loglayer_%s_logalpha_%s' "$SEED" "$VSV_LAMBDA" "$LOGITS_LAYERS" "$1"; }
raw_result() { printf '%s/exp_results/%s/%s/%s_otattn_nodust_layerhid_lmhead_tlogit_m%s_kunpooled_it%s_tol%s_eps%s_ltemp%s_apow%s_amix%s_uot_mrel%s_masslayer_dirgate_induni_greedy_max_new_tokens_%s.jsonl' "$SCRIPT_DIR" "$REFERENCE_OT_FOLDER" "$MODEL" "$(base_stem "$1")" "$OT_TOPK" "$OT_SINKHORN_ITERS" "$OT_SINKHORN_TOLERANCE" "$OT_EPSILON" "$OT_LAYER_TEMPERATURE" "$OT_ATTENTION_POWER" "$OT_UNIFORM_MIX" "$2" "$MAX_NEW_TOKENS"; }
vista_result() { printf '%s/exp_results/%s/%s/%s_greedy_max_new_tokens_%s.jsonl' "$SCRIPT_DIR" "$REFERENCE_VISTA_FOLDER" "$MODEL" "$(base_stem "$1")" "$MAX_NEW_TOKENS"; }
head_result() { local alpha="$1" rho="$2" tag="$3"; printf '%s/exp_results/%s/%s/%s_otattn_nodust_layerhid_lmhead_tlogit_m%s_kunpooled_it%s_tol%s_eps%s_ltemp%s_apow%s_amix%s_uot_mrel%s_masslayer_dirgate_induni_%s_greedy_max_new_tokens_%s.jsonl' "$SCRIPT_DIR" "$OT_EXP_FOLDER" "$MODEL" "$(base_stem "$alpha")" "$OT_TOPK" "$OT_SINKHORN_ITERS" "$OT_SINKHORN_TOLERANCE" "$OT_EPSILON" "$OT_LAYER_TEMPERATURE" "$OT_ATTENTION_POWER" "$OT_UNIFORM_MIX" "$rho" "$tag" "$MAX_NEW_TOKENS"; }
is_complete() { [[ -f "$1" && "$(wc -l < "$1")" -eq "$SUBSET_SIZE" ]]; }

# Older sweeps used different exp_folder names (and, in a few cases, a
# different Sinkhorn-iteration tag).  Controls are still valid as long as
# their generation configuration encoded in the filename has the same alpha,
# rho, layer range and visual-marginal/direction-aware mode.  Prefer the
# canonical path above; otherwise discover one complete matching result.
resolve_existing_result() {
  local kind="$1" alpha="$2" rho="${3:-}" exact pattern candidate
  if [[ "$kind" == raw ]]; then
    exact="$(raw_result "$alpha" "$rho")"
    pattern="*/${MODEL}/$(base_stem "$alpha")_otattn_nodust_layerhid_lmhead_tlogit_m*_kunpooled_it*_tol*_eps*_ltemp*_apow*_amix*_uot_mrel${rho}_masslayer_dirgate_induni_greedy_max_new_tokens_${MAX_NEW_TOKENS}.jsonl"
  else
    exact="$(vista_result "$alpha")"
    pattern="*/${MODEL}/$(base_stem "$alpha")_greedy_max_new_tokens_${MAX_NEW_TOKENS}.jsonl"
  fi
  if is_complete "$exact"; then printf '%s\n' "$exact"; return 0; fi
  while IFS= read -r candidate; do
    is_complete "$candidate" && { printf '%s\n' "$candidate"; return 0; }
  done < <(find "$SCRIPT_DIR/exp_results" -type f -path "$pattern" -print | LC_ALL=C sort)
  return 1
}
MANIFEST="$SWEEP_DIR/manifest.tsv"; printf 'method\tsetting\tseed\tlogits_alpha\tmarginal_relaxation\thead_temperature\thead_uniform_mix\thead_topk\thead_mass_weight\tgpu\tids_file\tresult_jsonl\tchair_json\tstats_jsonl\n' > "$MANIFEST"
declare -a JOB_METHODS=() JOB_SETTINGS=() JOB_ALPHAS=() JOB_RHOS=() JOB_MODES=() JOB_TEMPS=() JOB_MIXES=() JOB_RESULTS=(); pending=0
add_row() { local method="$1" setting="$2" alpha="$3" rho="$4" mode="$5" temp="$6" mix="$7" result="$8" gpu stats; stats="${result%.jsonl}_ot_stats.jsonl"; [[ "$method" == vista ]] && stats=na; if is_complete "$result"; then gpu=-1; else gpu="${GPUS[$((pending % ${#GPUS[@]}))]}"; JOB_METHODS+=("$method"); JOB_SETTINGS+=("$setting"); JOB_ALPHAS+=("$alpha"); JOB_RHOS+=("$rho"); JOB_MODES+=("$mode"); JOB_TEMPS+=("$temp"); JOB_MIXES+=("$mix"); JOB_RESULTS+=("$result"); ((pending+=1)); fi; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$method" "$setting" "$SEED" "$alpha" "$rho" "$temp" "$mix" "$HEAD_TOPK" "$HEAD_MASS_WEIGHT" "$gpu" "$IDS_FILE" "$result" "${result%.jsonl}_chair.json" "$stats" >> "$MANIFEST"; }
for point in "${WORKPOINTS[@]}"; do
  IFS=: read -r raw_alpha raw_rho <<< "$point"
  alpha="$(canonical "$raw_alpha")"; rho="$(canonical "$raw_rho")"
  raw_path="$(resolve_existing_result raw "$alpha" "$rho")" || { echo "Missing raw-UOT reference alpha=$alpha rho=$rho (searched exp_results recursively)" >&2; exit 1; }
  echo "Using raw-UOT control alpha=$alpha rho=$rho: $raw_path"
  if vista_path="$(resolve_existing_result vista "$alpha")"; then
    echo "Using VISTA reference alpha=$alpha: $vista_path"
    add_row vista "alpha${alpha}" "$alpha" na none na na "$vista_path"
  fi
  add_row raw_direction_aware_uot "alpha${alpha}_rho${rho}" "$alpha" "$rho" none na na "$raw_path"
  for raw_temp in "${MASS_TEMPERATURES[@]}"; do temp="$(canonical "$raw_temp")"; add_row head_mass "alpha${alpha}_rho${rho}_t${temp}" "$alpha" "$rho" mass "$temp" 0 "$(head_result "$alpha" "$rho" "hM_t${temp}")"; done
  for raw_temp in "${UOT_TEMPERATURES[@]}"; do temp="$(canonical "$raw_temp")"; add_row head_uot "alpha${alpha}_rho${rho}_t${temp}" "$alpha" "$rho" uot "$temp" 0 "$(head_result "$alpha" "$rho" "hO_t${temp}")"; done
  for raw_mix in "${UOT_UNIFORM_MIXES[@]}"; do mix="$(canonical "$raw_mix")"; add_row head_uot_uniform "alpha${alpha}_rho${rho}_u${mix}" "$alpha" "$rho" uot_uniform 0.1 "$mix" "$(head_result "$alpha" "$rho" "hU_u${mix}")"; done
done
run_job() { local index="$1" gpu="$2" method="${JOB_METHODS[$index]}" setting="${JOB_SETTINGS[$index]}" alpha="${JOB_ALPHAS[$index]}" rho="${JOB_RHOS[$index]}" mode="${JOB_MODES[$index]}" temp="${JOB_TEMPS[$index]}" mix="${JOB_MIXES[$index]}" result="${JOB_RESULTS[$index]}" backup; local -a args=(--vsv --vsv-lambda "$VSV_LAMBDA" --logits-aug --logits-layers "$LOGITS_LAYERS" --logits-alpha "$alpha" --use-ot-bary-sla --ot-attention-visual-marginal --ot-topk "$OT_TOPK" --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" --ot-sinkhorn-tolerance "$OT_SINKHORN_TOLERANCE" --ot-epsilon "$OT_EPSILON" --ot-layer-temperature "$OT_LAYER_TEMPERATURE" --ot-attention-power "$OT_ATTENTION_POWER" --ot-attention-uniform-mix "$OT_UNIFORM_MIX" --ot-unbalanced --ot-marginal-relaxation "$rho" --ot-mass-aware-layer-weights --ot-direction-aware-gating --ot-independent-uniform-layer-weights --ot-head-aware-mode "$mode" --ot-head-topk "$HEAD_TOPK" --ot-head-temperature "$temp" --ot-head-uniform-mix "$mix" --ot-head-mass-weight "$HEAD_MASS_WEIGHT" --ot-log-stats); [[ -f "$result" ]] && { backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"; mv "$result" "$backup"; }; echo "[GPU $gpu] start method=$method setting=$setting"; CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_eval.py --exp_folder "$OT_EXP_FOLDER" --model "$MODEL" --data-path "$VISTA_COCO_ROOT/val2014" --subset-size "$SUBSET_SIZE" --subset-ids-file "$IDS_FILE" --seed "$SEED" --max-new-tokens "$MAX_NEW_TOKENS" "${args[@]}" > "$SWEEP_DIR/logs/${method}_${setting}.log" 2>&1; is_complete "$result" || { echo "Incomplete result: $result" >&2; return 1; }; }
run_worker() { local worker="$1" index; for ((index=worker; index<${#JOB_METHODS[@]}; index+=${#GPUS[@]})); do run_job "$index" "${GPUS[$worker]}"; done; }
echo "Head-aware UOT: ${#JOB_METHODS[@]} pending jobs (24 new expected) on GPUs ${GPUS[*]}"; failed=0; if (( ${#JOB_METHODS[@]} > 0 )); then declare -a PIDS=(); for ((worker=0; worker<${#GPUS[@]} && worker<${#JOB_METHODS[@]}; worker+=1)); do run_worker "$worker" & PIDS+=("$!"); done; for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done; fi; (( failed == 0 )) || { echo "Generation failed; see $SWEEP_DIR/logs" >&2; exit 1; }
while IFS=$'\t' read -r method setting seed alpha rho temp mix topk mass gpu ids result chair_json stats_jsonl; do [[ "$method" == method ]] && continue; if [[ ! -f "$chair_json" || "$chair_json" -ot "$result" ]]; then "$PYTHON_BIN" chair_ans.py --cap_file "$result" --coco_path "$VISTA_COCO_ROOT/annotations" --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$chair_json" > "$SWEEP_DIR/logs/chair_${method}_${setting}.log" 2>&1; fi; done < "$MANIFEST"
"$PYTHON_BIN" scripts/summarize_head_aware_uot_grid.py --manifest "$MANIFEST" --summary-csv "$SWEEP_DIR/summary.csv" --markdown "$SWEEP_DIR/summary.md"; echo "Head-aware UOT complete: $SWEEP_DIR/summary.md"
