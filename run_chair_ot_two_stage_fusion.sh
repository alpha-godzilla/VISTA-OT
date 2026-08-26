#!/usr/bin/env bash
# VISTA proposal -> second-pass OT verification/fusion on 8 GPUs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a SEEDS <<< "${SEEDS:-1994 2024 3407}"
read -r -a MODES <<< "${FUSION_MODES:-ot_self vista_only dual_fusion}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"

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
ADAPTIVE_MIN_RATIO="${ADAPTIVE_MIN_RATIO:-0.2}"
RECOVERY_RHO="${RECOVERY_RHO:-0.25}"
RECALL_CANDIDATE_TOPK="${RECALL_CANDIDATE_TOPK:-32}"
RECALL_TEMPERATURE="${RECALL_TEMPERATURE:-0.1}"
RECALL_COVERAGE_DECAY="${RECALL_COVERAGE_DECAY:-1.0}"
RECOVERY_RHO="$($PYTHON_BIN -c 'import sys; print(float(sys.argv[1]))' "$RECOVERY_RHO")"
if [[ ! "$RECALL_CANDIDATE_TOPK" =~ ^[1-9][0-9]*$ ]]; then
  echo "RECALL_CANDIDATE_TOPK must be a positive integer." >&2; exit 1
fi

SOURCE_MANIFEST="${SOURCE_MANIFEST:-$SCRIPT_DIR/exp_results/chair_ot_recall_recovery_ablation/manifest.tsv}"
SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/exp_results/chair_ot_two_stage_fusion}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/exp_results/chair_ot_two_stage_fusion_outputs/$MODEL}"
FINAL_TAG="a${LOGITS_ALPHA}_amin${ADAPTIVE_MIN_RATIO}_rho${RECOVERY_RHO}_k${RECALL_CANDIDATE_TOPK}_lt${OT_LAYER_TEMPERATURE}_p${OT_ATTENTION_POWER}"

export VISTA_COCO_ROOT="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
export NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"
if [[ -z "${HF_HOME:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" && -d /data/sun_yuxi/huggingface ]]; then
  export HF_HOME=/data/sun_yuxi/huggingface
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then
  for candidate in /data/sun_yuxi/models/llava-v1.5-7b /data/sun_yuxi/models/llava-1.5-7b-hf /home/ljc/code/models/llava-v1.5-7b; do
    if [[ -f "$candidate/config.json" ]]; then export VISTA_LLAVA_MODEL_PATH="$candidate"; break; fi
  done
fi
if [[ ! -f "$SOURCE_MANIFEST" ]]; then
  echo "Missing source manifest: $SOURCE_MANIFEST" >&2
  echo "Finish run_chair_ot_recall_recovery_ablation.sh first." >&2
  exit 1
fi
if [[ ! -d "$VISTA_COCO_ROOT/val2014" || ! -f "${VISTA_LLAVA_MODEL_PATH:-}/config.json" ]]; then
  echo "Set VISTA_COCO_ROOT and VISTA_LLAVA_MODEL_PATH before running." >&2; exit 1
fi
if (( ${#SEEDS[@]} == 0 || ${#MODES[@]} == 0 || ${#GPUS[@]} == 0 )); then
  echo "SEEDS, FUSION_MODES, and GPU_IDS must be non-empty." >&2; exit 1
fi
for mode in "${MODES[@]}"; do
  case "$mode" in ot_self|vista_only|dual_fusion) ;; *) echo "Invalid fusion mode: $mode" >&2; exit 1 ;; esac
done

mkdir -p "$SWEEP_DIR/logs" "$OUTPUT_DIR"
MANIFEST="$SWEEP_DIR/manifest.tsv"
printf 'method\tsetting\tseed\tgpu\tids_file\tresult_jsonl\tchair_json\n' > "$MANIFEST"

lookup_source() {
  local method="$1" setting="$2" seed="$3" column="$4"
  awk -F '\t' -v method="$method" -v setting="$setting" -v seed="$seed" -v column="$column" \
    'NR > 1 && $1 == method && $2 == setting && $3 == seed { print $column; found=1; exit } END { if (!found) exit 1 }' \
    "$SOURCE_MANIFEST"
}
is_complete() { [[ -f "$1" && "$(wc -l < "$1")" -eq "$SUBSET_SIZE" ]]; }

declare -a JOB_MODES=() JOB_SEEDS=() JOB_IDS=() JOB_VISTA=() JOB_OT=() JOB_RESULTS=() JOB_STATS=()
pending=0
enqueue() {
  local mode="$1" seed="$2" ids="$3" vista="$4" ot="$5" result="$6" stats="$7" gpu
  if is_complete "$result"; then gpu=-1; else
    gpu="${GPUS[$((pending % ${#GPUS[@]}))]}"
    JOB_MODES+=("$mode"); JOB_SEEDS+=("$seed"); JOB_IDS+=("$ids"); JOB_VISTA+=("$vista"); JOB_OT+=("$ot"); JOB_RESULTS+=("$result"); JOB_STATS+=("$stats")
    ((pending += 1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" final_ot "$seed" "$gpu" "$ids" "$result" "${result%.jsonl}_chair.json" >> "$MANIFEST"
}

for seed in "${SEEDS[@]}"; do
  ids="$(lookup_source vista original "$seed" 5)"
  vista_result="$(lookup_source vista original "$seed" 6)"
  vista_chair="$(lookup_source vista original "$seed" 7)"
  recovery_setting="rho${RECOVERY_RHO}_k${RECALL_CANDIDATE_TOPK}"
  ot_result="$(lookup_source recall_recovery "$recovery_setting" "$seed" 6)"
  ot_chair="$(lookup_source recall_recovery "$recovery_setting" "$seed" 7)"
  if ! is_complete "$vista_result" || ! is_complete "$ot_result"; then
    echo "Incomplete stage-one proposal pair for seed=$seed" >&2; exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' vista original "$seed" -1 "$ids" "$vista_result" "$vista_chair" >> "$MANIFEST"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' ot_stage1 "$recovery_setting" "$seed" -1 "$ids" "$ot_result" "$ot_chair" >> "$MANIFEST"
  for mode in "${MODES[@]}"; do
    result="$OUTPUT_DIR/seed${seed}_${mode}_${FINAL_TAG}.jsonl"
    stats="$OUTPUT_DIR/seed${seed}_${mode}_${FINAL_TAG}_stats.jsonl"
    enqueue "$mode" "$seed" "$ids" "$vista_result" "$ot_result" "$result" "$stats"
  done
done

run_job() {
  local index="$1"
  local gpu="$2"
  local mode="${JOB_MODES[$index]}"
  local seed="${JOB_SEEDS[$index]}"
  local ids="${JOB_IDS[$index]}"
  local vista="${JOB_VISTA[$index]}"
  local ot="${JOB_OT[$index]}"
  local result="${JOB_RESULTS[$index]}"
  local stats="${JOB_STATS[$index]}"
  local backup log_file
  if [[ -f "$result" ]]; then backup="${result}.partial.$(date +%Y%m%d_%H%M%S)"; mv "$result" "$backup"; fi
  if [[ -f "$stats" ]]; then backup="${stats}.partial.$(date +%Y%m%d_%H%M%S)"; mv "$stats" "$backup"; fi
  log_file="$SWEEP_DIR/logs/${mode}_seed${seed}.log"
  echo "[GPU $gpu] start mode=$mode seed=$seed"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_ot_two_stage.py \
    --model "$MODEL" --data-path "$VISTA_COCO_ROOT/val2014" --subset-ids-file "$ids" \
    --vista-proposals "$vista" --ot-proposals "$ot" --fusion-mode "$mode" \
    --output "$result" --stats-output "$stats" --seed "$seed" --max-new-tokens "$MAX_NEW_TOKENS" \
    --vsv --vsv-lambda "$VSV_LAMBDA" --logits-aug --logits-layers "$LOGITS_LAYERS" --logits-alpha "$LOGITS_ALPHA" \
    --use-ot-bary-sla --ot-attention-visual-marginal --ot-topk "$OT_TOPK" \
    --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" --ot-sinkhorn-tolerance "$OT_SINKHORN_TOLERANCE" \
    --ot-epsilon "$OT_EPSILON" --ot-layer-temperature "$OT_LAYER_TEMPERATURE" \
    --ot-attention-power "$OT_ATTENTION_POWER" --ot-attention-uniform-mix "$OT_UNIFORM_MIX" \
    --ot-adaptive-alpha --ot-adaptive-alpha-min-ratio "$ADAPTIVE_MIN_RATIO" \
    --ot-recall-recovery-rho "$RECOVERY_RHO" --ot-recall-candidate-topk "$RECALL_CANDIDATE_TOPK" \
    --ot-recall-temperature "$RECALL_TEMPERATURE" --ot-recall-coverage-decay "$RECALL_COVERAGE_DECAY" \
    --ot-log-stats > "$log_file" 2>&1
  is_complete "$result" || { echo "Incomplete result: $result" >&2; return 1; }
}
run_worker() {
  local worker="$1"
  local gpu="${GPUS[$worker]}"
  local index
  for ((index=worker; index<${#JOB_MODES[@]}; index+=${#GPUS[@]})); do
    run_job "$index" "$gpu"
  done
}

echo "Two-stage OT fusion: ${#JOB_MODES[@]} pending jobs on GPUs ${GPUS[*]}"
declare -a PIDS=()
for ((worker=0; worker<${#GPUS[@]} && worker<${#JOB_MODES[@]}; worker+=1)); do run_worker "$worker" & PIDS+=("$!"); done
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "Generation failed; see $SWEEP_DIR/logs" >&2; exit 1; }

while IFS=$'\t' read -r method setting seed gpu ids result chair_json; do
  [[ "$method" == method ]] && continue
  if [[ ! -f "$chair_json" || "$chair_json" -ot "$result" ]]; then
    "$PYTHON_BIN" chair_ans.py --cap_file "$result" --coco_path "$VISTA_COCO_ROOT/annotations" --cache "$VISTA_COCO_ROOT/chair.pkl" --save_path "$chair_json" > "$SWEEP_DIR/logs/chair_${method}_seed${seed}.log" 2>&1
  fi
done < "$MANIFEST"

"$PYTHON_BIN" scripts/analyze_chair_proposal_overlap.py --manifest "$MANIFEST" --csv "$SWEEP_DIR/proposal_overlap.csv" --markdown "$SWEEP_DIR/proposal_overlap.md"
"$PYTHON_BIN" scripts/summarize_chair_ot_attention_modules.py --manifest "$MANIFEST" --csv "$SWEEP_DIR/summary.csv" --markdown "$SWEEP_DIR/summary.md"
echo "Two-stage fusion complete: $SWEEP_DIR/summary.md"
echo "Proposal ceiling: $SWEEP_DIR/proposal_overlap.md"
