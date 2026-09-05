#!/usr/bin/env bash
# Phase A+B: contribution single-UOT, exact-uniform experts, strict VHD oracle.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"; SEED="${SEED:-1994}"; MODEL="${MODEL:-llava-1.5}"; COCO="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"; SWEEP="${SWEEP_DIR:-$ROOT/exp_results/chair_uot_phase_ab_seed1994}"; EXP="${OT_EXP_FOLDER:-chair_uot_phase_ab}"
export VISTA_COCO_ROOT="$COCO" NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
[[ -n "${HF_HOME:-}" || -n "${HUGGINGFACE_HUB_CACHE:-}" || ! -d /data/sun_yuxi/huggingface ]] || export HF_HOME=/data/sun_yuxi/huggingface
if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then for p in /data/sun_yuxi/models/llava-v1.5-7b /data/sun_yuxi/models/llava-1.5-7b-hf; do [[ -f "$p/config.json" ]] && { export VISTA_LLAVA_MODEL_PATH="$p"; break; }; done; fi
mkdir -p "$SWEEP/logs" "$SWEEP/manifests"; IDS="$SWEEP/manifests/seed_${SEED}_ids.txt"
[[ -f "$IDS" && "$(wc -l < "$IDS")" -eq 500 ]] || "$PYTHON_BIN" scripts/make_chair_seed_manifest.py --data-path "$COCO/val2014" --seed "$SEED" --subset-size 500 --output "$IDS"
complete() { [[ -f "$1" && "$(wc -l < "$1")" -eq 500 ]]; }
stem() { printf 'seed%s_vsv_lambda_0.17_logaug_loglayer_25,30_logalpha_%s_otattn_nodust_layerhid_lmhead_tlogit_m16_kunpooled_it100_tol0.001_eps0.05_ltemp0.06_apow0.75_amix0.02_uot_mrel%s_masslayer_dirgate_induni_%s_greedy_max_new_tokens_512.jsonl' "$SEED" "$1" "$2" "$3"; }
out() { printf '%s/exp_results/%s/%s/%s' "$ROOT" "$EXP" "$MODEL" "$(stem "$1" "$2" "$3")"; }
declare -a A=(0.3 0.35 0.4 0.3 0.4) R=(0.4 0.6 0.6 0.4 0.6) MODE=(contribution contribution contribution uot_equal uot_equal) K=(16 16 16 4 4) TAG=(hC_k16 hC_k16 hC_k16 hE_k4 hE_k4) OUT=()
for i in "${!A[@]}"; do OUT+=("$(out "${A[$i]}" "${R[$i]}" "${TAG[$i]}")"); done
run() { local i="$1" gpu="$2" file="${OUT[$i]}"; if [[ -f "$file" ]] && ! complete "$file"; then mv "$file" "$SWEEP/logs/partial_${i}_$(date +%s).jsonl"; fi; echo "[GPU $gpu] ${MODE[$i]} alpha=${A[$i]} rho=${R[$i]} k=${K[$i]}"; CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_eval.py --exp_folder "$EXP" --model "$MODEL" --data-path "$COCO/val2014" --subset-size 500 --subset-ids-file "$IDS" --seed "$SEED" --max-new-tokens 512 --vsv --vsv-lambda 0.17 --logits-aug --logits-layers 25,30 --logits-alpha "${A[$i]}" --use-ot-bary-sla --ot-attention-visual-marginal --ot-topk 16 --ot-sinkhorn-iters 100 --ot-sinkhorn-tolerance 0.001 --ot-epsilon 0.05 --ot-layer-temperature 0.06 --ot-attention-power 0.75 --ot-attention-uniform-mix 0.02 --ot-unbalanced --ot-marginal-relaxation "${R[$i]}" --ot-mass-aware-layer-weights --ot-direction-aware-gating --ot-independent-uniform-layer-weights --ot-head-aware-mode "${MODE[$i]}" --ot-head-topk "${K[$i]}" --ot-log-stats > "$SWEEP/logs/job_${i}.log" 2>&1; complete "$file"; }
declare -a PIDS=(); for i in "${!OUT[@]}"; do complete "${OUT[$i]}" || { run "$i" "${GPUS[$((i % ${#GPUS[@]}))]}" & PIDS+=("$!"); }; done; failed=0; if (( ${#PIDS[@]} > 0 )); then for p in "${PIDS[@]}"; do wait "$p" || failed=1; done; fi; (( failed == 0 )) || { echo "Generation failed; see $SWEEP/logs" >&2; exit 1; }
for i in "${!OUT[@]}"; do chair="${OUT[$i]%.jsonl}_chair.json"; [[ -f "$chair" && "$chair" -nt "${OUT[$i]}" ]] || "$PYTHON_BIN" chair_ans.py --cap_file "${OUT[$i]}" --coco_path "$COCO/annotations" --cache "$COCO/chair.pkl" --save_path "$chair" > "$SWEEP/logs/chair_${i}.log" 2>&1; done
PIDS=(); for i in 0 1 2; do CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" scripts/vhd_teacher_forced_oracle.py --captions "${OUT[$i]}" --data-path "$COCO/val2014" --output "$SWEEP/vhd_alpha${A[$i]}_rho${R[$i]}.jsonl" --limit 100 --layers 25,30 > "$SWEEP/logs/vhd_${i}.log" 2>&1 & PIDS+=("$!"); done; failed=0; for p in "${PIDS[@]}"; do wait "$p" || failed=1; done; (( failed == 0 )) || { echo "VHD oracle failed; see $SWEEP/logs" >&2; exit 1; }
echo "Phase A+B complete: $SWEEP"
