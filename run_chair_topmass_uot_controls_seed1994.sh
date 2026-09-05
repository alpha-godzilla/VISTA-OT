#!/usr/bin/env bash
# Isolate top-M truncation from per-head UOT: one mass-preserving pool, one UOT.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"; SEED="${SEED:-1994}"; MODEL="${MODEL:-llava-1.5}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
WORKPOINTS=( ${WORKPOINTS:-0.25:0.5 0.30:0.4 0.35:0.6 0.40:0.6} )
TOPKS=( ${HEAD_TOPKS:-4 8 16} ); SUBSET_SIZE="${SUBSET_SIZE:-500}"
COCO="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"; EXP="${OT_EXP_FOLDER:-chair_topmass_uot_controls}"
SWEEP="${SWEEP_DIR:-$ROOT/exp_results/chair_topmass_uot_controls_seed1994}"
export VISTA_COCO_ROOT="$COCO" NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ -z "${HF_HOME:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" && -d /data/sun_yuxi/huggingface ]]; then export HF_HOME=/data/sun_yuxi/huggingface; fi
if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then for p in /data/sun_yuxi/models/llava-v1.5-7b /data/sun_yuxi/models/llava-1.5-7b-hf; do [[ -f "$p/config.json" ]] && { export VISTA_LLAVA_MODEL_PATH="$p"; break; }; done; fi
[[ -d "$COCO/val2014" && -f "${VISTA_LLAVA_MODEL_PATH:-}/config.json" ]] || { echo "Set VISTA_COCO_ROOT and VISTA_LLAVA_MODEL_PATH" >&2; exit 1; }
mkdir -p "$SWEEP/logs" "$SWEEP/manifests"; IDS="$SWEEP/manifests/seed_${SEED}_ids.txt"
[[ -f "$IDS" && "$(wc -l < "$IDS")" -eq "$SUBSET_SIZE" ]] || "$PYTHON_BIN" scripts/make_chair_seed_manifest.py --data-path "$COCO/val2014" --seed "$SEED" --subset-size "$SUBSET_SIZE" --output "$IDS"
canon() { "$PYTHON_BIN" -c 'import sys; print(float(sys.argv[1]))' "$1"; }
result() { printf '%s/exp_results/%s/%s/seed%s_vsv_lambda_0.17_logaug_loglayer_25,30_logalpha_%s_otattn_nodust_layerhid_lmhead_tlogit_m16_kunpooled_it100_tol0.001_eps0.05_ltemp0.06_apow0.75_amix0.02_uot_mrel%s_masslayer_dirgate_induni_hT_k%s_greedy_max_new_tokens_512.jsonl' "$ROOT" "$EXP" "$MODEL" "$SEED" "$1" "$2" "$3"; }
complete() { [[ -f "$1" && "$(wc -l < "$1")" -eq "$SUBSET_SIZE" ]]; }
MAN="$SWEEP/manifest.tsv"; printf 'method\tsetting\tseed\tlogits_alpha\tmarginal_relaxation\thead_topk\tresult_jsonl\tchair_json\n' > "$MAN"
declare -a ALPHA=() RHO=() K=() OUT=(); n=0
for item in "${WORKPOINTS[@]}"; do IFS=: read -r a r <<< "$item"; a="$(canon "$a")"; r="$(canon "$r")"; for k in "${TOPKS[@]}"; do out="$(result "$a" "$r" "$k")"; printf 'topmass\talpha%s_rho%s_k%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$a" "$r" "$k" "$SEED" "$a" "$r" "$k" "$out" "${out%.jsonl}_chair.json" >> "$MAN"; complete "$out" || { ALPHA+=("$a"); RHO+=("$r"); K+=("$k"); OUT+=("$out"); ((n+=1)); }; done; done
run() { local i="$1" gpu="$2" out="${OUT[$i]}" stamp short; if [[ -f "$out" ]]; then stamp="$(date +%Y%m%d_%H%M%S)"; short="a${ALPHA[$i]}_r${RHO[$i]}_k${K[$i]}_${stamp}"; mv "$out" "$SWEEP/logs/partial_${short}.jsonl"; [[ -f "${out%.jsonl}_ot_stats.jsonl" ]] && mv "${out%.jsonl}_ot_stats.jsonl" "$SWEEP/logs/partial_${short}_ot_stats.jsonl"; fi; echo "[GPU $gpu] topmass alpha=${ALPHA[$i]} rho=${RHO[$i]} k=${K[$i]}"; CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_eval.py --exp_folder "$EXP" --model "$MODEL" --data-path "$COCO/val2014" --subset-size "$SUBSET_SIZE" --subset-ids-file "$IDS" --seed "$SEED" --max-new-tokens 512 --vsv --vsv-lambda 0.17 --logits-aug --logits-layers 25,30 --logits-alpha "${ALPHA[$i]}" --use-ot-bary-sla --ot-attention-visual-marginal --ot-topk 16 --ot-sinkhorn-iters 100 --ot-sinkhorn-tolerance 0.001 --ot-epsilon 0.05 --ot-layer-temperature 0.06 --ot-attention-power 0.75 --ot-attention-uniform-mix 0.02 --ot-unbalanced --ot-marginal-relaxation "${RHO[$i]}" --ot-mass-aware-layer-weights --ot-direction-aware-gating --ot-independent-uniform-layer-weights --ot-head-aware-mode topmass --ot-head-topk "${K[$i]}" --ot-log-stats > "$SWEEP/logs/topmass_a${ALPHA[$i]}_r${RHO[$i]}_k${K[$i]}.log" 2>&1; complete "$out"; }
run_worker() { local worker="$1" i; for ((i=worker;i<n;i+=${#GPUS[@]})); do run "$i" "${GPUS[$worker]}"; done; }
echo "Top-M single-UOT control: $n pending jobs on GPUs ${GPUS[*]}"; failed=0; if (( n > 0 )); then declare -a PIDS=(); for ((worker=0;worker<${#GPUS[@]} && worker<n;worker++)); do run_worker "$worker" & PIDS+=("$!"); done; for p in "${PIDS[@]}"; do wait "$p" || failed=1; done; fi; (( failed == 0 )) || { echo "Generation failed; inspect $SWEEP/logs" >&2; exit 1; }
while IFS=$'\t' read -r method setting seed alpha rho topk out chair; do [[ "$method" == method ]] && continue; [[ -f "$chair" && "$chair" -nt "$out" ]] || "$PYTHON_BIN" chair_ans.py --cap_file "$out" --coco_path "$COCO/annotations" --cache "$COCO/chair.pkl" --save_path "$chair" > "$SWEEP/logs/chair_${setting}.log" 2>&1; done < "$MAN"
"$PYTHON_BIN" scripts/summarize_topmass_uot.py --manifest "$MAN" --output "$SWEEP/summary.csv"; echo "Done: $SWEEP/summary.csv"
