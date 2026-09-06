#!/usr/bin/env bash
# Eight-GPU, measurement-only retrieval-shift collection.  Every worker owns
# separate manifests, result folders, trace shards, and logs.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-1994}"
SAMPLES="${SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RUN_NAME="${RUN_NAME:-retrieval_shift_multisample_seed${SEED}}"
COCO="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
(( ${#GPUS[@]} > 0 )) || { echo "GPU_IDS must not be empty" >&2; exit 1; }
export VISTA_COCO_ROOT="$COCO" NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
[[ -n "${HF_HOME:-}" || -n "${HUGGINGFACE_HUB_CACHE:-}" || ! -d /data/sun_yuxi/huggingface ]] || export HF_HOME=/data/sun_yuxi/huggingface
if [[ -z "${VISTA_LLAVA_MODEL_PATH:-}" ]]; then
  for p in /data/sun_yuxi/models/llava-v1.5-7b /data/sun_yuxi/models/llava-1.5-7b-hf; do
    [[ -f "$p/config.json" ]] && { export VISTA_LLAVA_MODEL_PATH="$p"; break; }
  done
fi
[[ -d "$COCO/val2014" && -f "${VISTA_LLAVA_MODEL_PATH:-}/config.json" ]] || {
  echo "Set VISTA_COCO_ROOT and VISTA_LLAVA_MODEL_PATH to valid local directories." >&2; exit 1;
}
OUT="$ROOT/exp_results/$RUN_NAME"; mkdir -p "$OUT/manifests" "$OUT/logs"
IDS="$OUT/manifests/seed_${SEED}_${SAMPLES}_ids.txt"
[[ -f "$IDS" && "$(wc -l < "$IDS")" -eq "$SAMPLES" ]] || "$PYTHON_BIN" scripts/make_chair_seed_manifest.py --data-path "$COCO/val2014" --seed "$SEED" --subset-size "$SAMPLES" --output "$IDS"
for i in "${!GPUS[@]}"; do
  shard="$OUT/manifests/shard_${i}.txt"
  awk -v n="${#GPUS[@]}" -v idx="$i" '((NR-1)%n)==idx {print}' "$IDS" > "$shard"
  [[ -s "$shard" ]] || { echo "Empty shard: $shard" >&2; exit 1; }
done
run_worker() {
  local idx="$1" gpu="${GPUS[$1]}" shard="$OUT/manifests/shard_${idx}.txt"
  local shard_count; shard_count="$(wc -l < "$shard")"
  local exp="${RUN_NAME}/shard_${idx}" trace="$OUT/traces/shard_${idx}"
  echo "[GPU $gpu] collecting $shard_count samples into shard_${idx}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_eval.py \
    --model llava-1.5 --exp_folder "$exp" --data-path "$COCO/val2014" \
    --subset-size "$shard_count" --subset-ids-file "$shard" --seed "$SEED" \
    --max-new-tokens "$MAX_NEW_TOKENS" --num-beams 1 \
    --retrieval-shift-trace-dir "$trace" \
    > "$OUT/logs/shard_${idx}.log" 2>&1
}
declare -a PIDS=()
for i in "${!GPUS[@]}"; do run_worker "$i" & PIDS+=("$!"); done
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "At least one shard failed; inspect $OUT/logs" >&2; exit 1; }
echo "Collection complete: $OUT"
echo "Per-sample traces are sharded under $OUT/traces; do not build a monolithic summary for 200+ samples."
