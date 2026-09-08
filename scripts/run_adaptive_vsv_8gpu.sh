#!/usr/bin/env bash
# Independent-process sweep; every GPU receives a disjoint deterministic shard.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"; RUN="${RUN_NAME:-adaptive_vsv_laws}"; SEED="${SEED:-1994}"
COCO="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"; OUT="exp_results/$RUN"; MODEL_DATA="${ADAPTIVE_VSV_DATA_ROOT:-/data/sun_yuxi/adaptive_vsv_laws}/$RUN"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"; GRID="${LAMBDA_GRID:-0.00,0.05,0.08,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18}"
mkdir -p "$OUT/manifests" "$OUT/logs" "$MODEL_DATA/shards"
MASTER="$OUT/manifests/master_500_ids.txt"
[[ -f "$MASTER" ]] || "$PYTHON_BIN" scripts/make_chair_seed_manifest.py --data-path "$COCO/val2014" --seed "$SEED" --subset-size 500 --output "$MASTER"
if [[ ! -f "$OUT/manifests/calibration_ids.txt" ]]; then head -n 300 "$MASTER" > "$OUT/manifests/calibration_ids.txt"; tail -n 200 "$MASTER" > "$OUT/manifests/heldout_ids.txt"; fi
TARGET_IDS="${TARGET_IDS:-$MASTER}"; CONTROLLER="${CONTROLLER:-fixed}"
for i in "${!GPUS[@]}"; do awk -v n="${#GPUS[@]}" -v x="$i" '((NR-1)%n)==x{print}' "$TARGET_IDS" > "$OUT/manifests/shard_${i}.txt"; done
declare -a PIDS=()
for i in "${!GPUS[@]}"; do
  shard="$OUT/manifests/shard_${i}.txt"; output="$MODEL_DATA/shards/shard_${i}.jsonl"
  extra=(); [[ "$CONTROLLER" == geometry ]] && extra+=(--geometry-target "${GEOMETRY_TARGET:?set GEOMETRY_TARGET for geometry controller}");
  extra+=(--law-scale "${LAW_SCALE:-0.17}" --law-feature "${LAW_FEATURE:-relative_raw_diff_norm_mean}")
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" scripts/chair_adaptive_sweep.py --image-id-file "$shard" --output "$output" --data-path "$COCO/val2014" --lambda-grid "$GRID" --controller "$CONTROLLER" --vsv-sim-gate "${VSV_SIM_GATE:-legacy}" --max-new-tokens "${MAX_NEW_TOKENS:-128}" "${extra[@]}" --resume > "$OUT/logs/shard_${i}.log" 2>&1 & PIDS+=("$!")
done
failed=0; for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done; ((failed==0)) || { echo "adaptive VSV shard failed; inspect $OUT/logs" >&2; exit 1; }
"$PYTHON_BIN" scripts/merge_adaptive_vsv_shards.py --shards "$MODEL_DATA"/shards/shard_*.jsonl --image-id-file "$TARGET_IDS" --lambda-grid "$GRID" --output "$MODEL_DATA/merged.jsonl"
echo "Adaptive VSV sweep complete: $MODEL_DATA/merged.jsonl"
