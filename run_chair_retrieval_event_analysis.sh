#!/usr/bin/env bash
# Offline, measurement-only CHAIR event analysis for an existing trace run.
# The eight workers are CPU/IO workers.  This phase has no model forward pass,
# so assigning CUDA devices would not make it faster or use the V100s.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-retrieval_shift_200_seed1994}"
TRACE_ROOT="${RETRIEVAL_SHIFT_DATA_ROOT:-/data/sun_yuxi/retrieval_shift_data}/$RUN_NAME"
OUT="$ROOT/exp_results/$RUN_NAME"
EVENT_DATA_ROOT="${RETRIEVAL_EVENT_DATA_ROOT:-/data/sun_yuxi/retrieval_shift_event_analysis}/$RUN_NAME"
MODEL_PATH="${VISTA_LLAVA_MODEL_PATH:-/data/sun_yuxi/models/llava-v1.5-7b}"
CHAIR_CACHE="${CHAIR_CACHE:-/data/sun_yuxi/datasets/coco/chair.pkl}"
COCO_PATH="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}/annotations"
WINDOW="${WINDOW:-5}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
read -r -a WORKERS <<< "${WORKER_IDS:-0 1 2 3 4 5 6 7}"
(( ${#WORKERS[@]} > 0 )) || { echo "WORKER_IDS must not be empty" >&2; exit 1; }
[[ -d "$TRACE_ROOT" && -f "$MODEL_PATH/config.json" ]] || { echo "Set RETRIEVAL_SHIFT_DATA_ROOT and VISTA_LLAVA_MODEL_PATH to existing paths." >&2; exit 1; }
mkdir -p "$OUT/logs" "$OUT/chair_retrieval_analysis" "$EVENT_DATA_ROOT/windows"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vista-retrieval-mpl}"
mkdir -p "$MPLCONFIGDIR"

echo "[1/4] Extracting strict CHAIR object/token events (CPU only)"
"$PYTHON_BIN" scripts/extract_chair_object_events.py \
  --trace-root "$TRACE_ROOT" --output-dir "$OUT" --model-path "$MODEL_PATH" \
  --chair-cache "$CHAIR_CACHE" --coco-path "$COCO_PATH" \
  > "$OUT/logs/chair_event_extraction.log" 2>&1
failure_count=$(( $(wc -l < "$OUT/chair_alignment_failures.csv") - 1 ))
if (( failure_count > 0 )) && [[ "${ALLOW_ALIGNMENT_FAILURES:-0}" != "1" ]]; then
  echo "Strict alignment found $failure_count failures; stopping before analysis." >&2
  echo "Inspect $OUT/chair_alignment_failures.csv and $OUT/token_query_alignment_audit.csv" >&2
  exit 1
fi
[[ -s "$OUT/chair_events_token_aligned.csv" ]] || { echo "No aligned object events" >&2; exit 1; }

echo "[2/4] Extracting compact event windows with ${#WORKERS[@]} parallel CPU/IO workers"
declare -a PIDS=() WINDOW_SHARDS=()
run_worker() {
  local idx="$1"
  local manifest="$OUT/manifests/shard_${idx}.txt"
  local target="$EVENT_DATA_ROOT/windows/shard_${idx}.npz"
  [[ -s "$manifest" ]] || { echo "Missing trace manifest $manifest" >&2; return 1; }
  "$PYTHON_BIN" scripts/extract_retrieval_event_windows.py \
    --events "$OUT/chair_events_token_aligned.csv" --trace-root "$TRACE_ROOT" \
    --sample-ids "$manifest" --window "$WINDOW" --output "$target" \
    > "$OUT/logs/event_window_shard_${idx}.log" 2>&1
}
for idx in "${!WORKERS[@]}"; do
  run_worker "$idx" & PIDS+=("$!")
  WINDOW_SHARDS+=("$EVENT_DATA_ROOT/windows/shard_${idx}.npz")
done
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "At least one event-window worker failed; inspect $OUT/logs" >&2; exit 1; }

echo "[3/4] Merging compact event windows (not raw traces)"
MERGED="$EVENT_DATA_ROOT/event_window_metrics.npz"
"$PYTHON_BIN" scripts/merge_retrieval_event_windows.py --inputs "${WINDOW_SHARDS[@]}" --output "$MERGED" \
  > "$OUT/logs/event_window_merge.log" 2>&1

echo "[4/4] Position-matched, image-clustered event statistics (CPU only)"
"$PYTHON_BIN" scripts/analyze_chair_retrieval_events.py \
  --events "$OUT/chair_events_token_aligned.csv" --windows "$MERGED" \
  --output-dir "$OUT/chair_retrieval_analysis" --bootstrap "$BOOTSTRAP" \
  > "$OUT/logs/chair_retrieval_analysis.log" 2>&1
echo "Analysis complete: $OUT/chair_retrieval_analysis"
echo "Compact event-window tensor: $MERGED"
