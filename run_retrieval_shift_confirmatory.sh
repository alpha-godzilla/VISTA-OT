#!/usr/bin/env bash
# Confirmatory retrieval-shift study. Stages are intentionally explicit: do
# not start the long held-out generation before inspecting Stage-0 validation.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"; RUN_NAME="${RUN_NAME:-retrieval_shift_confirmatory_sameimage_v1}"
STAGE="${STAGE:-preflight}"; SEED="${SEED:-20260907}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
COCO="${VISTA_COCO_ROOT:-/data/sun_yuxi/datasets/coco}"; DATA_ROOT="${RETRIEVAL_SHIFT_DATA_ROOT:-/data/sun_yuxi/retrieval_shift_data}"
OUT="$ROOT/exp_results/$RUN_NAME"; HEAVY="$DATA_ROOT/$RUN_NAME"; OLD_RUN="$ROOT/exp_results/retrieval_shift_200_seed1994"
MODEL_PATH="${VISTA_LLAVA_MODEL_PATH:-/data/sun_yuxi/models/llava-v1.5-7b}"; CHAIR_CACHE="${CHAIR_CACHE:-$COCO/chair.pkl}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"; export VISTA_COCO_ROOT="$COCO" VISTA_LLAVA_MODEL_PATH="$MODEL_PATH" NLTK_DATA="${NLTK_DATA:-/data/sun_yuxi/nltk_data}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
[[ -n "${HF_HOME:-}" || ! -d /data/sun_yuxi/huggingface ]] || export HF_HOME=/data/sun_yuxi/huggingface
[[ -d "$COCO/val2014" && -f "$MODEL_PATH/config.json" ]] || { echo "Invalid COCO/model path" >&2; exit 1; }
mkdir -p "$OUT/manifests" "$OUT/logs" "$HEAVY"
if [[ "$STAGE" == preflight ]]; then
  OLD_TRACE_ROOT="${OLD_TRACE_ROOT:-${RETRIEVAL_SHIFT_DATA_ROOT:-/data/sun_yuxi/retrieval_shift_data}/retrieval_shift_200_seed1994}"
  "$PYTHON_BIN" scripts/validate_confirmatory_preflight.py --old-trace-root "$OLD_TRACE_ROOT" --max-new-tokens "$MAX_NEW_TOKENS" --output "$OUT/config_recovery.json"
  "$PYTHON_BIN" scripts/make_confirmatory_heldout_manifest.py --data-path "$COCO/val2014" --old-run "$OLD_RUN" --output-dir "$OUT/manifests" --seed "$SEED"
  git rev-parse HEAD > "$OUT/git_commit.txt"; git status --short > "$OUT/git_dirty_status.txt"
  "$PYTHON_BIN" - <<'PY' > "$OUT/environment.json"
import json, torch, transformers
print(json.dumps({"torch":torch.__version__,"transformers":transformers.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},indent=2))
PY
  echo "Preflight manifest written: $OUT/manifests (overlap must be 0)."
  exit 0
fi
generate() {
  local count="$1" offset="$2" tag="$3" candidate="$OUT/manifests/new_candidate_image_ids.txt" ids="$OUT/manifests/${tag}_ids.txt"
  sed -n "$((offset+1)),$((offset+count))p" "$candidate" > "$ids"; [[ -s "$ids" ]] || { echo "No candidate IDs for $tag" >&2; return 1; }
  local i; for i in "${!GPUS[@]}"; do awk -v n="${#GPUS[@]}" -v x="$i" '((NR-1)%n)==x{print}' "$ids" > "$OUT/manifests/${tag}_shard_${i}.txt"; done
  declare -a PIDS=(); local start; start="$(date +%s)"
  worker() { local idx="$1"; local gpu="${GPUS[$idx]}"; local shard="$OUT/manifests/${tag}_shard_${idx}.txt"; local exp="$RUN_NAME/${tag}/shard_${idx}"; local rec="$HEAVY/generation_shards/${tag}/shard_${idx}"; mkdir -p "$rec"; echo "[GPU $gpu] $tag $(wc -l < "$shard") images";
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" chair_eval.py --model llava-1.5 --exp_folder "$exp" --data-path "$COCO/val2014" --subset-ids-file "$shard" --subset-size "$(wc -l < "$shard")" --seed "$SEED" --max-new-tokens "$MAX_NEW_TOKENS" --num-beams 1 --generation-record-dir "$rec" --generation-decision-stats --resume > "$OUT/logs/${tag}_shard_${idx}.log" 2>&1; }
  for i in "${!GPUS[@]}"; do worker "$i" & PIDS+=("$!"); done; local failed=0 pid; for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
  local end; end="$(date +%s)"; printf '{"tag":"%s","requested":%s,"seconds":%s}\n' "$tag" "$count" "$((end-start))" > "$OUT/${tag}_runtime.json"; ((failed==0)) || return 1
}
if [[ "$STAGE" == benchmark ]]; then generate "${BENCHMARK_IMAGES:-64}" 0 benchmark; echo "Run STAGE=chair_benchmark next; inspect throughput before Phase A."; exit 0; fi
if [[ "$STAGE" == replay_validate ]]; then
  OLD_TRACE_ROOT="${OLD_TRACE_ROOT:-${RETRIEVAL_SHIFT_DATA_ROOT:-/data/sun_yuxi/retrieval_shift_data}/retrieval_shift_200_seed1994}"
  OLD_IDS="$OUT/manifests/replay_validation_ids.txt"
  find "$OLD_TRACE_ROOT" -name 'sample_*.npz' -printf '%f\n' | sed 's/sample_//;s/.npz//' | sort -n | head -n "${VALIDATION_IMAGES:-16}" > "$OLD_IDS"
  [[ -s "$OLD_IDS" ]] || { echo "No old traces at $OLD_TRACE_ROOT" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" scripts/replay_retrieval_shift_tokens.py --generation-root "$OLD_TRACE_ROOT" --sample-ids "$OLD_IDS" --data-path "$COCO/val2014" --output-dir "$HEAVY/replay_validation" --compact
  "$PYTHON_BIN" scripts/validate_replay_retrieval_shift.py --reference-root "$OLD_TRACE_ROOT" --replay-root "$HEAVY/replay_validation" --output "$OUT/replay_validation.json"
  echo "Stage-0 replay validation passed."; exit 0
fi
if [[ "$STAGE" == phase_a ]]; then generate "${PHASE_A_IMAGES:?set PHASE_A_IMAGES from benchmark}" "${PHASE_A_OFFSET:-64}" phase_a; echo "Phase A done; run STAGE=chair."; exit 0; fi
if [[ "$STAGE" == chair ]]; then
  "$PYTHON_BIN" scripts/extract_chair_object_events.py --generation-root "$HEAVY/generation_shards" --output-dir "$OUT" --model-path "$MODEL_PATH" --chair-cache "$CHAIR_CACHE" --coco-path "$COCO/annotations"
  "$PYTHON_BIN" scripts/build_confirmatory_cohorts.py --events "$OUT/chair_events_token_aligned.csv" --output-dir "$OUT/cohorts"
  echo "CHAIR/cohorts complete. Review $OUT/cohorts/cohort_yield.csv before targeted replay."; exit 0
fi
if [[ "$STAGE" == replay ]]; then
  ids="$OUT/cohorts/targeted_replay_image_ids.txt"
  "$PYTHON_BIN" - <<'PY' "$OUT/cohorts/same_image_pairs_tolerance5.csv" "$OUT/cohorts/same_category_pairs_tolerance5.csv" "$ids"
import csv,sys
x=set()
for p,fields in ((sys.argv[1],('image_id',)),(sys.argv[2],('hall_image_id','ground_image_id'))):
  with open(p) as f:
    for r in csv.DictReader(f): x.update(int(r[k]) for k in fields)
open(sys.argv[3],'w').write(''.join(f'{z}\n' for z in sorted(x)))
PY
  # One GPU per deterministic image shard; each writes only its own trace tree.
  for i in "${!GPUS[@]}"; do awk -v n="${#GPUS[@]}" -v x="$i" '((NR-1)%n)==x{print}' "$ids" > "$OUT/manifests/replay_shard_${i}.txt"; done
  declare -a PIDS=(); for i in "${!GPUS[@]}"; do CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" scripts/replay_retrieval_shift_tokens.py --generation-root "$HEAVY/generation_shards" --sample-ids "$OUT/manifests/replay_shard_${i}.txt" --data-path "$COCO/val2014" --output-dir "$HEAVY/replay_shards/shard_${i}" --compact > "$OUT/logs/replay_shard_${i}.log" 2>&1 & PIDS+=("$!"); done
  failed=0; for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done; ((failed==0)) || { echo "Targeted replay failed; inspect logs" >&2; exit 1; }
  echo "Replay done. Use the existing compact-window extractor then STAGE=analyze."; exit 0
fi
if [[ "$STAGE" == analyze ]]; then
  mkdir -p "$HEAVY/event_windows"
  declare -a WINDOWS=() PIDS=()
  for i in "${!GPUS[@]}"; do
    window="$HEAVY/event_windows/shard_${i}.npz"; WINDOWS+=("$window")
    "$PYTHON_BIN" scripts/extract_retrieval_event_windows.py --events "$OUT/chair_events_token_aligned.csv" --trace-root "$HEAVY/replay_shards" --sample-ids "$OUT/manifests/replay_shard_${i}.txt" --window 5 --output "$window" > "$OUT/logs/window_shard_${i}.log" 2>&1 & PIDS+=("$!")
  done
  failed=0; for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done; ((failed==0)) || { echo "Window extraction failed" >&2; exit 1; }
  "$PYTHON_BIN" scripts/merge_retrieval_event_windows.py --inputs "${WINDOWS[@]}" --output "$HEAVY/event_window_metrics.npz"
  "$PYTHON_BIN" scripts/analyze_confirmatory_retrieval.py --events "$OUT/chair_events_token_aligned.csv" --windows "$HEAVY/event_window_metrics.npz" --same-image-pairs "$OUT/cohorts/same_image_pairs_tolerance5.csv" --same-category-pairs "$OUT/cohorts/same_category_pairs_tolerance5.csv" --output-dir "$OUT/analysis"
  echo "Confirmatory analysis complete: $OUT/analysis"; exit 0
fi
echo "Unknown STAGE=$STAGE" >&2; exit 2
