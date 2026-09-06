#!/usr/bin/env bash
# Gate 1 only: offline re-audit of already collected event windows.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-retrieval_shift_200_seed1994}"
OUT="$ROOT/exp_results/$RUN_NAME"
WINDOWS="${RETRIEVAL_EVENT_DATA_ROOT:-/data/sun_yuxi/retrieval_shift_event_analysis}/$RUN_NAME/event_window_metrics.npz"
ANALYSIS="$OUT/chair_retrieval_analysis/strict_statistics_audit"
[[ -f "$OUT/chair_events_token_aligned.csv" && -f "$OUT/chair_retrieval_analysis/matched_event_pairs.csv" && -f "$WINDOWS" ]] || {
  echo "Missing aligned events, matched pairs, or compact event windows for $RUN_NAME" >&2; exit 1;
}
mkdir -p "$ANALYSIS"
"$PYTHON_BIN" scripts/audit_retrieval_event_statistics.py \
  --events "$OUT/chair_events_token_aligned.csv" --windows "$WINDOWS" \
  --pairs "$OUT/chair_retrieval_analysis/matched_event_pairs.csv" \
  --output-dir "$ANALYSIS" --bootstrap "${BOOTSTRAP:-2000}" \
  | tee "$ANALYSIS/run.log"
echo "Gate 1 complete: $ANALYSIS"
