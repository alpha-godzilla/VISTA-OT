#!/usr/bin/env python3
"""Compare generation-time and fixed-token replay compact traces exactly."""
import argparse
import json
from pathlib import Path
import numpy as np

METRICS = ("m_V", "m_G", "compat_GV", "compat_PV", "query_content_effect", "query_position_effect", "new_K_effect")

def paths(root):
    return {int(item.stem.removeprefix("sample_")): item for item in Path(root).rglob("sample_*.npz")}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--reference-root", required=True, type=Path)
    p.add_argument("--replay-root", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-mean-abs", type=float, default=1e-3); p.add_argument("--max-p99-abs", type=float, default=5e-3)
    args = p.parse_args(); old, replay = paths(args.reference_root), paths(args.replay_root)
    common = sorted(set(old) & set(replay))
    if not common: raise RuntimeError("No common replay/reference traces")
    rows = []
    for sample_id in common:
        with np.load(old[sample_id]) as a, np.load(replay[sample_id]) as b:
            if not np.array_equal(a["token_ids"], b["token_ids"]): raise RuntimeError(f"token mismatch: {sample_id}")
            for name in METRICS:
                if name not in a or name not in b: continue
                error = np.abs(a[name].astype(np.float32) - b[name].astype(np.float32)).ravel()
                rows.append({"sample_id": sample_id, "metric": name, "max_abs": float(error.max()),
                             "mean_abs": float(error.mean()), "p99_abs": float(np.quantile(error, .99))})
    passed = all(row["mean_abs"] <= args.max_mean_abs and row["p99_abs"] <= args.max_p99_abs for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": passed, "samples": len(common), "thresholds": {"max_mean_abs": args.max_mean_abs, "max_p99_abs": args.max_p99_abs}, "rows": rows}, indent=2))
    print(args.output); print("PASS" if passed else "FAIL")
    if not passed: raise SystemExit(2)

if __name__ == "__main__": main()
