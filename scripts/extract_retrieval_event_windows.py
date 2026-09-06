#!/usr/bin/env python3
"""Extract compact event-centered [event, relative_t, layer, head] metrics.

The source traces remain untouched.  This script writes only the small subset
of rows around strictly token-aligned CHAIR object events.  It is intended to
run once per trace shard and be merged afterwards.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


METRICS = (
    "size_effect", "query_effect", "query_content_effect", "query_position_effect",
    "new_K_effect", "compat_GV", "compat_PV", "m_V", "m_P", "m_G",
    "mass_ratio_log", "n_V", "n_P", "n_G",
)


def trace_paths(root):
    return {int(path.stem.removeprefix("sample_")): path for path in Path(root).rglob("sample_*.npz")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path, default=None, help="Optional one-image-id-per-line shard filter")
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()
    if args.window < 0:
        raise ValueError("--window must be non-negative")
    allowed = None if args.sample_ids is None else {int(line) for line in args.sample_ids.read_text().splitlines() if line.strip()}
    events = []
    with args.events.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if allowed is None or int(row["sample_id"]) in allowed:
                events.append(row)
    if not events:
        raise ValueError("No aligned events selected for this shard")
    paths = trace_paths(args.trace_root)
    relative = np.arange(-args.window, 1, dtype=np.int16)
    by_sample = {}
    for event_idx, event in enumerate(events):
        by_sample.setdefault(int(event["sample_id"]), []).append(event_idx)
    first_path = paths[int(events[0]["sample_id"])]
    with np.load(first_path, allow_pickle=False) as data:
        _, layers, heads = data["compat_GV"].shape
    arrays = {metric: np.zeros((len(events), len(relative), layers, heads), dtype=np.float16) for metric in METRICS}
    valid = np.zeros((len(events), len(relative)), dtype=np.bool_)
    for sample_id, event_indices in by_sample.items():
        if sample_id not in paths:
            raise FileNotFoundError(f"No trace for aligned event sample {sample_id}")
        with np.load(paths[sample_id], allow_pickle=False) as data:
            for event_idx in event_indices:
                decision_t = int(events[event_idx]["decision_query_timestep"])
                for relative_idx, rel in enumerate(relative):
                    timestep = decision_t + int(rel)
                    if timestep < 0 or timestep >= data["token_ids"].shape[0]:
                        continue
                    valid[event_idx, relative_idx] = True
                    for metric in METRICS:
                        source = data[metric]
                        if source.ndim == 3:
                            arrays[metric][event_idx, relative_idx] = source[timestep]
                        elif source.ndim == 2:
                            arrays[metric][event_idx, relative_idx] = source[timestep, :, None]
                        else:
                            raise ValueError(f"Unexpected {metric} shape {source.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        event_ids=np.asarray([int(event["event_id"]) for event in events], dtype=np.int64),
        relative_t=relative,
        valid=valid,
        **arrays,
    )
    print(f"Wrote {len(events)} event windows to {args.output}")


if __name__ == "__main__":
    main()
