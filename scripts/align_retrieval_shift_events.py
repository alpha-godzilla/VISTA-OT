#!/usr/bin/env python3
"""Align externally annotated generated tokens with retrieval-shift traces."""
import argparse
import csv
from pathlib import Path

import numpy as np

METRICS = (
    "size_effect", "query_effect", "new_K_effect", "net_pressure",
    "compensation_margin", "compat_GV", "compat_PV", "m_V", "m_P",
    "m_G", "mass_ratio_log",
)


def traces_by_sample(root):
    result = {}
    for path in root.rglob("sample_*.npz"):
        sample_id = int(path.stem.removeprefix("sample_"))
        if sample_id in result:
            raise ValueError(f"Duplicate trace for sample_id={sample_id}: {path} and {result[sample_id]}")
        result[sample_id] = path
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()
    required = {"sample_id", "event_token_index", "event_type", "event_text"}
    with args.annotations.open(newline="", encoding="utf-8") as handle:
        annotations = list(csv.DictReader(handle))
    if not annotations or not required.issubset(annotations[0]):
        raise ValueError("Annotations need sample_id,event_token_index,event_type,event_text columns")
    traces = traces_by_sample(args.trace_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("sample_id", "event_token_index", "event_type", "event_text", "relative_t", "timestep", "layer", "head", *METRICS)
    kept = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in annotations:
            sample_id, event_t = int(event["sample_id"]), int(event["event_token_index"])
            if event["event_type"] not in {"grounded", "hallucinated"}:
                raise ValueError(f"Unknown event_type={event['event_type']!r}")
            if sample_id not in traces:
                raise ValueError(f"No trace for sample_id={sample_id}")
            with np.load(traces[sample_id], allow_pickle=False) as data:
                if not 0 <= event_t < data["token_ids"].shape[0]:
                    raise ValueError(f"event token index {event_t} outside trace for sample_id={sample_id}")
                for relative_t in range(-args.window, 1):
                    timestep = event_t + relative_t
                    if timestep < 0:
                        continue
                    for layer in range(data["compat_GV"].shape[1]):
                        for head in range(data["compat_GV"].shape[2]):
                            row = {
                                "sample_id": sample_id, "event_token_index": event_t,
                                "event_type": event["event_type"], "event_text": event["event_text"],
                                "relative_t": relative_t, "timestep": timestep,
                                "layer": layer, "head": head,
                            }
                            row.update({metric: float(data[metric][timestep, layer, head]) for metric in METRICS})
                            writer.writerow(row)
                            kept += 1
    print(f"Wrote {kept} aligned head records to {args.output}")


if __name__ == "__main__":
    main()
