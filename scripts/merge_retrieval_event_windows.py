#!/usr/bin/env python3
"""Merge disjoint per-shard event-window NPZ files without copying traces."""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pieces = [np.load(path, allow_pickle=False) for path in args.inputs]
    try:
        common = set(pieces[0].files)
        if any(set(piece.files) != common for piece in pieces[1:]):
            raise ValueError("Event-window shards have inconsistent schemas")
        if any(not np.array_equal(piece["relative_t"], pieces[0]["relative_t"]) for piece in pieces[1:]):
            raise ValueError("Event-window shards have different relative timestep grids")
        merged = {"relative_t": pieces[0]["relative_t"]}
        for key in common - {"relative_t"}:
            merged[key] = np.concatenate([piece[key] for piece in pieces], axis=0)
        order = np.argsort(merged["event_ids"])
        for key, value in list(merged.items()):
            if key != "relative_t":
                merged[key] = value[order]
        if len(np.unique(merged["event_ids"])) != len(merged["event_ids"]):
            raise ValueError("Duplicate event IDs while merging event windows")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **merged)
        print(f"Wrote {len(merged['event_ids'])} merged event windows to {args.output}")
    finally:
        for piece in pieces:
            piece.close()


if __name__ == "__main__":
    main()
