#!/usr/bin/env python3
"""Token-level temporal visualizations for one retrieval-shift trace.

This is a post-hoc reader only: it never loads a model or changes decoding.
The source trace remains headwise; head means are an explicitly labelled view.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ("compat_GV", "compat_PV", "query_effect", "new_K_effect", "m_V", "m_G")
VALIDITY = {
    "compat_GV": "has_G",
    "compat_PV": "has_P",
    "query_effect": "query_effect_valid",
    "new_K_effect": "new_K_effect_valid",
    "m_V": "has_V",
    "m_G": "has_G",
}


def head_mean(values, valid):
    """Mean across heads, retaining a valid-head count and finite zeros."""
    valid = valid.astype(bool)
    count = valid.sum(axis=-1)
    total = (values * valid).sum(axis=-1)
    return total / np.maximum(count, 1), count


def plot_heatmap(values, tokens, metric, output):
    # values [T, L], rendered layers × timesteps.
    fig_width = max(8, min(24, values.shape[0] * 0.22))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    image = ax.imshow(values.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_title(f"{metric}: head mean")
    ax.set_xlabel("Decoding timestep / current token")
    ax.set_ylabel("Layer")
    ticks = np.arange(0, len(tokens), max(1, len(tokens) // 16))
    labels = [f"{t}:{tokens[t]!r}" for t in ticks]
    ax.set_xticks(ticks, labels, rotation=55, ha="right", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True, help="One sample_<id>.npz trace")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.trace, allow_pickle=False) as trace:
        missing = [name for name in METRICS if name not in trace]
        if missing:
            raise ValueError(f"Trace lacks expected fields: {missing}")
        tokens = trace["token_text"].tolist()
        aggregate, counts = {}, {}
        for metric in METRICS:
            aggregate[metric], counts[f"{metric}_valid_heads"] = head_mean(
                trace[metric], trace[VALIDITY[metric]],
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_dir / "temporal_headmean.npz",
            token_ids=trace["token_ids"], token_text=trace["token_text"],
            **aggregate, **counts,
        )
        with (args.output_dir / "temporal_headmean.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("timestep", "token_id", "token_text", "layer", *METRICS, *counts))
            writer.writeheader()
            for timestep, (token_id, token) in enumerate(zip(trace["token_ids"], tokens)):
                for layer in range(trace["compat_GV"].shape[1]):
                    row = {"timestep": timestep, "token_id": int(token_id), "token_text": token, "layer": layer}
                    row.update({metric: float(aggregate[metric][timestep, layer]) for metric in METRICS})
                    row.update({name: int(counts[name][timestep, layer]) for name in counts})
                    writer.writerow(row)
        for metric in METRICS:
            plot_heatmap(aggregate[metric], tokens, metric, args.output_dir / f"{metric}_heatmap.png")
    print(f"Wrote temporal views to {args.output_dir}")


if __name__ == "__main__":
    main()
