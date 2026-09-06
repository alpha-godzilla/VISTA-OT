#!/usr/bin/env python3
"""Descriptive grounded-vs-hallucinated analysis of aligned retrieval traces."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ("size_effect", "query_effect", "new_K_effect", "net_pressure", "compensation_margin", "compat_GV", "mass_ratio_log")


def bootstrap_mean(values, rng, n):
    if not values:
        return 0.0, 0.0, 0.0
    values = np.asarray(values, dtype=float)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)])
    return values.mean(), *np.quantile(means, [0.025, 0.975])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.aligned.open(encoding="utf-8")))
    if not rows:
        raise ValueError("Aligned table is empty")
    # First pool heads/layers within each event; bootstrap therefore operates on
    # annotated events rather than treating heads as independent samples.
    unit = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["sample_id"], row["event_token_index"], row["event_type"], row["event_text"], row["relative_t"])
        for metric in METRICS:
            unit[key][metric].append(float(row[metric]))
    pooled = defaultdict(lambda: defaultdict(list))
    for key, metrics in unit.items():
        event_type, relative_t = key[2], int(key[4])
        for metric, values in metrics.items():
            pooled[(event_type, relative_t)][metric].append(float(np.mean(values)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fields = ("event_type", "relative_t", "metric", "event_count", "mean", "ci_low", "ci_high")
    with (args.output_dir / "event_bootstrap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for (event_type, relative_t), metrics in sorted(pooled.items()):
            for metric, values in metrics.items():
                mean, lo, hi = bootstrap_mean(values, rng, args.bootstrap)
                writer.writerow(dict(event_type=event_type, relative_t=relative_t, metric=metric, event_count=len(values), mean=mean, ci_low=lo, ci_high=hi))
    cumulative = defaultdict(lambda: defaultdict(list))
    for key, metrics in unit.items():
        sample, event_index, event_type, event_text, relative_t = key
        if -5 <= int(relative_t) <= -1:
            event_key = (sample, event_index, event_type, event_text)
            for metric in ("size_effect", "query_effect", "new_K_effect", "net_pressure"):
                cumulative[event_key][metric].append(float(np.mean(metrics[metric])))
    with (args.output_dir / "cumulative_pre_event.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("event_type", "metric", "event_count", "mean", "ci_low", "ci_high")); writer.writeheader()
        by_type = defaultdict(lambda: defaultdict(list))
        for (_, _, event_type, _), metrics in cumulative.items():
            for metric, values in metrics.items():
                by_type[event_type][metric].append(float(np.sum(values)))
        for event_type, metrics in by_type.items():
            for metric, values in metrics.items():
                mean, lo, hi = bootstrap_mean(values, rng, args.bootstrap)
                writer.writerow(dict(event_type=event_type, metric=f"A_{metric}", event_count=len(values), mean=mean, ci_low=lo, ci_high=hi))
    # Layer-wise and head-wise hallucinated-minus-grounded differences, kept
    # separate for every requested metric.
    for axis in ("layer", "head"):
        for metric in METRICS:
            grouped = defaultdict(list)
            for row in rows:
                grouped[(row["event_type"], int(row["relative_t"]), int(row[axis]))].append(float(row[metric]))
            rels = sorted({int(row["relative_t"]) for row in rows})
            indices = sorted({int(row[axis]) for row in rows})
            image = np.zeros((len(rels), len(indices)))
            for i, rel in enumerate(rels):
                for j, idx in enumerate(indices):
                    hall = grouped[("hallucinated", rel, idx)]
                    ground = grouped[("grounded", rel, idx)]
                    image[i, j] = np.mean(hall) - np.mean(ground) if hall and ground else 0.0
            fig, ax = plt.subplots(figsize=(max(8, len(indices) * .3), 4))
            im = ax.imshow(image, aspect="auto", cmap="coolwarm")
            ax.set_title(f"Hallucinated − grounded {metric} by {axis}")
            ax.set_xlabel(axis); ax.set_ylabel("relative timestep")
            ax.set_yticks(range(len(rels)), rels); fig.colorbar(im, ax=ax); fig.tight_layout()
            fig.savefig(args.output_dir / f"{metric}_{axis}_difference.png", dpi=180); plt.close(fig)
    # Descriptive sign probabilities at head/layer/event-row level.
    with (args.output_dir / "sign_probabilities.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("event_type", "statistic", "probability", "records")); writer.writeheader()
        for kind in ("grounded", "hallucinated"):
            subset = [row for row in rows if row["event_type"] == kind]
            for name, fn in (("query_effect_lt_0", lambda x: x < 0), ("new_K_effect_gt_0", lambda x: x > 0), ("net_pressure_gt_0", lambda x: x > 0)):
                metric = "query_effect" if name.startswith("query") else "new_K_effect" if name.startswith("new") else "net_pressure"
                writer.writerow(dict(event_type=kind, statistic=name, probability=np.mean([fn(float(r[metric])) for r in subset]) if subset else 0.0, records=len(subset)))
    print(f"Wrote event analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
