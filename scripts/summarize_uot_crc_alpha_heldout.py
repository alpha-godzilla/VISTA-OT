#!/usr/bin/env python3
"""Report a UOT-CRC logits-alpha sweep on strictly held-out seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry", action="append", required=True, metavar="ALPHA=MANIFEST",
        help="May be repeated; manifest is the calibrated UOT-CRC manifest.",
    )
    parser.add_argument("--heldout-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--calibration-seed", type=int, required=True)
    parser.add_argument("--by-seed-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def parse_entry(value):
    if "=" not in value:
        raise ValueError(f"Expected ALPHA=MANIFEST, got {value!r}")
    alpha, manifest = value.split("=", 1)
    return float(alpha), Path(manifest)


def load_metrics(path):
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)["overall_metrics"]
    return {name: float(values[name]) for name in METRICS}


def mean_std(values):
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    )


def load_manifest(alpha, path, heldout_seeds):
    wanted = set(heldout_seeds)
    records = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            seed = int(row["seed"])
            if seed not in wanted:
                continue
            method = row["method"]
            if method not in {"vista", "ot_stage1", "uot_crc"}:
                continue
            if method in records[seed]:
                raise ValueError(
                    f"Duplicate {method} row for alpha={alpha:g}, seed={seed}"
                )
            records[seed][method] = load_metrics(Path(row["chair_json"]))
    rows = []
    for seed in heldout_seeds:
        missing = {"vista", "ot_stage1", "uot_crc"} - set(records[seed])
        if missing:
            raise ValueError(
                f"Missing {sorted(missing)} for alpha={alpha:g}, seed={seed}"
            )
        baseline = records[seed]["vista"]
        for method in ("vista", "ot_stage1", "uot_crc"):
            values = records[seed][method]
            rows.append({
                "logits_alpha": alpha,
                "method": method,
                "seed": seed,
                **values,
                **{
                    f"delta_{name}": values[name] - baseline[name]
                    for name in METRICS
                },
            })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.calibration_seed in args.heldout_seeds:
        raise ValueError("Calibration seed must not appear in held-out seeds")
    if len(set(args.heldout_seeds)) != len(args.heldout_seeds):
        raise ValueError("Held-out seeds must be unique")

    entries = [parse_entry(value) for value in args.entry]
    if len({alpha for alpha, _ in entries}) != len(entries):
        raise ValueError("Each logits_alpha must appear exactly once")
    by_seed = []
    for alpha, manifest in sorted(entries):
        by_seed.extend(load_manifest(alpha, manifest, args.heldout_seeds))
    if not by_seed:
        raise ValueError("No held-out results found")

    summary = []
    grouped = defaultdict(list)
    for row in by_seed:
        grouped[(row["logits_alpha"], row["method"])].append(row)
    for (alpha, method), group in sorted(grouped.items()):
        item = {
            "logits_alpha": alpha,
            "method": method,
            "heldout_seeds": len(group),
        }
        for name in METRICS:
            for prefix in ("", "delta_"):
                mean, std = mean_std([row[f"{prefix}{name}"] for row in group])
                item[f"{prefix}{name}_mean"] = mean
                item[f"{prefix}{name}_std"] = std
        summary.append(item)

    write_csv(args.by_seed_csv, by_seed)
    write_csv(args.summary_csv, summary)

    uot_rows = [row for row in summary if row["method"] == "uot_crc"]
    lines = [
        "# Held-out UOT-CRC logits-alpha sweep", "",
        f"Calibration uses seed `{args.calibration_seed}` only. The table below uses only held-out seeds "
        f"`{', '.join(map(str, args.heldout_seeds))}`; no held-out metric is used for fitting or threshold selection.",
        "",
        "`logits_alpha` is VISTA/SLA's existing `--logits-alpha` intervention strength. "
        "Every alpha is independently tuned and CRC-calibrated on the calibration seed.",
        "",
        "These held-out rows are for transparent sensitivity reporting. Selecting the best alpha from this table would turn these seeds into validation data; a formal final alpha must be fixed from development data before reading this table.",
        "",
        "## Per-seed UOT-CRC absolute results", "",
        "| Alpha | Seed | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Precision ↑ | F1 ↑ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_seed:
        if row["method"] != "uot_crc":
            continue
        lines.append(
            f"| {row['logits_alpha']:g} | {row['seed']} | "
            f"{row['CHAIRs']:.4f} | {row['CHAIRi']:.4f} | "
            f"{row['Recall']:.4f} | {row['Precision']:.4f} | {row['F1']:.4f} |"
        )
    lines.extend([
        "", "## Held-out mean ± sample standard deviation", "",
        "| Alpha | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Precision ↑ | F1 ↑ | ΔCHAIRs vs VISTA | ΔF1 vs VISTA |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in uot_rows:
        lines.append(
            f"| {row['logits_alpha']:g} | "
            f"{row['CHAIRs_mean']:.4f} ± {row['CHAIRs_std']:.4f} | "
            f"{row['CHAIRi_mean']:.4f} ± {row['CHAIRi_std']:.4f} | "
            f"{row['Recall_mean']:.4f} ± {row['Recall_std']:.4f} | "
            f"{row['Precision_mean']:.4f} ± {row['Precision_std']:.4f} | "
            f"{row['F1_mean']:.4f} ± {row['F1_std']:.4f} | "
            f"{row['delta_CHAIRs_mean']:+.4f} | {row['delta_F1_mean']:+.4f} |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.by_seed_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
