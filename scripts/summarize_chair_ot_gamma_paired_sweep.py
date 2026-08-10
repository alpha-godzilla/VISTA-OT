#!/usr/bin/env python3
"""Summarize paired VISTA/OT CHAIR results for each SLA gamma."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--ot-topk", required=True, type=int)
    parser.add_argument("--ot-visual-tokens", required=True, type=int)
    parser.add_argument("--vsv-lambda", required=True, type=float)
    return parser.parse_args()


def metrics(path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)["overall_metrics"]
    return {name: float(data[name]) for name in METRICS}


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main():
    cli = args()
    pairs = defaultdict(dict)
    with cli.manifest.open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            pairs[(float(entry["gamma"]), int(entry["seed"]))][entry["method"]] = metrics(Path(entry["chair_json"]))
    rows = []
    for (gamma, seed), pair in sorted(pairs.items()):
        if set(pair) != {"vista", "ot"}:
            raise ValueError(f"Incomplete pair for gamma={gamma:g}, seed={seed}")
        row = {"gamma": gamma, "seed": seed}
        for method in ("vista", "ot"):
            row.update({f"{method}_{name}": value for name, value in pair[method].items()})
        row.update({f"delta_{name}": pair["ot"][name] - pair["vista"][name] for name in METRICS})
        rows.append(row)
    fields = ["gamma", "seed", *(f"{method}_{name}" for method in ("vista", "ot") for name in METRICS), *(f"delta_{name}" for name in METRICS)]
    cli.csv.parent.mkdir(parents=True, exist_ok=True)
    with cli.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# Paired VISTA–OT Gamma Sweep", "", f"VSV lambda: {cli.vsv_lambda:g}; OT: topk={cli.ot_topk}, visual_tokens={cli.ot_visual_tokens}.", "Each VISTA/OT pair shares the same seed-specific 500-image subset.", "", "| Gamma | Seeds | VISTA F1 | OT F1 | OT−VISTA F1 | VISTA CHAIRs | OT CHAIRs | OT−VISTA CHAIRs |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for gamma in sorted({row["gamma"] for row in rows}):
        group = [row for row in rows if row["gamma"] == gamma]
        vals = {key: mean_std([row[key] for row in group])[0] for key in ("vista_F1", "ot_F1", "delta_F1", "vista_CHAIRs", "ot_CHAIRs", "delta_CHAIRs")}
        lines.append(f"| {gamma:g} | {len(group)} | {vals['vista_F1']:.4f} | {vals['ot_F1']:.4f} | {vals['delta_F1']:+.4f} | {vals['vista_CHAIRs']:.4f} | {vals['ot_CHAIRs']:.4f} | {vals['delta_CHAIRs']:+.4f} |")
    cli.markdown.parent.mkdir(parents=True, exist_ok=True)
    cli.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {cli.csv}")
    print(f"Wrote {cli.markdown}")


if __name__ == "__main__":
    main()
