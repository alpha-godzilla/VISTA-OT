#!/usr/bin/env python3
"""Summarize CHAIR sweep JSON files into CSV and Markdown."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    return parser.parse_args()


def read_rows(manifest):
    rows = []
    missing = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            chair_path = Path(entry["chair_json"])
            if not chair_path.is_file():
                missing.append(str(chair_path))
                continue
            with chair_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            metrics = payload["overall_metrics"]
            row = {
                "gamma": float(entry["gamma"]),
                "lambda": float(entry["lambda"]),
                "gpu": int(entry["gpu"]),
                "result_jsonl": entry["result_jsonl"],
                "chair_json": entry["chair_json"],
            }
            row.update({name: float(metrics[name]) for name in METRICS})
            rows.append(row)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing CHAIR result files:\n{formatted}")
    return sorted(rows, key=lambda row: (row["gamma"], row["lambda"]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("gamma", "lambda", *METRICS, "gpu", "result_jsonl", "chair_json")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return f"{value * 100:.2f}"


def write_markdown(path, rows):
    best_chairs = min(rows, key=lambda row: (row["CHAIRs"], row["CHAIRi"]))
    best_chairi = min(rows, key=lambda row: (row["CHAIRi"], row["CHAIRs"]))

    lines = [
        "# CHAIR Gamma/Lambda Sweep",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "Gamma is passed to `--logits-alpha`; lambda is passed to `--vsv-lambda`.",
        "CHAIRs, CHAIRi, Recall, Precision, and F1 are percentages; Len is the average caption length.",
        "",
        "## Best configurations",
        "",
        (
            f"- Lowest CHAIRs: gamma={best_chairs['gamma']:g}, "
            f"lambda={best_chairs['lambda']:g}, "
            f"CHAIRs={percent(best_chairs['CHAIRs'])}, "
            f"CHAIRi={percent(best_chairs['CHAIRi'])}"
        ),
        (
            f"- Lowest CHAIRi: gamma={best_chairi['gamma']:g}, "
            f"lambda={best_chairi['lambda']:g}, "
            f"CHAIRs={percent(best_chairi['CHAIRs'])}, "
            f"CHAIRi={percent(best_chairi['CHAIRi'])}"
        ),
        "",
        "## All results",
        "",
        "| Gamma | Lambda | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Precision ↑ | F1 ↑ | Len |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {gamma:g} | {lam:g} | {chairs} | {chairi} | {recall} | "
            "{precision} | {f1} | {length} |".format(
                gamma=row["gamma"],
                lam=row["lambda"],
                chairs=percent(row["CHAIRs"]),
                chairi=percent(row["CHAIRi"]),
                recall=percent(row["Recall"]),
                precision=percent(row["Precision"]),
                f1=percent(row["F1"]),
                length=percent(row["Len"]),
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = read_rows(args.manifest)
    if not rows:
        raise RuntimeError("The manifest contains no sweep entries.")
    write_csv(args.csv, rows)
    write_markdown(args.markdown, rows)
    print(f"Wrote {len(rows)} rows to {args.markdown} and {args.csv}")


if __name__ == "__main__":
    main()
