#!/usr/bin/env python3
"""Measure the recall ceiling of VISTA-only proposals over an OT caption."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def sentence_map(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(row["image_id"]): row for row in payload["sentences"]}


def main():
    args = parse_args()
    per_seed = {}
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            seed = int(row["seed"])
            if row["method"] == "vista":
                per_seed.setdefault(seed, {})["vista"] = Path(row["chair_json"])
            elif row["method"] == "ot_stage1":
                per_seed.setdefault(seed, {})["ot"] = Path(row["chair_json"])

    rows = []
    for seed, paths in sorted(per_seed.items()):
        if set(paths) != {"vista", "ot"}:
            raise ValueError(f"Missing VISTA/OT stage-one pair for seed={seed}")
        vista, ot = sentence_map(paths["vista"]), sentence_map(paths["ot"])
        if set(vista) != set(ot):
            raise ValueError(f"Stage-one image sets differ for seed={seed}")
        gt_total = proposal_total = true_total = false_total = 0
        images_with_true_extra = 0
        for image_id in vista:
            gt = set(vista[image_id]["mscoco_gt_words"])
            vista_objects = set(vista[image_id]["mscoco_generated_words"])
            ot_objects = set(ot[image_id]["mscoco_generated_words"])
            proposals = vista_objects - ot_objects
            true_extra = proposals & gt
            false_extra = proposals - gt
            gt_total += len(gt)
            proposal_total += len(proposals)
            true_total += len(true_extra)
            false_total += len(false_extra)
            images_with_true_extra += bool(true_extra)
        rows.append({
            "seed": seed,
            "images": len(vista),
            "gt_objects": gt_total,
            "vista_only_proposals": proposal_total,
            "true_extra_objects": true_total,
            "false_extra_objects": false_total,
            "recoverable_recall": true_total / gt_total if gt_total else 0.0,
            "proposal_precision": true_total / proposal_total if proposal_total else 0.0,
            "images_with_true_extra": images_with_true_extra,
        })

    if not rows:
        raise ValueError("No paired stage-one rows found")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    recoverable = np.array([row["recoverable_recall"] for row in rows])
    precision = np.array([row["proposal_precision"] for row in rows])
    lines = [
        "# VISTA-only proposal ceiling",
        "",
        "Ground-truth labels are used only for this post-hoc diagnostic, not by generation.",
        "",
        "| Seed | VISTA-only | True extra | False extra | Recoverable Recall | Proposal Precision |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['vista_only_proposals']} | "
            f"{row['true_extra_objects']} | {row['false_extra_objects']} | "
            f"{row['recoverable_recall']:.4f} | "
            f"{row['proposal_precision']:.4f} |"
        )
    lines.extend([
        "",
        f"Mean recoverable Recall: **{recoverable.mean():.4f}**",
        f"Mean proposal precision: **{precision.mean():.4f}**",
    ])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
