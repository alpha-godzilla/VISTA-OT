#!/usr/bin/env python3
"""Map a single-image attention-OT trace onto COCO ground-truth boxes."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def overlap_matrix(boxes, grid_size):
    """Return fractional patch coverage for normalized xywh boxes."""
    cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            cells.append((col / grid_size, row / grid_size, 1 / grid_size, 1 / grid_size))
    result = np.zeros((len(boxes), len(cells)), dtype=np.float64)
    for index, (x, y, width, height) in enumerate(boxes):
        x2, y2 = x + width, y + height
        for patch, (px, py, pw, ph) in enumerate(cells):
            ix = max(0.0, min(x2, px + pw) - max(x, px))
            iy = max(0.0, min(y2, py + ph) - max(y, py))
            result[index, patch] = ix * iy / (pw * ph)
    return result


def main():
    args = parse_args()
    trace_rows = [json.loads(line) for line in args.trace_jsonl.read_text().splitlines() if line.strip()]
    trace_row = next((row for row in trace_rows if row["image_id"] == args.image_id), None)
    if trace_row is None:
        raise ValueError(f"image_id={args.image_id} not found in {args.trace_jsonl}")
    trace = trace_row.get("attention_trace")
    if not trace:
        raise ValueError("Trace file has no attention_trace; run with --ot-attention-trace")

    effective = np.asarray([step["effective_source_marginal"][0] for step in trace])
    layer_weights = np.asarray([step["layer_weights"][0] for step in trace])
    patch_count = effective.shape[1]
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Expected a square visual patch grid; got {patch_count} patches")

    annotation_data = json.loads(args.annotations.read_text())
    image = next(item for item in annotation_data["images"] if item["id"] == args.image_id)
    categories = {item["id"]: item["name"] for item in annotation_data["categories"]}
    annotations = [item for item in annotation_data["annotations"] if item["image_id"] == args.image_id and item.get("iscrowd", 0) == 0]
    boxes = [
        (box[0] / image["width"], box[1] / image["height"], box[2] / image["width"], box[3] / image["height"])
        for annotation in annotations for box in [annotation["bbox"]]
    ]
    coverage = overlap_matrix(boxes, grid_size)
    mean_mass = effective.mean(axis=0)
    rows = []
    for annotation, patch_coverage in zip(annotations, coverage):
        uniform_mass = float(patch_coverage.mean())
        attention_mass = float(mean_mass @ patch_coverage)
        rows.append({
            "category": categories[annotation["category_id"]],
            "bbox_area_fraction": float(annotation["area"] / (image["width"] * image["height"])),
            "uniform_patch_mass": uniform_mass,
            "effective_attention_mass": attention_mass,
            "enrichment_vs_uniform": attention_mass / uniform_mass if uniform_mass else None,
        })
    rows.sort(key=lambda row: row["effective_attention_mass"])
    output = {
        "image_id": args.image_id,
        "image_size": [image["width"], image["height"]],
        "generated_steps": len(trace),
        "patch_grid": [grid_size, grid_size],
        "mean_layer_weights": layer_weights.mean(axis=0).tolist(),
        "objects": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        f"# Attention-OT coverage diagnosis: COCO {args.image_id}",
        "",
        f"Generated steps: {len(trace)}; visual patch grid: {grid_size}x{grid_size}.",
        "`effective attention mass` is the per-step attention marginal, combined using OT layer weights, then averaged over generated steps.",
        "`enrichment` compares that mass to a uniform distribution over patches; values below 1 indicate under-attention relative to box coverage.",
        "",
        "| Object | Box area | Uniform mass | Effective attention mass | Enrichment |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        enrichment = "n/a" if row["enrichment_vs_uniform"] is None else f"{row['enrichment_vs_uniform']:.3f}"
        lines.append(
            f"| {row['category']} | {row['bbox_area_fraction']:.3f} | {row['uniform_patch_mass']:.4f} | "
            f"{row['effective_attention_mass']:.4f} | {enrichment} |"
        )
    args.output_markdown.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
