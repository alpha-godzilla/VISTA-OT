#!/usr/bin/env python3
"""Create a fixed CHAIR subset matching chair_eval.py's seeded selection."""

import argparse
import os
import random
from pathlib import Path

import numpy as np


def coco_image_id(image_file):
    return int(Path(image_file).stem[-12:])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--subset-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.subset_size <= 0:
        raise ValueError("subset-size must be positive")

    image_files = [
        name for name in os.listdir(args.data_path) if name.lower().endswith(".jpg")
    ]
    if args.subset_size > len(image_files):
        raise ValueError(
            f"subset-size={args.subset_size} exceeds {len(image_files)} available images"
        )

    # This is the ordering used by chair_eval.py before it samples its subset.
    random.seed(args.seed)
    np.random.seed(args.seed)
    random.shuffle(image_files)
    selected_indices = np.random.choice(
        len(image_files), args.subset_size, replace=False
    )
    image_ids = [coco_image_id(image_files[index]) for index in selected_indices]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{image_id}\n" for image_id in image_ids), encoding="utf-8"
    )
    print(f"Wrote {len(image_ids)} fixed image IDs to {args.output}")


if __name__ == "__main__":
    main()
