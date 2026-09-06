#!/usr/bin/env python3
"""Make a deterministic COCO held-out pool, excluding a prior experiment."""
import argparse
from pathlib import Path

from make_chair_seed_manifest import coco_image_id


def ids_from_path(path):
    path = Path(path)
    values = set()
    for candidate in list(path.rglob("*.txt")) + list(path.rglob("generation.jsonl")):
        try:
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if candidate.name.endswith(".txt"):
                    if line and not line.startswith("#"):
                        values.add(int(line))
                elif line:
                    import json
                    values.add(int(json.loads(line)["sample_id"]))
        except (ValueError, KeyError, UnicodeDecodeError):
            continue
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--old-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260907)
    args = p.parse_args()
    old = ids_from_path(args.old_run)
    if not old:
        raise RuntimeError(f"Could not recover old image IDs under {args.old_run}")
    available = sorted(coco_image_id(item) for item in args.data_path.glob("*.jpg"))
    import numpy as np
    candidate = np.asarray([item for item in available if item not in old], dtype=int)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(candidate)
    if set(candidate) & old:
        raise AssertionError("old/new image intersection is non-zero")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "old_image_ids.txt").write_text("".join(f"{item}\n" for item in sorted(old)))
    (args.output_dir / "new_candidate_image_ids.txt").write_text("".join(f"{item}\n" for item in candidate))
    (args.output_dir / "heldout_manifest_audit.json").write_text(
        '{\n  "old_image_count": %d,\n  "candidate_image_count": %d,\n  "intersection_size": 0,\n  "selection_seed": %d\n}\n' % (len(old), len(candidate), args.seed)
    )
    print(f"old={len(old)} candidate={len(candidate)} overlap=0")


if __name__ == "__main__":
    main()
