#!/usr/bin/env python3
"""Build generic VISTA-only candidate work items from completed stage one."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ot_candidate_verifier import vista_only_candidates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1994, 2024, 3407])
    parser.add_argument("--vista-method", default="vista")
    parser.add_argument("--vista-setting", default="original")
    parser.add_argument("--ot-method", default="recall_recovery")
    parser.add_argument("--ot-setting", default="rho0.25_k32")
    parser.add_argument("--max-candidates", type=int, default=6)
    return parser.parse_args()


def load_jsonl(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["image_id"])
            if image_id in rows:
                raise ValueError(f"Duplicate image_id={image_id} in {path}:{line_number}")
            rows[image_id] = str(row["caption"]).strip()
    return rows


def work_id(row):
    payload = json.dumps(row, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def main():
    args = parse_args()
    wanted = set(args.seeds)
    paired = {seed: {} for seed in args.seeds}
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            seed = int(row["seed"])
            if seed not in wanted:
                continue
            if row["method"] == args.vista_method and row["setting"] == args.vista_setting:
                paired[seed]["vista"] = row
            if row["method"] == args.ot_method and row["setting"] == args.ot_setting:
                paired[seed]["ot"] = row

    output_rows = []
    per_seed = {}
    for seed in args.seeds:
        if set(paired[seed]) != {"vista", "ot"}:
            raise ValueError(
                f"Missing stage-one pair for seed={seed}; found {sorted(paired[seed])}"
            )
        vista_row, ot_row = paired[seed]["vista"], paired[seed]["ot"]
        vista = load_jsonl(vista_row["result_jsonl"])
        ot = load_jsonl(ot_row["result_jsonl"])
        if set(vista) != set(ot):
            raise ValueError(f"VISTA/OT image sets differ for seed={seed}")
        candidate_count = image_count = 0
        for image_id in vista:
            candidates = vista_only_candidates(
                vista[image_id], ot[image_id], max_candidates=args.max_candidates,
            )
            image_count += bool(candidates)
            for candidate in candidates:
                row = {
                    "seed": seed,
                    "image_id": image_id,
                    **candidate.to_dict(),
                    "vista_caption": vista[image_id],
                    "ot_caption": ot[image_id],
                }
                row["work_id"] = work_id(row)
                output_rows.append(row)
                candidate_count += 1
        per_seed[seed] = {
            "images": len(vista),
            "images_with_candidates": image_count,
            "candidates": candidate_count,
        }

    output_rows.sort(key=lambda row: (row["seed"], row["image_id"], row["phrase"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"total_candidates": len(output_rows), "per_seed": per_seed}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
