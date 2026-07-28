#!/usr/bin/env python3
"""Export an ordered, validated COCO image-ID manifest from CHAIR JSONL."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_jsonl", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--expected-count", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_result_image_ids(path):
    image_ids = []
    seen = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
                image_id = int(record["image_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid CHAIR record at {path}:{line_number}"
                ) from exc
            if image_id in seen:
                raise ValueError(
                    f"Duplicate image_id at {path}:{line_number}: {image_id}"
                )
            seen.add(image_id)
            image_ids.append(image_id)
    return image_ids


def main():
    args = parse_args()
    image_ids = read_result_image_ids(args.result_jsonl)
    if args.expected_count > 0 and len(image_ids) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} unique image IDs, "
            f"found {len(image_ids)}"
        )
    if args.output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_file}; "
            "pass --overwrite to replace it"
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        "".join(f"{image_id}\n" for image_id in image_ids),
        encoding="utf-8",
    )
    print(f"Wrote {len(image_ids)} image IDs to {args.output_file}")


if __name__ == "__main__":
    main()
