#!/usr/bin/env python3
"""Return success only for a complete, parseable CHAIR evaluation JSON."""

import argparse
import json
from pathlib import Path


REQUIRED_METRICS = {"CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len"}


def is_valid_chair_output(path, expected_images):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    metrics = payload.get("overall_metrics")
    sentences = payload.get("sentences")
    return (
        isinstance(metrics, dict)
        and REQUIRED_METRICS <= set(metrics)
        and isinstance(sentences, list)
        and len(sentences) == expected_images
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    args = parser.parse_args()
    if args.expected_images <= 0:
        parser.error("--expected-images must be positive")
    raise SystemExit(0 if is_valid_chair_output(args.path, args.expected_images) else 1)


if __name__ == "__main__":
    main()
