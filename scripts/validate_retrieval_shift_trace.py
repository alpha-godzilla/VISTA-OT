#!/usr/bin/env python3
"""Report numerical checks for one trace or a recursively sharded trace root."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.trace_root.rglob("sample_*.npz"))
    if not files:
        raise FileNotFoundError(f"No sample trace files in {args.trace_root}")
    maxima = {"L_GV_identity_error": 0.0, "reconstruction_error": 0.0, "mass_ratio_decomposition_error": 0.0, "query_rope_decomposition_error": 0.0}
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            for metric in maxima:
                valid = metric.replace("_error", "_valid")
                values = data[metric]
                if valid in data:
                    values = values[data[valid]]
                if values.size:
                    maxima[metric] = max(maxima[metric], float(np.abs(values).max()))
    print(f"samples: {len(files)}")
    for metric, value in maxima.items():
        print(f"max_abs_{metric}: {value:.8g}")


if __name__ == "__main__":
    main()
