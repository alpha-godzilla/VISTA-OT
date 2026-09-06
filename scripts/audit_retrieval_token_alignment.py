#!/usr/bin/env python3
"""Print the exact causal-token alignment used by retrieval traces."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    with np.load(args.trace, allow_pickle=False) as data:
        ids, text = data["token_ids"], data["token_text"]
        print("timestep\tq input / appended KV\tpredicted next token")
        for t in range(min(args.limit, len(ids))):
            source = "<prefill final prompt>" if t == 0 else repr(str(text[t - 1]))
            print(f"{t}\t{source}\t{int(ids[t])}\t{str(text[t])!r}")
    print("Invariant: generated token index j is produced by q timestep j.")


if __name__ == "__main__":
    main()
