#!/usr/bin/env python3
"""Fail closed if the previous retrieval run cannot define this replication."""
import argparse
import json
from pathlib import Path

PROMPT = "Please help me describe the image in detail."

def main():
    p = argparse.ArgumentParser(); p.add_argument("--old-trace-root", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, required=True); p.add_argument("--output", type=Path, required=True)
    args=p.parse_args(); traces=sorted(args.old_trace_root.rglob("sample_*.npz")); sidecars=sorted(args.old_trace_root.rglob("sample_*.json"))
    if len(traces) != 200 or len(sidecars) != 200:
        raise RuntimeError(f"Expected exactly 200 old traces/sidecars, got traces={len(traces)} sidecars={len(sidecars)}")
    prompts={json.loads(item.read_text())["prompt"] for item in sidecars}
    if prompts != {PROMPT}: raise RuntimeError(f"Old prompt mismatch: {prompts}")
    generation=[]
    for item in args.old_trace_root.rglob("generation.jsonl"):
        generation.extend(json.loads(line) for line in item.read_text().splitlines() if line.strip())
    if len(generation) != 200: raise RuntimeError(f"Expected 200 old generation records, got {len(generation)}")
    longest=max(len(row["generated_token_ids"]) for row in generation)
    if longest > args.max_new_tokens: raise RuntimeError("Current max_new_tokens is lower than the old observed generation length")
    payload={"passed":True,"old_samples":200,"prompt":PROMPT,"current_max_new_tokens":args.max_new_tokens,"old_max_observed_length":longest,
             "note":"The current runner reuses chair_eval.py greedy/no-beam/default-stop semantics; no VSV/OT flag is passed."}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)); print(args.output)
if __name__=="__main__": main()
