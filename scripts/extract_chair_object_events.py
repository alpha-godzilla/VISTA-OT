#!/usr/bin/env python3
"""Build strictly aligned CHAIR object events from retrieval-trace captions.

No model inference occurs here.  An event is retained only when (1) CHAIR's
object extractor can provide an exact source character span, (2) that span is
covered by an exact generated-token span, and (3) the trace token IDs equal
the token IDs recorded during generation.  All other captions are written to
an audit file rather than approximated.
"""
import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from chair_ans import CHAIR
from vista_paths import COCO_ANNOTATIONS_PATH, LLAVA_MODEL_PATH


EVENT_FIELDS = (
    "event_id", "sample_id", "caption", "object_text", "object_word",
    "object_category", "event_type", "char_start", "char_end",
    "object_token_start", "object_token_end", "first_query_timestep",
    "last_query_timestep", "decision_query_timestep", "event_order",
    "caption_token_count",
)
AUDIT_FIELDS = (
    "sample_id", "generated_token_index", "generated_token_id",
    "decoded_token_piece", "char_start", "char_end", "input_to_decode_step",
    "query_timestep", "predicted_next_token", "kv_appended_token",
)


def trace_paths(trace_root):
    paths = {}
    for path in Path(trace_root).rglob("sample_*.npz"):
        sample_id = int(path.stem.removeprefix("sample_"))
        if sample_id in paths:
            raise ValueError(f"Duplicate trace for sample {sample_id}")
        paths[sample_id] = path
    return paths


def generation_records(trace_root):
    records = {}
    for path in Path(trace_root).rglob("generation.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = int(row["sample_id"])
            if sample_id in records:
                raise ValueError(f"Duplicate generation record for sample {sample_id}")
            records[sample_id] = row
    if not records:
        raise FileNotFoundError(f"No generation.jsonl files under {trace_root}")
    return records


def exact_token_char_spans(tokenizer, token_ids, caption):
    """Return generated-token character intervals, or a strict failure reason."""
    raw = tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    trimmed = raw.strip()
    if trimmed != caption:
        return None, (
            "decoded generated IDs do not exactly equal saved caption after "
            f"boundary stripping (decoded={trimmed!r}, caption={caption!r})"
        )
    visible_end = len(raw) - trailing
    spans = []
    before = ""
    for index, token_id in enumerate(token_ids):
        after = tokenizer.decode(
            token_ids[:index + 1], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not after.startswith(before):
            return None, f"non-prefix tokenizer decode transition at token {index}"
        start_raw, end_raw = len(before), len(after)
        # Intersect the token's visible contribution with the stripped caption.
        start = max(start_raw, leading) - leading
        end = min(end_raw, visible_end) - leading
        start, end = max(start, 0), max(end, 0)
        spans.append((start, end))
        before = after
    if before != raw:
        return None, "token-prefix reconstruction does not equal full decode"
    return spans, None


def object_token_span(char_start, char_end, token_spans):
    indices = [
        index for index, (start, end) in enumerate(token_spans)
        if end > char_start and start < char_end
    ]
    if not indices:
        return None
    return min(indices), max(indices) + 1


def load_evaluator(cache, coco_path):
    if cache and Path(cache).is_file():
        with open(cache, "rb") as handle:
            return pickle.load(handle)
    return CHAIR(coco_path)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=os.environ.get("VISTA_LLAVA_MODEL_PATH", LLAVA_MODEL_PATH))
    parser.add_argument("--chair-cache", default=None)
    parser.add_argument("--coco-path", default=COCO_ANNOTATIONS_PATH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    evaluator = load_evaluator(args.chair_cache, args.coco_path)
    traces, generations = trace_paths(args.trace_root), generation_records(args.trace_root)
    if set(traces) != set(generations):
        missing_trace, missing_generation = sorted(set(generations) - set(traces)), sorted(set(traces) - set(generations))
        raise RuntimeError(f"Trace/generation sample mismatch; no trace={missing_trace[:5]}, no generation={missing_generation[:5]}")

    all_events, aligned_events, audit_rows, failures = [], [], [], []
    event_id = 0
    for sample_id in sorted(generations):
        generated = generations[sample_id]
        caption = generated["generated_text"]
        token_ids = [int(token) for token in generated["generated_token_ids"]]
        with np.load(traces[sample_id], allow_pickle=False) as trace:
            trace_ids = trace["token_ids"].astype(np.int64).tolist()
        if trace_ids != token_ids:
            failures.append({"sample_id": sample_id, "reason": "trace_token_ids_mismatch"})
            continue
        token_spans, reason = exact_token_char_spans(tokenizer, token_ids, caption)
        if reason is not None:
            failures.append({"sample_id": sample_id, "reason": reason})
            continue
        try:
            mentions = evaluator.caption_to_object_mentions(caption)
        except Exception as error:  # Intentionally visible in the audit output.
            failures.append({"sample_id": sample_id, "reason": f"chair_span_error: {error}"})
            continue
        for index, token_id in enumerate(token_ids):
            start, end = token_spans[index]
            audit_rows.append({
                "sample_id": sample_id,
                "generated_token_index": index,
                "generated_token_id": token_id,
                "decoded_token_piece": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                "char_start": start,
                "char_end": end,
                "input_to_decode_step": "<prefill final prompt>" if index == 0 else str(token_ids[index - 1]),
                "query_timestep": index,
                "predicted_next_token": token_id,
                "kv_appended_token": "" if index == 0 else str(token_ids[index - 1]),
            })
        gt_objects = evaluator.imid_to_objects[sample_id]
        for order, mention in enumerate(mentions):
            span = object_token_span(mention["char_start"], mention["char_end"], token_spans)
            row = {
                "event_id": event_id,
                "sample_id": sample_id,
                "caption": caption,
                "object_text": mention["object_text"],
                "object_word": mention["word"],
                "object_category": mention["object_category"],
                "event_type": "grounded" if mention["object_category"] in gt_objects else "hallucinated",
                "char_start": mention["char_start"], "char_end": mention["char_end"],
                "object_token_start": "", "object_token_end": "",
                "first_query_timestep": "", "last_query_timestep": "",
                "decision_query_timestep": "", "event_order": order,
                "caption_token_count": len(token_ids),
            }
            event_id += 1
            all_events.append(row)
            if span is None:
                failures.append({"sample_id": sample_id, "reason": f"object_no_token_overlap: {mention['object_text']!r}"})
                continue
            first, end = span
            row.update({
                "object_token_start": first, "object_token_end": end,
                "first_query_timestep": first, "last_query_timestep": end - 1,
                "decision_query_timestep": first,
            })
            aligned_events.append(row)

    write_csv(args.output_dir / "chair_events.csv", EVENT_FIELDS, all_events)
    write_csv(args.output_dir / "chair_events_token_aligned.csv", EVENT_FIELDS, aligned_events)
    write_csv(args.output_dir / "token_query_alignment_audit.csv", AUDIT_FIELDS, audit_rows)
    write_csv(args.output_dir / "chair_alignment_failures.csv", ("sample_id", "reason"), failures)
    print(json.dumps({
        "captions": len(generations), "object_events": len(all_events),
        "token_aligned_events": len(aligned_events), "alignment_failures": len(failures),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
