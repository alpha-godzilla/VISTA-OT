#!/usr/bin/env python3
"""Fixed-token incremental replay using LLaVA's ordinary cached forward path.

The first multimodal prefill predicts saved token 0.  Thereafter each cached
forward consumes saved token t-1 and predicts saved token t.  It never calls
``generate`` and never chooses a replacement token.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import myutils
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from visual_memory_retrieval import RetrievalShiftTracer


def records(root):
    result = {}
    for path in Path(root).rglob("generation.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                result[int(row["sample_id"])] = row
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generation-root", type=Path, required=True)
    p.add_argument("--sample-ids", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    wanted = [int(line) for line in args.sample_ids.read_text().splitlines() if line.strip()]
    if args.limit:
        wanted = wanted[:args.limit]
    saved = records(args.generation_root)
    missing = sorted(set(wanted) - set(saved))
    if missing:
        raise RuntimeError(f"Missing generation records for {missing[:10]}")
    disable_torch_init()
    loader = ModelLoader("llava-1.5")
    model = loader.llm_model
    tracer = RetrievalShiftTracer(model, loader.tokenizer, args.output_dir, compact=args.compact)
    tracer.install(); model.retrieval_shift_tracer = tracer
    template = myutils.prepare_template(argparse.Namespace(model="llava-1.5"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_file = (args.output_dir / "generation.jsonl").open("w", encoding="utf-8")
    audit_file = (args.output_dir / "replay_audit.jsonl").open("w", encoding="utf-8")
    try:
        with torch.inference_mode():
            for sample_id in wanted:
                row = saved[sample_id]
                token_ids = [int(item) for item in row["generated_token_ids"]]
                if not token_ids:
                    continue
                image_path = args.data_path / f"COCO_val2014_{sample_id:012d}.jpg"
                image = loader.image_processor(Image.open(image_path).convert("RGB"))
                question = [row.get("prompt", "Please help me describe the image in detail.")]
                tracer.start_sample(sample_id, prompt=question[0])
                _, kwargs = loader.prepare_inputs_for_model(template, question, image)
                output = model(**kwargs, use_cache=True, output_attentions=False, return_dict=True)
                initial_argmax = int(output.logits[0, -1].argmax())
                past = output.past_key_values
                for index in range(1, len(token_ids)):
                    consumed = torch.tensor([[token_ids[index - 1]]], dtype=torch.long, device=output.logits.device)
                    output = model(input_ids=consumed, past_key_values=past, use_cache=True,
                                   output_attentions=False, return_dict=True)
                    past = output.past_key_values
                path = tracer.finish_sample(token_ids, row["generated_text"])
                generated_file.write(json.dumps(row) + "\n"); generated_file.flush()
                audit_file.write(json.dumps({"sample_id": sample_id, "trace": str(path),
                    "tokens": len(token_ids), "first_token_matches_greedy": initial_argmax == token_ids[0],
                    "first_token_expected": token_ids[0], "first_token_argmax": initial_argmax}) + "\n"); audit_file.flush()
    finally:
        tracer.remove(); del model.retrieval_shift_tracer
        generated_file.close(); audit_file.close()


if __name__ == "__main__":
    main()
