#!/usr/bin/env python3
"""Benchmark original VISTA SLA against OT-BarySLA on one COCO image."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=os.path.join(
            os.environ.get(
                "VISTA_COCO_ROOT",
                "/data/sun_yuxi/datasets/coco",
            ),
            "val2014",
        ),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("original", "ot", "uniform"),
        default=("original", "ot"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--vsv-lambda", type=float, default=0.17)
    parser.add_argument("--logits-layers", default="25,30")
    parser.add_argument("--logits-alpha", type=float, default=0.3)
    parser.add_argument("--ot-topk", type=int, default=8)
    parser.add_argument("--ot-visual-tokens", type=int, default=36)
    parser.add_argument("--ot-sinkhorn-iters", type=int, default=3)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    return parser.parse_args()


def mode_args(cli_args, mode):
    return SimpleNamespace(
        model="llava-1.5",
        vsv=True,
        vsv_lambda=cli_args.vsv_lambda,
        layers=None,
        logits_aug=True,
        logits_layers=cli_args.logits_layers,
        logits_alpha=cli_args.logits_alpha,
        use_ot_bary_sla=mode != "original",
        ot_topk=cli_args.ot_topk,
        ot_visual_tokens=cli_args.ot_visual_tokens,
        ot_sinkhorn_iters=cli_args.ot_sinkhorn_iters,
        ot_epsilon=cli_args.ot_epsilon,
        ot_log_stats=False,
        ot_force_uniform=mode == "uniform",
    )


def output_sequences(outputs):
    return outputs.sequences if hasattr(outputs, "sequences") else outputs


def main():
    args = parse_args()
    import myutils
    from eval_data_loader import COCODataSet
    from llava.utils import disable_torch_init
    from llm_layers import add_vsv_layers, remove_vsv_layers
    from model_loader import ModelLoader
    from steering_vector import (
        add_logits_flag,
        obtain_vsv,
        remove_logits_flag,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("The end-to-end benchmark requires a CUDA GPU")
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("--repeats must be positive and --warmup nonnegative")

    myutils.seed_everything(args.seed)
    disable_torch_init()
    loader = ModelLoader("llava-1.5")
    dataset = COCODataSet(args.data_path, trans=loader.image_processor)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f"sample index {args.sample_index} is outside dataset size {len(dataset)}"
        )
    batch = next(
        iter(
            DataLoader(
                Subset(dataset, [args.sample_index]),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        )
    )
    image = batch["image"]
    image_id = int(batch["img_id"][0])
    template = myutils.prepare_template(SimpleNamespace(model="llava-1.5"))
    query = ["Please help me describe the image in detail."]
    questions, kwargs = loader.prepare_inputs_for_model(
        template,
        query,
        image,
    )
    steering_args = mode_args(args, "original")
    negative = loader.prepare_neg_prompt(
        steering_args,
        questions,
        template=template,
    )
    positive = loader.prepare_pos_prompt(steering_args, kwargs)
    visual_vector, _ = obtain_vsv(
        steering_args,
        loader.llm_model,
        [[negative, positive]],
        rank=1,
    )
    visual_vector = torch.stack([visual_vector], dim=1).cuda()

    results = []
    for mode in args.modes:
        current_args = mode_args(args, mode)
        elapsed_values = []
        token_values = []
        memory_values = []

        for repeat in range(args.warmup + args.repeats):
            myutils.seed_everything(args.seed)
            add_vsv_layers(
                loader.llm_model,
                visual_vector,
                [args.vsv_lambda],
                None,
            )
            add_logits_flag(
                loader.llm_model,
                current_args,
                tokenizer=loader.tokenizer,
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = loader.llm_model.generate(
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    num_beams=1,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                    **kwargs,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_memory = torch.cuda.max_memory_allocated()
            remove_logits_flag(loader.llm_model)
            remove_vsv_layers(loader.llm_model)

            sequences = output_sequences(outputs)
            prompt_tokens = kwargs["input_ids"].shape[1]
            generated_tokens = max(1, sequences.shape[1] - prompt_tokens)
            if repeat >= args.warmup:
                elapsed_values.append(elapsed)
                token_values.append(generated_tokens)
                memory_values.append(peak_memory)

        mean_seconds = statistics.mean(elapsed_values)
        mean_tokens = statistics.mean(token_values)
        row = {
            "mode": mode,
            "seconds": mean_seconds,
            "generated_tokens": mean_tokens,
            "ms_per_token": 1000.0 * mean_seconds / mean_tokens,
            "tokens_per_second": mean_tokens / mean_seconds,
            "peak_memory_mib": statistics.mean(memory_values) / (1024**2),
        }
        results.append(row)

    original = next(
        (row for row in results if row["mode"] == "original"),
        None,
    )
    if original is not None:
        for row in results:
            row["latency_overhead"] = (
                row["ms_per_token"] / original["ms_per_token"] - 1.0
            )

    report = {
        "image_id": image_id,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "config": {
            "vsv_lambda": args.vsv_lambda,
            "logits_layers": args.logits_layers,
            "logits_alpha": args.logits_alpha,
            "ot_topk": args.ot_topk,
            "ot_visual_tokens": args.ot_visual_tokens,
            "ot_sinkhorn_iters": args.ot_sinkhorn_iters,
            "ot_epsilon": args.ot_epsilon,
        },
        "results": results,
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
