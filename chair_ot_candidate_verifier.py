#!/usr/bin/env python3
"""Collect candidate-conditioned local-OT verification features on LLaVA."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import myutils
from eval_data_loader import COCODataSet
from llava.constants import IMAGE_TOKEN_INDEX
from llava.utils import disable_torch_init
from llm_layers import add_vsv_layers, remove_vsv_layers
from model_loader import ModelLoader
from ot_bary_sla import OTBarySLA
from ot_candidate_verifier import (
    candidate_ot_features,
    candidate_token_span,
    candidate_uot_features,
)
from steering_vector import obtain_vsv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-1.5")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--work-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--vsv", action="store_true")
    parser.add_argument("--vsv-lambda", type=float, default=0.17)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--logits-layers", default="25,30")
    parser.add_argument("--region-topks", default="8,16,32")
    parser.add_argument("--ot-sinkhorn-iters", type=int, default=50)
    parser.add_argument("--ot-sinkhorn-tolerance", type=float, default=0.001)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-layer-temperature", type=float, default=0.06)
    parser.add_argument("--ot-attention-power", type=float, default=0.75)
    parser.add_argument("--ot-attention-uniform-mix", type=float, default=0.02)
    parser.add_argument(
        "--uot-marginal-relaxations", default="",
        help="Comma-separated UOT marginal KL strengths; empty preserves v1 scoring.",
    )
    parser.add_argument(
        "--counterfactual-noise-std", type=float, default=0.0,
        help="Optional normalized-pixel Gaussian noise for a sequential contrast pass.",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_existing(path, config_id):
    if not path.exists():
        return set()
    valid_rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                valid_rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Ignoring truncated output line {path}:{line_number}")
    # Sanitize a possible interrupted final write before resuming.
    with path.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row) + "\n")
    return {
        row["work_id"] for row in valid_rows
        if row.get("verifier_config_id") == config_id
    }


def verifier_config_id(args, layer_indices, region_topks, uot_relaxations):
    payload = {
        "model": args.model,
        "vsv": args.vsv,
        "vsv_lambda": args.vsv_lambda,
        "layers": args.layers,
        "logits_layers": layer_indices,
        "region_topks": region_topks,
        "epsilon": args.ot_epsilon,
        "sinkhorn_iters": args.ot_sinkhorn_iters,
        "sinkhorn_tolerance": args.ot_sinkhorn_tolerance,
        "layer_temperature": args.ot_layer_temperature,
        "attention_power": args.ot_attention_power,
        "uniform_mix": args.ot_attention_uniform_mix,
    }
    # Keep the exact v1 hash so completed legacy shards remain reusable.
    if uot_relaxations:
        payload["uot_marginal_relaxations"] = uot_relaxations
        payload["counterfactual_noise_std"] = args.counterfactual_noise_std
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_query(candidate):
    return (
        "Inspect the image only to verify the proposed object. "
        f"Candidate object: [ {candidate} ]. "
        "Check whether this exact object is visibly supported."
    )


def install_visual_cache(model, args):
    if hasattr(model, "use_ot_bary_sla") or hasattr(model, "ot_bary_sla"):
        raise RuntimeError("OT state leaked from an earlier verifier forward")
    model.use_ot_bary_sla = True
    model.ot_bary_sla = OTBarySLA(
        topk=1,
        epsilon=args.ot_epsilon,
        sinkhorn_iters=args.ot_sinkhorn_iters,
        sinkhorn_tolerance=args.ot_sinkhorn_tolerance,
        layer_temperature=args.ot_layer_temperature,
        attention_visual_marginal=True,
        attention_power=args.ot_attention_power,
        attention_uniform_mix=args.ot_attention_uniform_mix,
    )


def remove_visual_cache(model):
    model.ot_bary_sla.clear()
    del model.ot_bary_sla
    del model.use_ot_bary_sla


def _candidate_tensors(model_loader, template, image, row, args, layer_indices):
    query = build_query(row["phrase"])
    _questions, kwargs = model_loader.prepare_inputs_for_model(template, [query], image)
    original_ids = kwargs["input_ids"][0]
    original_start, original_end, candidate_ids = candidate_token_span(
        model_loader.tokenizer, original_ids, row["phrase"],
        image_token_index=IMAGE_TOKEN_INDEX,
    )
    install_visual_cache(model_loader.llm_model, args)
    try:
        outputs = model_loader.llm_model(
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        mask = model_loader.llm_model.ot_bary_sla._visual_attention_positions[0]
        visual_count = int(mask.sum().item())
        shift = visual_count - 1
        query_positions = torch.arange(
            original_start + shift,
            original_end + shift,
            device=mask.device,
        )
        if query_positions[-1].item() >= outputs.hidden_states[0].shape[1]:
            raise RuntimeError("Expanded candidate token position exceeds sequence length")

        layer_visual = []
        layer_attention = []
        for layer_index in layer_indices:
            hidden = outputs.hidden_states[layer_index + 1][0]
            attention = outputs.attentions[layer_index][0].float().mean(dim=0)
            layer_visual.append(hidden[mask])
            layer_attention.append(attention[query_positions][:, mask])
        layer_visual = torch.stack(layer_visual)
        layer_attention = torch.stack(layer_attention)
        token_features = model_loader.llm_model.lm_head.weight[candidate_ids]
        token_strings = model_loader.tokenizer.convert_ids_to_tokens(candidate_ids.tolist())
        return {
            "candidate_token_ids": candidate_ids.detach().cpu().tolist(),
            "candidate_tokens": token_strings,
            "visual_tokens": visual_count,
            "layers": layer_indices,
            "layer_visual": layer_visual,
            "layer_attention": layer_attention,
            "token_features": token_features,
        }
    finally:
        remove_visual_cache(model_loader.llm_model)


def _distorted_image(image, noise_std, work_id):
    if noise_std <= 0:
        return image
    base_seed = int(work_id[:16], 16) % (2**63 - 1)

    def distort_pixels(value, offset=0):
        """Preserve BatchFeature/DataLoader nesting around pixel tensors."""
        if torch.is_tensor(value):
            generator = torch.Generator(device=value.device)
            generator.manual_seed((base_seed + offset) % (2**63 - 1))
            noise = torch.randn(
                value.shape, generator=generator, device=value.device,
                dtype=value.dtype,
            )
            reduce_dims = tuple(range(max(0, value.ndim - 2), value.ndim))
            lower = value.amin(dim=reduce_dims, keepdim=True)
            upper = value.amax(dim=reduce_dims, keepdim=True)
            return (value + noise_std * noise).clamp(lower, upper)
        if isinstance(value, list):
            return [distort_pixels(item, offset + index + 1)
                    for index, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(distort_pixels(item, offset + index + 1)
                         for index, item in enumerate(value))
        raise TypeError(
            "pixel_values must be a tensor or a list/tuple of tensors; got "
            f"{type(value).__name__}"
        )

    distorted = dict(image)
    distorted["pixel_values"] = distort_pixels(image["pixel_values"])
    return distorted


def score_candidate(
    model_loader, template, image, row, args, layer_indices, region_topks,
    uot_relaxations,
):
    clean = _candidate_tensors(
        model_loader, template, image, row, args, layer_indices,
    )
    features = candidate_ot_features(
        layer_visual=clean["layer_visual"],
        layer_attention=clean["layer_attention"],
        token_features=clean["token_features"],
        token_strings=clean["candidate_tokens"],
        region_topks=region_topks,
        attention_power=args.ot_attention_power,
        uniform_mix=args.ot_attention_uniform_mix,
        epsilon=args.ot_epsilon,
        sinkhorn_iters=args.ot_sinkhorn_iters,
        sinkhorn_tolerance=args.ot_sinkhorn_tolerance,
        layer_temperature=args.ot_layer_temperature,
    )
    result = {
        **row,
        "candidate_token_ids": clean["candidate_token_ids"],
        "candidate_tokens": clean["candidate_tokens"],
        "visual_tokens": clean["visual_tokens"],
        "layers": clean["layers"],
        **features,
    }
    if uot_relaxations:
        uot_kwargs = {
            "region_topks": region_topks,
            "marginal_relaxations": uot_relaxations,
            "attention_power": args.ot_attention_power,
            "uniform_mix": args.ot_attention_uniform_mix,
            "epsilon": args.ot_epsilon,
            "sinkhorn_iters": args.ot_sinkhorn_iters,
            "sinkhorn_tolerance": args.ot_sinkhorn_tolerance,
        }
        result["uot"] = candidate_uot_features(
            clean["layer_visual"], clean["layer_attention"],
            clean["token_features"], clean["candidate_tokens"], **uot_kwargs,
        )
        if args.counterfactual_noise_std > 0:
            noisy_image = _distorted_image(
                image, args.counterfactual_noise_std, row["work_id"],
            )
            noisy = _candidate_tensors(
                model_loader, template, noisy_image, row, args, layer_indices,
            )
            result["uot_counterfactual"] = candidate_uot_features(
                noisy["layer_visual"], noisy["layer_attention"],
                noisy["token_features"], noisy["candidate_tokens"], **uot_kwargs,
            )
    return result


def main():
    args = parse_args()
    if args.model != "llava-1.5":
        raise ValueError("Candidate verifier currently supports llava-1.5 only")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    start_layer, end_layer = (int(x) for x in args.logits_layers.split(","))
    if start_layer > end_layer:
        raise ValueError("logits-layers must be an inclusive increasing range")
    layer_indices = list(range(start_layer, end_layer + 1))
    region_topks = [int(x) for x in args.region_topks.split(",")]
    if any(x <= 0 for x in region_topks):
        raise ValueError("region-topks must all be positive")
    uot_relaxations = [
        float(value) for value in args.uot_marginal_relaxations.split(",")
        if value.strip()
    ]
    if any(value <= 0 for value in uot_relaxations):
        raise ValueError("uot-marginal-relaxations must all be positive")
    if args.counterfactual_noise_std < 0:
        raise ValueError("counterfactual-noise-std must be non-negative")
    if args.counterfactual_noise_std > 0 and not uot_relaxations:
        raise ValueError("counterfactual scoring requires UOT relaxations")

    config_id = verifier_config_id(
        args, layer_indices, region_topks, uot_relaxations,
    )
    all_rows = read_jsonl(args.work_manifest)
    assigned = [
        row for index, row in enumerate(all_rows)
        if index % args.num_shards == args.shard_index
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_existing(args.output, config_id)
    args.output.touch(exist_ok=True)
    pending = [row for row in assigned if row["work_id"] not in completed]
    if not pending:
        print(f"Shard {args.shard_index}: all {len(assigned)} candidates already scored")
        return

    myutils.seed_everything(args.seed + args.shard_index)
    disable_torch_init()
    model_loader = ModelLoader(args.model)
    template = myutils.prepare_template(args)
    grouped = defaultdict(list)
    for row in pending:
        grouped[int(row["image_id"])].append(row)
    dataset = COCODataSet(
        data_path=args.data_path,
        trans=model_loader.image_processor,
        image_ids=list(grouped),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    with args.output.open("a", encoding="utf-8") as output_handle:
        for data in tqdm(loader, total=len(loader), desc=f"shard {args.shard_index}"):
            image_id = int(data["img_id"][0])
            image = data["image"]
            visual_installed = False
            try:
                with torch.inference_mode(), myutils.maybe_autocast(
                    args.model, model_loader.vlm_model.device,
                ):
                    if args.vsv:
                        questions, base_kwargs = model_loader.prepare_inputs_for_model(
                            template,
                            ["Please help me describe the image in detail."],
                            image,
                        )
                        neg_kwargs = model_loader.prepare_neg_prompt(args, questions, template=template)
                        pos_kwargs = model_loader.prepare_pos_prompt(args, base_kwargs)
                        visual_vector, _ = obtain_vsv(
                            args, model_loader.llm_model,
                            [[neg_kwargs, pos_kwargs]], rank=1,
                        )
                        add_vsv_layers(
                            model_loader.llm_model,
                            torch.stack([visual_vector], dim=1).cuda(),
                            [args.vsv_lambda], args.layers,
                        )
                        visual_installed = True
                    for row in grouped[image_id]:
                        scored = score_candidate(
                            model_loader, template, image, row, args,
                            layer_indices, region_topks, uot_relaxations,
                        )
                        scored["verifier_config_id"] = config_id
                        output_handle.write(json.dumps(scored) + "\n")
                        output_handle.flush()
            finally:
                if visual_installed:
                    remove_vsv_layers(model_loader.llm_model)

    final_completed = read_existing(args.output, config_id)
    missing = [row["work_id"] for row in assigned if row["work_id"] not in final_completed]
    if missing:
        raise RuntimeError(f"Shard {args.shard_index} is missing {len(missing)} scores")
    print(f"Shard {args.shard_index}: complete ({len(assigned)} candidates)")


if __name__ == "__main__":
    main()
