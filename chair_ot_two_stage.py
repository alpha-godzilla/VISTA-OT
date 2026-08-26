#!/usr/bin/env python3
"""Second-pass caption fusion with VISTA proposals and final OT decoding."""

import argparse
import json
from pathlib import Path

def load_proposals(path):
    proposals = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["image_id"])
            if image_id in proposals:
                raise ValueError(
                    f"Duplicate image_id={image_id} in {path}:{line_number}"
                )
            proposals[image_id] = str(row["caption"]).strip()
    if not proposals:
        raise ValueError(f"No proposals found in {path}")
    return proposals


def build_fusion_query(mode, vista_caption, ot_caption):
    common = (
        "Re-examine the image and write one accurate, detailed caption. "
        "Mention distinct objects only when they are visibly supported. "
        "Do not mention drafts or this instruction in the answer."
    )
    if mode == "ot_self":
        return (
            f"{common}\n\nA conservative draft is provided below. Preserve its "
            "supported content and add a missing object only when it is clear "
            f"in the image.\nConservative draft: {ot_caption}"
        )
    if mode == "vista_only":
        return (
            f"{common}\n\nA high-coverage draft is provided below. Verify every "
            "object against the image and omit unsupported content.\n"
            f"High-coverage draft: {vista_caption}"
        )
    if mode == "dual_fusion":
        return (
            f"{common}\n\nTwo drafts are provided as proposals. Use the "
            "conservative draft as the reliable core, and recover distinct "
            "objects from the high-coverage draft only when the image supports "
            f"them.\nConservative draft: {ot_caption}\n"
            f"High-coverage draft: {vista_caption}"
        )
    raise ValueError(f"Unknown fusion mode: {mode}")


def parse_args():
    import myutils

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-1.5")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--subset-ids-file", required=True)
    parser.add_argument("--vista-proposals", required=True)
    parser.add_argument("--ot-proposals", required=True)
    parser.add_argument(
        "--fusion-mode",
        choices=("ot_self", "vista_only", "dual_fusion"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vsv", action="store_true")
    parser.add_argument("--vsv-lambda", type=float, default=0.17)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--logits-aug", action="store_true")
    parser.add_argument("--logits-layers", default="25,30")
    parser.add_argument("--logits-alpha", type=float, default=0.3)
    myutils.add_ot_bary_sla_arguments(parser)
    return parser.parse_args()


def main(args):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    import myutils
    from eval_data_loader import COCODataSet, read_image_ids_file
    from llava.utils import disable_torch_init
    from llm_layers import add_vsv_layers, remove_vsv_layers
    from model_loader import ModelLoader
    from steering_vector import (
        add_logits_flag,
        obtain_vsv,
        remove_logits_flag,
    )

    if args.batch_size != 1:
        raise ValueError("Two-stage generation currently requires batch-size=1")
    myutils.validate_ot_bary_sla_arguments(args)
    myutils.seed_everything(args.seed)
    disable_torch_init()

    image_ids = read_image_ids_file(args.subset_ids_file)
    vista_proposals = load_proposals(args.vista_proposals)
    ot_proposals = load_proposals(args.ot_proposals)
    for name, proposals in (
        ("VISTA", vista_proposals),
        ("OT", ot_proposals),
    ):
        missing = [image_id for image_id in image_ids if image_id not in proposals]
        if missing:
            raise ValueError(
                f"{name} proposals miss {len(missing)} requested images; "
                f"first missing image_id={missing[0]}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.stats_output is not None:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        if args.stats_output.exists():
            raise FileExistsError(
                f"Stats output already exists: {args.stats_output}"
            )

    model_loader = ModelLoader(args.model)
    dataset = COCODataSet(
        data_path=args.data_path,
        trans=model_loader.image_processor,
        image_ids=image_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    template = myutils.prepare_template(args)

    stats_handle = None
    with args.output.open("w", encoding="utf-8") as output_handle:
        if args.stats_output is not None:
            stats_handle = args.stats_output.open("w", encoding="utf-8")
        try:
            for data in tqdm(loader, total=len(loader)):
                image_id = int(data["img_id"][0])
                image = data["image"]
                query = build_fusion_query(
                    args.fusion_mode,
                    vista_proposals[image_id],
                    ot_proposals[image_id],
                )
                with torch.inference_mode(), myutils.maybe_autocast(
                    args.model, model_loader.vlm_model.device,
                ):
                    questions, kwargs = model_loader.prepare_inputs_for_model(
                        template, [query], image,
                    )
                    if args.vsv:
                        neg_kwargs = model_loader.prepare_neg_prompt(
                            args, questions, template=template,
                        )
                        pos_kwargs = model_loader.prepare_pos_prompt(args, kwargs)
                        visual_vector, _ = obtain_vsv(
                            args,
                            model_loader.llm_model,
                            [[neg_kwargs, pos_kwargs]],
                            rank=1,
                        )
                        add_vsv_layers(
                            model_loader.llm_model,
                            torch.stack([visual_vector], dim=1).cuda(),
                            [args.vsv_lambda],
                            args.layers,
                        )

                    add_logits_flag(
                        model_loader.llm_model,
                        args,
                        tokenizer=model_loader.tokenizer,
                    )
                    if args.do_sample:
                        kwargs["top_p"] = args.top_p
                        kwargs["top_k"] = args.top_k
                    outputs = model_loader.llm_model.generate(
                        do_sample=args.do_sample,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        num_beams=args.num_beams,
                        output_attentions=False,
                        output_hidden_states=bool(args.logits_aug),
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        return_dict=True,
                        **kwargs,
                    )
                    diagnostics = remove_logits_flag(model_loader.llm_model)
                    if args.vsv:
                        remove_vsv_layers(model_loader.llm_model)
                    caption = model_loader.decode(outputs)[0]

                output_handle.write(json.dumps({
                    "image_id": image_id,
                    "caption": caption,
                    "fusion_mode": args.fusion_mode,
                    "vista_proposal": vista_proposals[image_id],
                    "ot_proposal": ot_proposals[image_id],
                }) + "\n")
                output_handle.flush()
                if stats_handle is not None:
                    stats_handle.write(json.dumps({
                        "image_id": image_id,
                        **diagnostics,
                    }) + "\n")
                    stats_handle.flush()
        finally:
            if stats_handle is not None:
                stats_handle.close()


if __name__ == "__main__":
    main(parse_args())
