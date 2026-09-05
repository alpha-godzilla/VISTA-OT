#!/usr/bin/env python3
"""Strict image/text-only teacher-forced visual-head-dependence oracle.

This diagnostic never feeds CHAIR labels into the model.  For each already
generated caption it compares matching caption-token positions under an image
prefix and a true text-only prefix (the LLaVA IMAGE_TOKEN_INDEX is removed).
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from anchor import IMAGE_TOKEN_INDEX, INSTRUCTION_TEMPLATE
from llava.utils import disable_torch_init
from model_loader import ModelLoader, prepare_llava_inputs


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", type=Path, required=True)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--layers", default="25,30")
    return p.parse_args()


def image_path(root, image_id):
    return root / f"COCO_val2014_{int(image_id):012d}.jpg"


def main():
    cli = args()
    start, end = (int(x) for x in cli.layers.split(","))
    selected = list(range(start, end + 1))
    captions = [json.loads(line) for line in cli.captions.open() if line.strip()]
    captions = captions[:cli.limit]
    if not captions:
        raise ValueError("No captions available for VHD")
    disable_torch_init()
    loader = ModelLoader("llava-1.5")
    model, tokenizer = loader.llm_model, loader.tokenizer
    modules = [model.model.layers[i].self_attn.o_proj for i in selected]
    captured = {}

    def hook(index):
        def save(_, inputs):
            captured[index] = inputs[0].detach()
        return save

    handles = [module.register_forward_pre_hook(hook(i)) for i, module in enumerate(modules)]
    template = INSTRUCTION_TEMPLATE["llava-1.5"]
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("w", encoding="utf-8") as output, torch.inference_mode():
        for row in captions:
            path = image_path(cli.data_path, row["image_id"])
            if not path.exists():
                raise FileNotFoundError(path)
            processed = loader.image_processor(Image.open(path).convert("RGB"))
            if hasattr(processed, "to"):
                processed = processed.to("cuda")
            _, kwargs = prepare_llava_inputs(
                template, ["Please help me describe the image in detail."], processed, tokenizer,
            )
            caption_ids = tokenizer(row["caption"], add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
            if caption_ids.numel() == 0:
                continue
            kwargs["input_ids"] = torch.cat([kwargs["input_ids"], caption_ids], dim=1)
            captured.clear()
            model(**kwargs, use_cache=False, output_attentions=False, return_dict=True)
            image_heads = [captured[i] for i in range(len(selected))]
            text_ids = kwargs["input_ids"][kwargs["input_ids"] != IMAGE_TOKEN_INDEX].view(1, -1)
            captured.clear()
            model(input_ids=text_ids, use_cache=False, output_attentions=False, return_dict=True)
            text_heads = [captured[i] for i in range(len(selected))]
            values = []
            count = caption_ids.shape[1]
            for image_value, text_value, module in zip(image_heads, text_heads, modules):
                heads = module.in_features // model.config.num_attention_heads
                image_value = image_value[:, -count:].float().view(1, count, model.config.num_attention_heads, heads)
                text_value = text_value[:, -count:].float().view(1, count, model.config.num_attention_heads, heads)
                # o_proj mixes head slices; project each slice independently.
                weight = module.weight.float().t().view(model.config.num_attention_heads, heads, -1)
                image_out = torch.einsum("bthd,hdo->btho", image_value, weight)
                text_out = torch.einsum("bthd,hdo->btho", text_value, weight)
                values.append((image_out - text_out).norm(dim=-1).squeeze(0).cpu().tolist())
            output.write(json.dumps({"image_id": int(row["image_id"]), "caption": row["caption"], "layers": selected, "vhd": values}) + "\n")
            output.flush()
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
