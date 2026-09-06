import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset

import myutils
from eval_data_loader import COCODataSet, read_image_ids_file
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from steering_vector import obtain_vsv, add_logits_flag, remove_logits_flag
from llm_layers import add_vsv_layers, remove_vsv_layers
from vista_paths import COCO_VAL2014_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR evaluation on MLLMs.")
    # General arguments
    parser.add_argument("--exp_folder", type=str, default="chair_eval", help="save folder name")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--data-path", type=str, default=COCO_VAL2014_PATH, help="COCO val2014 image directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument(
        "--subset-ids-file",
        type=str,
        default=None,
        help=(
            "Text file containing one COCO image ID per line. When set, this "
            "fixed ordered subset replaces random --subset-size sampling."
        ),
    )

    # Visual steering vector arguments
    parser.add_argument("--vsv", action="store_true", help='Use visual steering vector')
    parser.add_argument("--vsv-lambda", type=float, default=0.1)
    parser.add_argument("--layers", default=None)

    # penultimate logits augmentation
    parser.add_argument("--logits-aug", action="store_true", help='Use penultimate logits augmentation')
    parser.add_argument("--logits-layers", type=str, default='25,30', help='Layer for penultimate logits augmentation')
    parser.add_argument("--logits-alpha", type=float, default=0.3, help='Alpha for penultimate logits augmentation')
    myutils.add_ot_bary_sla_arguments(parser)

    # Decoding arguments
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Resume only samples already present in generation records.")
    parser.add_argument(
        "--retrieval-shift-trace-dir", type=str, default=None,
        help=(
            "Optional directory for non-invasive per-layer/head visual-prompt-"
            "generated retrieval metrics. When omitted, no hooks are installed "
            "and normal generation is unchanged."
        ),
    )
    parser.add_argument(
        "--retrieval-shift-debug", type=str, default=None,
        help="Optional raw Q/K/V dump selector: sample_id:layer:head:timestep.",
    )
    parser.add_argument(
        "--retrieval-shift-compact", action="store_true",
        help=(
            "Store only the event-analysis metrics in float16 plus validity "
            "masks. Numerical reconstruction checks are kept in sanity.jsonl; "
            "this substantially reduces trace-disk usage."
        ),
    )
    parser.add_argument(
        "--retrieval-shift-summary-file", type=str, default=None,
        help=(
            "Optional single flat NPZ summary written after tracing. It retains "
            "sample_id/timestep/layer/head rather than averaging them."
        ),
    )
    parser.add_argument(
        "--generation-record-dir", type=str, default=None,
        help=(
            "Optional light-weight generation records. This does not install "
            "retrieval hooks and is intended for the Phase-A confirmatory "
            "collection."
        ),
    )
    parser.add_argument(
        "--generation-decision-stats", action="store_true",
        help=(
            "Store only per-token logprob, entropy, top-1/top-2 logits and "
            "their margin in --generation-record-dir. Generation itself is "
            "unchanged; full score tensors are never written."
        ),
    )

    return parser.parse_args()


def get_file_name(args):
    file_name = "_".join(myutils.prepare_common_fileparts(args))
    return file_name


def main(args):
    # bath size should be 1 as we are generating image specific steering vectors
    assert args.batch_size == 1, "Batch size should be 1"
    if args.retrieval_shift_trace_dir is not None and args.num_beams != 1:
        raise ValueError("--retrieval-shift-trace-dir currently requires --num-beams 1")
    if args.retrieval_shift_summary_file is not None and args.retrieval_shift_trace_dir is None:
        raise ValueError("--retrieval-shift-summary-file requires --retrieval-shift-trace-dir")
    if args.generation_decision_stats and args.generation_record_dir is None:
        raise ValueError("--generation-decision-stats requires --generation-record-dir")
    myutils.validate_ot_bary_sla_arguments(args)
    # seed everything
    myutils.seed_everything(args.seed)
    # disable torch init
    disable_torch_init()
    # init_folder_structure
    args.save_dir = myutils.init_folder_structure(args)
    # prepare file name
    args.file_name = get_file_name(args)

    # prepare save file
    result_file = os.path.join(args.save_dir, args.file_name + ".jsonl")
    if os.path.exists(result_file) and not args.resume:
        exit(f"Result file {result_file} already exists. Exiting.")
    f = open(result_file, "a" if args.resume else "w", encoding="utf-8")
    stats_file = None
    if args.use_ot_bary_sla and (args.ot_log_stats or args.ot_attention_trace):
        stats_file = open(
            os.path.join(args.save_dir, args.file_name + "_ot_stats.jsonl"),
            "w",
            encoding="utf-8",
        )

    # get model loader
    model_loader = ModelLoader(args.model)
    retrieval_shift_tracer = None
    trace_generation_file = None
    generation_record_file = None
    completed_generation_ids = set()
    if args.retrieval_shift_trace_dir is not None:
        from visual_memory_retrieval import RetrievalShiftTracer

        retrieval_shift_tracer = RetrievalShiftTracer(
            model_loader.llm_model,
            model_loader.tokenizer,
            args.retrieval_shift_trace_dir,
            debug=args.retrieval_shift_debug,
            compact=args.retrieval_shift_compact,
        )
        retrieval_shift_tracer.install()
        model_loader.llm_model.retrieval_shift_tracer = retrieval_shift_tracer
        os.makedirs(args.retrieval_shift_trace_dir, exist_ok=True)
        trace_generation_file = open(
            os.path.join(args.retrieval_shift_trace_dir, "generation.jsonl"),
            "a", encoding="utf-8",
        )
    if args.generation_record_dir is not None:
        os.makedirs(args.generation_record_dir, exist_ok=True)
        record_path = os.path.join(args.generation_record_dir, "generation.jsonl")
        if args.resume and os.path.exists(record_path):
            with open(record_path, encoding="utf-8") as existing:
                for line in existing:
                    try:
                        completed_generation_ids.add(int(json.loads(line)["sample_id"]))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # A torn final append is intentionally not considered complete.
                        continue
        generation_record_file = open(
            record_path,
            "a", encoding="utf-8",
        )
    # get dataloader
    fixed_image_ids = None
    if args.subset_ids_file is not None:
        fixed_image_ids = read_image_ids_file(args.subset_ids_file)
        print(
            f"Using {len(fixed_image_ids)} fixed COCO image IDs from "
            f"{args.subset_ids_file}"
        )
    coco_dataset = COCODataSet(
        data_path=args.data_path,
        trans=model_loader.image_processor,
        image_ids=fixed_image_ids,
    )
    # get a randomly sample subdataset without replacement and fixed seed
    if (
        fixed_image_ids is None
        and args.subset_size > 0
        and args.subset_size < len(coco_dataset)
    ):
        subset_indices = np.random.choice(len(coco_dataset), args.subset_size, replace=False)
        coco_dataset = Subset(coco_dataset, subset_indices)

    coco_loader = DataLoader(coco_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    # prepare template
    template = myutils.prepare_template(args)

    # inference
    for _, data in tqdm(enumerate(coco_loader), total=len(coco_loader)):
        with torch.inference_mode():
            img_id = data["img_id"]
            if int(img_id[0]) in completed_generation_ids:
                continue
            image = data["image"]
            batch_size = img_id.shape[0]
            query = ["Please help me describe the image in detail."] * batch_size
            if retrieval_shift_tracer is not None:
                retrieval_shift_tracer.start_sample(int(img_id[0]), prompt=query[0])

            with myutils.maybe_autocast(args.model, model_loader.vlm_model.device):
                # prepare inputs
                questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image)

                # add visual steering vectors
                if args.vsv:
                    neg_kwargs = model_loader.prepare_neg_prompt(args, questions, template=template)
                    pos_kwargs = model_loader.prepare_pos_prompt(args, kwargs)
                    # generate visual steering vectors
                    visual_vector, _ = obtain_vsv(args, model_loader.llm_model, [[neg_kwargs, pos_kwargs]], rank=1)

                    # add steering vectors
                    add_vsv_layers(model_loader.llm_model, torch.stack([visual_vector], dim=1).cuda(), [args.vsv_lambda], args.layers)
                    
                # add logits augmentation flag
                add_logits_flag(
                    model_loader.llm_model,
                    args,
                    tokenizer=model_loader.tokenizer,
                )

                # generate
                if args.do_sample:
                    kwargs['top_p'] = args.top_p
                    kwargs['top_k'] = args.top_k

                # generate
                generation_input_length = kwargs["input_ids"].shape[1]
                outputs = model_loader.llm_model.generate(
                    do_sample=args.do_sample,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    num_beams=args.num_beams,
                    output_attentions=False,
                    output_hidden_states=True if args.logits_aug else False,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    return_dict=True,
                    output_scores=args.generation_decision_stats,
                    **kwargs
                    )

                # remove logits augmentation flag
                ot_diagnostics = remove_logits_flag(model_loader.llm_model)

                if args.vsv:
                    # remove steering vectors 
                    remove_vsv_layers(model_loader.llm_model)

            output_text = model_loader.decode(outputs)

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated_ids = sequences[0, generation_input_length:].detach().cpu().tolist()
        generation_row = None
        if retrieval_shift_tracer is not None or generation_record_file is not None:
            generation_row = {
                "sample_id": int(img_id[0]),
                "image_id": int(img_id[0]),
                "image_path": os.path.join(args.data_path, f"COCO_val2014_{int(img_id[0]):012d}.jpg"),
                "prompt": query[0],
                "generated_text": output_text[0],
                "generated_token_ids": generated_ids,
                "generated_tokens": [
                    model_loader.tokenizer.decode([token], clean_up_tokenization_spaces=False)
                    for token in generated_ids
                ],
                "generation_length": len(generated_ids),
                "token_alignment": "generated token index j is predicted by q timestep j",
            }
            if args.generation_decision_stats:
                decision_stats = []
                scores = getattr(outputs, "scores", None)
                if scores is None or len(scores) != len(generated_ids):
                    raise RuntimeError("Generation score/token length mismatch")
                for token_index, (token_id, score) in enumerate(zip(generated_ids, scores)):
                    logits = score[0].float()
                    log_norm = torch.logsumexp(logits, dim=-1)
                    logprob = float((logits[int(token_id)] - log_norm).cpu())
                    probabilities = torch.softmax(logits, dim=-1)
                    entropy = float((-(probabilities * torch.log_softmax(logits, dim=-1)).sum()).cpu())
                    top_values, top_ids = torch.topk(logits, k=2)
                    decision_stats.append({
                        "token_index": token_index, "token_id": int(token_id),
                        "logprob": logprob, "entropy": entropy,
                        "top1_id": int(top_ids[0]), "top1_logit": float(top_values[0]),
                        "top2_id": int(top_ids[1]), "top2_logit": float(top_values[1]),
                        "top1_top2_margin": float(top_values[0] - top_values[1]),
                    })
                generation_row["decision_stats"] = decision_stats
        if retrieval_shift_tracer is not None:
            trace_path = retrieval_shift_tracer.finish_sample(
                generated_token_ids=generated_ids,
                generated_text=output_text[0],
            )
            print(f"Wrote retrieval-shift trace to {trace_path}")
            trace_generation_file.write(json.dumps(generation_row) + "\n")
            trace_generation_file.flush()
        if generation_record_file is not None:
            generation_record_file.write(json.dumps(generation_row) + "\n")
            generation_record_file.flush()

        # write to file
        for i in range(len(output_text)):
            f.write(json.dumps({"image_id": int(img_id[i]), "caption": output_text[i]}) + "\n")
        f.flush()
        if stats_file is not None:
            stats_file.write(
                json.dumps(
                    {
                        "image_id": int(img_id[0]),
                        **ot_diagnostics,
                    }
                )
                + "\n"
            )
            stats_file.flush()
    f.close()
    if stats_file is not None:
        stats_file.close()
    if retrieval_shift_tracer is not None:
        retrieval_shift_tracer.remove()
        del model_loader.llm_model.retrieval_shift_tracer
        trace_generation_file.close()
    if generation_record_file is not None:
        generation_record_file.close()
    if args.retrieval_shift_summary_file is not None:
        from visual_memory_retrieval import merge_trace_directory

        summary_path = merge_trace_directory(
            args.retrieval_shift_trace_dir,
            args.retrieval_shift_summary_file,
        )
        print(f"Wrote retrieval-shift flat summary to {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
