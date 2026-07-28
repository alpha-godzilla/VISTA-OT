import os
import random
import contextlib
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from anchor import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE


def seed_everything(seed):
    # seed everything for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def init_folder_structure(args):
    save_dir = f"./exp_results/{args.exp_folder}/{args.model}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def prepare_template(args):
    template = INSTRUCTION_TEMPLATE[args.model]
    if args.model == "llava-1.5" or args.model == "shikra":
        template = SYSTEM_MESSAGE + template
    return template


def add_ot_bary_sla_arguments(parser):
    group = parser.add_argument_group("OT-BarySLA")
    group.add_argument(
        "--use-ot-bary-sla",
        action="store_true",
        help="Use OT-derived dynamic weights instead of uniform SLA weights.",
    )
    group.add_argument("--ot-topk", type=int, default=8)
    group.add_argument("--ot-visual-tokens", type=int, default=36)
    group.add_argument("--ot-sinkhorn-iters", type=int, default=3)
    group.add_argument("--ot-epsilon", type=float, default=0.05)
    group.add_argument(
        "--ot-log-stats",
        action="store_true",
        help="Save generation-level OT diagnostics without per-token printing.",
    )
    group.add_argument(
        "--ot-force-uniform",
        action="store_true",
        help="Regression mode: force uniform early-layer weights.",
    )
    return parser


def validate_ot_bary_sla_arguments(args):
    if not getattr(args, "use_ot_bary_sla", False):
        return
    if args.model != "llava-1.5":
        raise ValueError(
            "The first OT-BarySLA integration supports only --model llava-1.5"
        )
    if not args.logits_aug:
        raise ValueError("--use-ot-bary-sla requires --logits-aug")
    if not 0.0 <= args.logits_alpha <= 1.0:
        raise ValueError("--logits-alpha must be in [0, 1]")
    try:
        start_layer, end_layer = map(int, args.logits_layers.split(","))
    except ValueError as exc:
        raise ValueError(
            "--logits-layers must be an inclusive START,END pair"
        ) from exc
    if start_layer > end_layer:
        raise ValueError("--logits-layers START must not exceed END")


def prepare_common_fileparts(args):
    file_parts = []

    # fix seed
    file_parts.append(f"seed{args.seed}")

    # visual steering vector
    if args.vsv:
        file_parts.append("vsv")
        file_parts.append(f"lambda_{args.vsv_lambda}")
        if args.layers is not None:
            file_parts.append(f"layers_{args.layers}")
    else:
        file_parts.append("org")

    # logits augmentation
    if args.logits_aug:
        file_parts.append("logaug")
        file_parts.append(f"loglayer_{args.logits_layers}")
        file_parts.append(f"logalpha_{args.logits_alpha}")
        if getattr(args, "use_ot_bary_sla", False):
            file_parts.extend(
                [
                    "otbary",
                    f"m{args.ot_topk}",
                    f"k{args.ot_visual_tokens}",
                    f"it{args.ot_sinkhorn_iters}",
                    f"eps{args.ot_epsilon}",
                ]
            )
            if args.ot_force_uniform:
                file_parts.append("uniform")

    # decoding strategy
    if args.do_sample:
        file_parts.append(f"nucleus_{args.top_p}")
    else:
        if args.num_beams > 1:
            file_parts.append(f"beam{args.num_beams}")
        elif args.num_beams == 1:
            file_parts.append("greedy")
        else:
            raise ValueError("Invalid beam size")
    if args.temperature != 1.0:
        file_parts.append(f"temp_{args.temperature}")
    if args.repetition_penalty != 1.0:
        file_parts.append(f"repe_{args.repetition_penalty}")
    if args.no_repeat_ngram_size is not None:
        file_parts.append(f"no_repeat_{args.no_repeat_ngram_size}")

    # add max new tokens
    file_parts.append(f"max_new_tokens_{args.max_new_tokens}")
    return file_parts


def get_coco_path_from_id(img_id, data_path):
    # get image path from image id
    if type(img_id) == torch.tensor:
        tem_img_id = img_id.item()
    else:
        tem_img_id = img_id
    tem_img_id = str(tem_img_id)
    if len(tem_img_id) < 6:  # add zeron in front of img_id
        tem_img_id = '0' * (6 - len(tem_img_id)) + tem_img_id
    img_name = f'COCO_val2014_000000{tem_img_id}.jpg'
    image_path = os.path.join(data_path, img_name)
    return image_path


def maybe_autocast(model_name, device, dtype=torch.float16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    target_model = ['instructblip']
    if model_name in target_model and device != 'cpu':
        return  torch.amp.autocast('cuda', dtype=dtype)
    else:
        return contextlib.nullcontext()


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_
