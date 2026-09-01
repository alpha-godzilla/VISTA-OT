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
    group.add_argument(
        "--ot-visual-tokens",
        type=int,
        default=36,
        help="Legacy OT2 pooling target; ignored by unpooled attention OT.",
    )
    group.add_argument("--ot-sinkhorn-iters", type=int, default=3)
    group.add_argument(
        "--ot-sinkhorn-tolerance",
        type=float,
        default=1e-3,
        help="Maximum marginal residual for Sinkhorn early stopping.",
    )
    group.add_argument("--ot-epsilon", type=float, default=0.05)
    group.add_argument(
        "--ot-layer-temperature",
        type=float,
        default=0.1,
        help="Softmax temperature for inverse-OT-cost early-layer weights.",
    )
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
    group.add_argument(
        "--ot-attention-visual-marginal",
        action="store_true",
        help=(
            "Use current-token decoder attention over real image patches as "
            "the OT visual marginal; no visual dustbin is added."
        ),
    )
    group.add_argument(
        "--ot-attention-power",
        type=float,
        default=0.5,
        help="Power applied to unpooled visual attention before normalization.",
    )
    group.add_argument(
        "--ot-attention-uniform-mix",
        type=float,
        default=0.02,
        help="Small uniform smoothing mass over real visual OT nodes.",
    )
    group.add_argument(
        "--ot-attention-trace",
        action="store_true",
        help=(
            "For single-image diagnosis, save per-step visual attention "
            "marginals and OT layer weights in the OT statistics JSONL."
        ),
    )
    group.add_argument(
        "--ot-attention-coverage-beta",
        type=float,
        default=0.0,
        help=(
            "Coverage-aware marginal strength. Positive values modestly "
            "upweight visual patches that received less OT mass previously."
        ),
    )
    group.add_argument(
        "--ot-attention-coverage-epsilon",
        type=float,
        default=0.1,
        help="Stability floor for coverage-aware visual-marginal reweighting.",
    )
    group.add_argument(
        "--ot-adaptive-alpha",
        action="store_true",
        help=(
            "Scale logits_alpha per decoding step using the entropy of the "
            "effective visual OT marginal."
        ),
    )
    group.add_argument(
        "--ot-adaptive-alpha-min-ratio",
        type=float,
        default=0.25,
        help="Minimum fraction of logits_alpha used by adaptive alpha.",
    )
    group.add_argument(
        "--ot-recall-reward-lambda",
        type=float,
        default=0.0,
        help=(
            "Add a token-specific reward for visually supported patches that "
            "have received little previous attention-OT mass. Zero disables it."
        ),
    )
    group.add_argument(
        "--ot-recall-candidate-topk",
        type=int,
        default=16,
        help=(
            "Candidates per final/OT/early-logit source for the recall reward."
        ),
    )
    group.add_argument(
        "--ot-recall-temperature",
        type=float,
        default=0.1,
        help="Patch-softmax temperature for token-specific recall reward.",
    )
    group.add_argument(
        "--ot-recall-coverage-decay",
        type=float,
        default=1.0,
        help="Decay applied to accumulated visual coverage in the recall reward.",
    )
    group.add_argument(
        "--ot-recall-recovery-rho",
        type=float,
        default=0.0,
        help=(
            "Boundedly restore visually supported candidates suppressed by OT "
            "toward the same-alpha uniform-layer reference."
        ),
    )
    group.add_argument(
        "--ot-unbalanced",
        action="store_true",
        help=(
            "Replace balanced Sinkhorn with dustbin-free UOT. This is a "
            "seed- and label-free first-stage option."
        ),
    )
    group.add_argument(
        "--ot-marginal-relaxation",
        type=float,
        default=0.5,
        help="Symmetric KL marginal penalty for first-stage UOT.",
    )
    group.add_argument(
        "--ot-mass-aware-layer-weights",
        action="store_true",
        help="Add log transported mass to each inverse-cost layer score.",
    )
    group.add_argument(
        "--ot-direction-aware-gating",
        action="store_true",
        help=(
            "Gate logit promotion by current-attention UOT support and "
            "suppression by absence under a uniform whole-image UOT solve."
        ),
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
    if args.ot_layer_temperature <= 0:
        raise ValueError("--ot-layer-temperature must be positive")
    if args.ot_sinkhorn_tolerance <= 0:
        raise ValueError("--ot-sinkhorn-tolerance must be positive")
    if args.ot_attention_power <= 0:
        raise ValueError("--ot-attention-power must be positive")
    if not 0.0 <= args.ot_attention_uniform_mix < 1.0:
        raise ValueError("--ot-attention-uniform-mix must be in [0, 1)")
    if args.ot_attention_coverage_beta < 0:
        raise ValueError("--ot-attention-coverage-beta must be non-negative")
    if args.ot_attention_coverage_epsilon <= 0:
        raise ValueError("--ot-attention-coverage-epsilon must be positive")
    if not 0.0 <= args.ot_adaptive_alpha_min_ratio <= 1.0:
        raise ValueError("--ot-adaptive-alpha-min-ratio must be in [0, 1]")
    if getattr(args, "ot_recall_reward_lambda", 0.0) < 0:
        raise ValueError("--ot-recall-reward-lambda must be non-negative")
    if getattr(args, "ot_recall_candidate_topk", 16) <= 0:
        raise ValueError("--ot-recall-candidate-topk must be positive")
    if getattr(args, "ot_recall_temperature", 0.1) <= 0:
        raise ValueError("--ot-recall-temperature must be positive")
    if getattr(args, "ot_recall_coverage_decay", 1.0) < 0:
        raise ValueError("--ot-recall-coverage-decay must be non-negative")
    recovery_rho = getattr(args, "ot_recall_recovery_rho", 0.0)
    if not 0.0 <= recovery_rho <= 1.0:
        raise ValueError("--ot-recall-recovery-rho must be in [0, 1]")
    if getattr(args, "ot_recall_reward_lambda", 0.0) > 0 and recovery_rho > 0:
        raise ValueError(
            "--ot-recall-reward-lambda and --ot-recall-recovery-rho "
            "cannot both be positive"
        )
    if getattr(args, "ot_marginal_relaxation", 0.5) <= 0:
        raise ValueError("--ot-marginal-relaxation must be positive")
    unbalanced = getattr(args, "ot_unbalanced", False)
    mass_aware = getattr(args, "ot_mass_aware_layer_weights", False)
    directional = getattr(args, "ot_direction_aware_gating", False)
    if mass_aware and not unbalanced:
        raise ValueError("--ot-mass-aware-layer-weights requires --ot-unbalanced")
    if directional and not mass_aware:
        raise ValueError(
            "--ot-direction-aware-gating requires "
            "--ot-mass-aware-layer-weights"
        )
    if directional and not args.ot_attention_visual_marginal:
        raise ValueError(
            "--ot-direction-aware-gating requires "
            "--ot-attention-visual-marginal"
        )
    if directional and (
        getattr(args, "ot_adaptive_alpha", False)
        or getattr(args, "ot_recall_reward_lambda", 0.0) > 0
        or recovery_rho > 0
    ):
        raise ValueError(
            "--ot-direction-aware-gating cannot be combined with adaptive "
            "alpha or recall-reward/recovery ablations"
        )
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
            if getattr(args, "ot_attention_visual_marginal", False):
                file_parts.extend(
                    [
                        "otattn",
                        "nodust",
                        "layerhid",
                        "lmhead",
                        "tlogit",
                        f"m{args.ot_topk}",
                        "kunpooled",
                        f"it{args.ot_sinkhorn_iters}",
                        f"tol{args.ot_sinkhorn_tolerance}",
                        f"eps{args.ot_epsilon}",
                        f"ltemp{args.ot_layer_temperature}",
                        f"apow{args.ot_attention_power}",
                        f"amix{args.ot_attention_uniform_mix}",
                    ]
                )
                coverage_beta = getattr(args, "ot_attention_coverage_beta", 0.0)
                if coverage_beta:
                    file_parts.extend(
                        [
                            f"covbeta{coverage_beta}",
                            f"coveps{getattr(args, 'ot_attention_coverage_epsilon', 0.1)}",
                        ]
                    )
                if getattr(args, "ot_adaptive_alpha", False):
                    file_parts.append(
                        f"adaptamin{getattr(args, 'ot_adaptive_alpha_min_ratio', 0.25)}"
                    )
                if getattr(args, "ot_unbalanced", False):
                    file_parts.extend(
                        ["uot", f"mrel{args.ot_marginal_relaxation}"]
                    )
                if getattr(args, "ot_mass_aware_layer_weights", False):
                    file_parts.append("masslayer")
                if getattr(args, "ot_direction_aware_gating", False):
                    file_parts.append("dirgate")
                recall_lambda = getattr(args, "ot_recall_reward_lambda", 0.0)
                recovery_rho = getattr(args, "ot_recall_recovery_rho", 0.0)
                if recall_lambda or recovery_rho:
                    file_parts.extend(
                        [
                            # Keep the per-example filename below common
                            # NAME_MAX=255 filesystem limits. The full values
                            # also remain recorded in the sweep manifest.
                            (
                                f"rrh{recall_lambda}"
                                if recall_lambda
                                else f"rc{recovery_rho}"
                            ),
                            f"k{getattr(args, 'ot_recall_candidate_topk', 16)}",
                            f"t{getattr(args, 'ot_recall_temperature', 0.1)}",
                            f"d{getattr(args, 'ot_recall_coverage_decay', 1.0)}",
                        ]
                    )
            else:
                file_parts.extend(
                    [
                        "otbary",
                        "vdust",
                        "tlogit",
                        f"m{args.ot_topk}",
                        f"k{args.ot_visual_tokens}",
                        f"it{args.ot_sinkhorn_iters}",
                        f"eps{args.ot_epsilon}",
                        f"ltemp{args.ot_layer_temperature}",
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
