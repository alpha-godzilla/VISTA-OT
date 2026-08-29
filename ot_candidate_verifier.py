"""Candidate extraction and local OT verification utilities.

The inference-side helpers in this module deliberately do not import CHAIR or
the COCO object vocabulary.  Dataset labels are only consumed later by the
offline calibration script.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

import torch

from ot_bary_sla import _normalize, log_sinkhorn


_GENERIC_HEAD_STOPWORDS = {
    "area", "background", "bunch", "collection", "foreground", "front",
    "group", "image", "kind", "lot", "number", "object", "pair", "part", "photo",
    "photograph", "picture", "scene", "side", "type", "view",
}
_DETERMINERS = {"a", "an", "another", "the", "this", "that", "these", "those"}


@dataclass(frozen=True)
class CandidatePhrase:
    phrase: str
    head: str
    plural: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _simple_lemma(word: str) -> str:
    word = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", word.lower())
    irregular = {
        "children": "child", "feet": "foot", "geese": "goose",
        "men": "man", "mice": "mouse", "people": "person",
        "teeth": "tooth", "women": "woman",
    }
    if word in irregular:
        return irregular[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _noun_equivalents(word: str) -> set[str]:
    lemma = _simple_lemma(word)
    equivalents = {lemma}
    try:
        from nltk.corpus import wordnet

        for synset in wordnet.synsets(lemma, pos=wordnet.NOUN):
            for item in synset.lemma_names():
                normalized = item.replace("_", " ").lower()
                if " " not in normalized:
                    equivalents.add(_simple_lemma(normalized))
    except LookupError:
        pass
    return equivalents


def _tagged_words(text: str):
    """Tokenize without punkt and POS-tag with the configured NLTK data."""
    try:
        import nltk
        from nltk.tokenize import TreebankWordTokenizer

        words = TreebankWordTokenizer().tokenize(text)
        return nltk.pos_tag(words)
    except LookupError as exc:
        raise RuntimeError(
            "NLTK POS resources are missing. Set NLTK_DATA to the existing "
            "server data directory (normally /data/sun_yuxi/nltk_data)."
        ) from exc


def extract_noun_phrases(text: str, max_words: int = 4) -> List[CandidatePhrase]:
    """Extract conservative adjective/compound-noun phrases using generic POS.

    This intentionally avoids any COCO/CHAIR category list.  The retained
    phrase ends in a noun and contains at most ``max_words`` lexical tokens.
    """
    tagged = _tagged_words(text)
    chunks: List[List[tuple[str, str]]] = []
    current: List[tuple[str, str]] = []
    for raw_word, tag in tagged + [(".", ".")]:
        word = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", raw_word.lower())
        allowed = tag.startswith("NN") or tag.startswith("JJ")
        if allowed and word:
            current.append((word, tag))
            continue
        if current:
            noun_positions = [i for i, (_, item_tag) in enumerate(current) if item_tag.startswith("NN")]
            if noun_positions:
                chunks.append(current[: noun_positions[-1] + 1])
            current = []

    result: List[CandidatePhrase] = []
    seen = set()
    for chunk in chunks:
        chunk = chunk[-max_words:]
        while chunk and chunk[0][0] in _DETERMINERS:
            chunk.pop(0)
        if not chunk:
            continue
        head_word, head_tag = chunk[-1]
        head = _simple_lemma(head_word)
        if head in _GENERIC_HEAD_STOPWORDS or len(head) < 2:
            continue
        phrase = " ".join(word for word, _ in chunk)
        key = (phrase, head)
        if key in seen:
            continue
        seen.add(key)
        result.append(CandidatePhrase(
            phrase=phrase,
            head=head,
            plural=head_tag == "NNS" or head_tag == "NNPS",
        ))
    return result


def vista_only_candidates(
    vista_caption: str,
    ot_caption: str,
    max_candidates: int = 6,
) -> List[CandidatePhrase]:
    """Return generic VISTA noun phrases not semantically covered by OT."""
    vista = extract_noun_phrases(vista_caption)
    ot = extract_noun_phrases(ot_caption)
    ot_equivalents = set()
    ot_phrases = set()
    for candidate in ot:
        ot_equivalents.update(_noun_equivalents(candidate.head))
        ot_phrases.add(" ".join(_simple_lemma(x) for x in candidate.phrase.split()))

    selected = []
    selected_heads = set()
    for candidate in vista:
        normalized_phrase = " ".join(
            _simple_lemma(x) for x in candidate.phrase.split()
        )
        equivalents = _noun_equivalents(candidate.head)
        if normalized_phrase in ot_phrases or equivalents & ot_equivalents:
            continue
        if equivalents & selected_heads:
            continue
        selected.append(candidate)
        selected_heads.update(equivalents)
        if len(selected) >= max_candidates:
            break
    return selected


def vista_only_candidates_v2(
    vista_caption: str,
    ot_caption: str,
    max_candidates: int = 12,
) -> List[CandidatePhrase]:
    """High-recall VISTA-only proposals with conservative deduplication.

    The v1 extractor expands every WordNet sense of a noun.  That is useful
    for precision, but polysemous heads (for example ``light``) can be removed
    as already covered even when the two captions refer to different objects.
    V2 deliberately uses only normalized phrases and exact head lemmas.  The
    downstream visual verifier, rather than lexical synonym expansion, is
    responsible for rejecting redundant proposals.
    """
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    vista = extract_noun_phrases(vista_caption)
    ot = extract_noun_phrases(ot_caption)
    ot_phrases = {
        " ".join(_simple_lemma(word) for word in candidate.phrase.split())
        for candidate in ot
    }
    ot_heads = {candidate.head for candidate in ot}

    selected: List[CandidatePhrase] = []
    selected_keys = set()
    for candidate in vista:
        normalized_phrase = " ".join(
            _simple_lemma(word) for word in candidate.phrase.split()
        )
        # Exact heads cover simple variants (car/cars); compound phrases are
        # compared as a whole so ``traffic light`` is not collapsed to every
        # unrelated occurrence of the polysemous head ``light``.
        is_compound = len(normalized_phrase.split()) > 1
        if normalized_phrase in ot_phrases:
            continue
        if not is_compound and candidate.head in ot_heads:
            continue
        key = (normalized_phrase, candidate.head)
        if key in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(key)
        if len(selected) >= max_candidates:
            break
    return selected


def log_unbalanced_sinkhorn(
    cost: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    marginal_relaxation: float,
    num_iters: int,
    tolerance: float | None = None,
) -> torch.Tensor:
    """Log-domain entropic unbalanced OT without an explicit dustbin.

    Both marginal KL penalties use ``marginal_relaxation``.  Unlike balanced
    Sinkhorn, the returned plan is allowed to transport less than unit mass.
    Convergence is checked on the dual updates because marginal residuals do
    not vanish in an unbalanced problem.
    """
    if cost.ndim < 2:
        raise ValueError("cost must have at least two dimensions")
    if epsilon <= 0 or marginal_relaxation <= 0:
        raise ValueError("epsilon and marginal_relaxation must be positive")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive")
    if tolerance is not None and tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if source.shape[-1] != cost.shape[-2] or target.shape[-1] != cost.shape[-1]:
        raise ValueError("marginals do not match cost dimensions")

    with torch.autocast(device_type=cost.device.type, enabled=False):
        cost = cost.float()
        source = source.to(device=cost.device, dtype=torch.float32)
        target = target.to(device=cost.device, dtype=torch.float32)
        tiny = torch.finfo(torch.float32).tiny
        log_kernel = -cost / epsilon
        log_source = source.clamp_min(tiny).log()
        log_target = target.clamp_min(tiny).log()
        log_u = torch.zeros_like(log_source)
        log_v = torch.zeros_like(log_target)
        exponent = marginal_relaxation / (marginal_relaxation + epsilon)

        for iteration in range(num_iters):
            previous_u = log_u
            previous_v = log_v
            log_u = exponent * (
                log_source
                - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            )
            log_v = exponent * (
                log_target
                - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
            )
            should_check = tolerance is not None and (
                (iteration + 1) % 5 == 0 or iteration + 1 == num_iters
            )
            if should_check:
                update = torch.maximum(
                    (log_u - previous_u).abs().amax(),
                    (log_v - previous_v).abs().amax(),
                )
                if update.item() <= tolerance:
                    break
        return (
            log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        ).exp()


def word_balanced_target_marginal(
    token_strings: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Give each lexical word equal mass, split over its SentencePiece pieces."""
    if not token_strings:
        raise ValueError("Candidate must contain at least one token")
    groups: List[List[int]] = []
    for index, token in enumerate(token_strings):
        starts_word = index == 0 or token.startswith(("▁", "Ġ"))
        if starts_word:
            groups.append([])
        groups[-1].append(index)
    marginal = torch.zeros(len(token_strings), dtype=torch.float32, device=device)
    for group in groups:
        value = 1.0 / (len(groups) * len(group))
        marginal[group] = value
    return marginal


def candidate_token_span(
    tokenizer,
    original_input_ids: torch.Tensor,
    candidate: str,
    image_token_index: int = -200,
):
    """Locate the bracketed candidate without assuming prefix-stable BPE.

    SentencePiece attaches a preceding blank to the following token, so a
    separately tokenized text prefix is not guaranteed to match the prefix of
    the complete prompt. Search token subsequences in the already-tokenized
    post-image prompt and use the verifier's brackets to disambiguate instead.
    """
    image_positions = (original_input_ids == image_token_index).nonzero()
    if image_positions.numel() != 1:
        raise RuntimeError("Verifier prompt must contain exactly one image token")
    image_index = int(image_positions[0].item())
    after_ids = original_input_ids[image_index + 1:].tolist()
    variants = []
    for text in (f" {candidate}", candidate):
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if token_ids and token_ids not in variants:
            variants.append(token_ids)
    matches = []
    for token_ids in variants:
        width = len(token_ids)
        for offset in range(len(after_ids) - width + 1):
            if after_ids[offset:offset + width] == token_ids:
                matches.append((offset, offset + width, token_ids))
    bracketed_matches = []
    for start, end, token_ids in matches:
        left = tokenizer.decode(after_ids[max(0, start - 4):start])
        right = tokenizer.decode(after_ids[end:min(len(after_ids), end + 4)])
        if "[" in left and "]" in right:
            bracketed_matches.append((start, end, token_ids))
    if bracketed_matches:
        matches = bracketed_matches
    unique = {(start, end): token_ids for start, end, token_ids in matches}
    if len(unique) != 1:
        raise RuntimeError(
            "Could not uniquely locate candidate token subsequence in verifier "
            f"prompt; candidate={candidate!r}, matches={sorted(unique)}"
        )
    relative_start, relative_end = next(iter(unique))
    start = image_index + 1 + relative_start
    end = image_index + 1 + relative_end
    if end <= start:
        raise RuntimeError(f"Candidate produced no tokens: {candidate!r}")
    return start, end, original_input_ids[start:end]


def _transport_costs(
    cost: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    sinkhorn_iters: int,
    sinkhorn_tolerance: float,
) -> torch.Tensor:
    plan = log_sinkhorn(
        cost,
        source,
        target,
        epsilon=epsilon,
        num_iters=sinkhorn_iters,
        tolerance=sinkhorn_tolerance,
    )
    return (plan * cost.float()).sum(dim=(-2, -1))


@torch.no_grad()
def candidate_ot_features(
    layer_visual: torch.Tensor,
    layer_attention: torch.Tensor,
    token_features: torch.Tensor,
    token_strings: Sequence[str],
    region_topks: Iterable[int] = (8, 16, 32),
    attention_power: float = 0.75,
    uniform_mix: float = 0.02,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 50,
    sinkhorn_tolerance: float = 1e-3,
    layer_temperature: float = 0.06,
) -> Dict[str, object]:
    """Compute candidate-conditioned no-dustbin OT verification features.

    Args:
        layer_visual: ``[layers, visual_tokens, hidden]`` decoder states.
        layer_attention: ``[layers, candidate_queries, visual_tokens]`` raw
            attention probabilities, already averaged over heads.
        token_features: ``[candidate_tokens, hidden]`` lm-head rows.
    """
    if layer_visual.ndim != 3 or layer_attention.ndim != 3:
        raise ValueError("Visual states and attention must both be rank 3")
    if layer_visual.shape[:2] != (layer_attention.shape[0], layer_attention.shape[2]):
        raise ValueError("Layer/visual dimensions do not align")
    if token_features.ndim != 2 or token_features.shape[-1] != layer_visual.shape[-1]:
        raise ValueError("Candidate lm-head rows do not align with visual hidden states")
    if not 0 <= uniform_mix < 1:
        raise ValueError("uniform_mix must be in [0, 1)")

    visual = _normalize(layer_visual.float())
    tokens = _normalize(token_features.float())
    cost = 1.0 - torch.einsum("lkd,md->lkm", visual, tokens)
    raw = layer_attention.float().clamp_min(0).mean(dim=1)
    absolute_mass_by_layer = raw.sum(dim=-1)
    powered = raw.pow(attention_power)
    powered_total = powered.sum(dim=-1, keepdim=True)
    full_source = torch.where(
        powered_total > torch.finfo(torch.float32).tiny,
        powered / powered_total.clamp_min(torch.finfo(torch.float32).tiny),
        torch.full_like(powered, 1.0 / powered.shape[-1]),
    )
    full_source = (1.0 - uniform_mix) * full_source + uniform_mix / full_source.shape[-1]
    target = word_balanced_target_marginal(token_strings, cost.device)
    full_costs = _transport_costs(
        cost, full_source, target, epsilon, sinkhorn_iters, sinkhorn_tolerance,
    )
    normalized_attention = raw.mean(dim=0)
    normalized_attention = normalized_attention / normalized_attention.sum().clamp_min(1e-12)
    entropy = -(
        normalized_attention
        * normalized_attention.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum()
    normalized_entropy = entropy / math.log(max(2, normalized_attention.numel()))

    by_region: Dict[str, object] = {}
    visual_tokens = cost.shape[-2]
    global_attention = raw.mean(dim=0)
    for raw_topk in region_topks:
        topk = min(int(raw_topk), max(1, visual_tokens // 2))
        positive_indices = global_attention.topk(topk).indices
        positive_cost = cost[:, positive_indices, :]
        positive_raw = powered[:, positive_indices]
        positive_total = positive_raw.sum(dim=-1, keepdim=True)
        positive_source = torch.where(
            positive_total > torch.finfo(torch.float32).tiny,
            positive_raw / positive_total.clamp_min(torch.finfo(torch.float32).tiny),
            torch.full_like(positive_raw, 1.0 / topk),
        )
        positive_source = (1.0 - uniform_mix) * positive_source + uniform_mix / topk
        positive_costs = _transport_costs(
            positive_cost, positive_source, target,
            epsilon, sinkhorn_iters, sinkhorn_tolerance,
        )

        # A deliberately hard same-image counterfactual: among patches outside
        # the attended region, select those with the lowest candidate cost.
        affinity_cost = cost.mean(dim=(0, 2)).clone()
        affinity_cost[positive_indices] = float("inf")
        negative_indices = affinity_cost.topk(topk, largest=False).indices
        negative_cost = cost[:, negative_indices, :]
        negative_source = torch.full(
            (cost.shape[0], topk), 1.0 / topk,
            dtype=torch.float32, device=cost.device,
        )
        negative_costs = _transport_costs(
            negative_cost, negative_source, target,
            epsilon, sinkhorn_iters, sinkhorn_tolerance,
        )
        layer_weights = torch.softmax(-positive_costs / layer_temperature, dim=0)
        margins = negative_costs - positive_costs
        by_region[str(raw_topk)] = {
            "region_topk_effective": topk,
            "full_cost": float((layer_weights * full_costs).sum().item()),
            "positive_cost": float((layer_weights * positive_costs).sum().item()),
            "hard_negative_cost": float((layer_weights * negative_costs).sum().item()),
            "weighted_margin": float((layer_weights * margins).sum().item()),
            "median_margin": float(margins.median().item()),
            "positive_layer_fraction": float((margins > 0).float().mean().item()),
            "layer_positive_costs": positive_costs.cpu().tolist(),
            "layer_negative_costs": negative_costs.cpu().tolist(),
            "layer_margins": margins.cpu().tolist(),
            "layer_weights": layer_weights.cpu().tolist(),
        }

    return {
        "visual_attention_mass": float(absolute_mass_by_layer.mean().item()),
        "visual_attention_mass_by_layer": absolute_mass_by_layer.cpu().tolist(),
        "normalized_attention_entropy": float(normalized_entropy.item()),
        "regions": by_region,
    }


def _uot_summary(
    cost: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    marginal_relaxation: float,
    sinkhorn_iters: int,
    sinkhorn_tolerance: float,
) -> Dict[str, object]:
    plan = log_unbalanced_sinkhorn(
        cost,
        source,
        target,
        epsilon=epsilon,
        marginal_relaxation=marginal_relaxation,
        num_iters=sinkhorn_iters,
        tolerance=sinkhorn_tolerance,
    )
    mass = plan.sum(dim=(-2, -1))
    normalized_cost = (plan * cost.float()).sum(dim=(-2, -1)) / mass.clamp_min(1e-12)
    return {
        "plan": plan,
        "mass": mass,
        "normalized_cost": normalized_cost,
    }


@torch.no_grad()
def candidate_uot_features(
    layer_visual: torch.Tensor,
    layer_attention: torch.Tensor,
    token_features: torch.Tensor,
    token_strings: Sequence[str],
    region_topks: Iterable[int] = (8, 16, 32),
    marginal_relaxations: Iterable[float] = (0.1, 0.2, 0.5),
    attention_power: float = 0.75,
    uniform_mix: float = 0.02,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 50,
    sinkhorn_tolerance: float = 1e-3,
) -> Dict[str, object]:
    """Candidate-conditioned UOT evidence over attended visual regions.

    The feature layout is ``relaxations[tau][region_topk]``.  Plans are not
    serialized; only mass, normalized cost, hard-negative margin, and layer
    stability statistics needed by the offline scalar verifier are retained.
    """
    if layer_visual.ndim != 3 or layer_attention.ndim != 3:
        raise ValueError("Visual states and attention must both be rank 3")
    if layer_visual.shape[:2] != (layer_attention.shape[0], layer_attention.shape[2]):
        raise ValueError("Layer/visual dimensions do not align")
    if token_features.ndim != 2 or token_features.shape[-1] != layer_visual.shape[-1]:
        raise ValueError("Candidate lm-head rows do not align with visual hidden states")
    relaxations = [float(value) for value in marginal_relaxations]
    if not relaxations or any(value <= 0 for value in relaxations):
        raise ValueError("marginal_relaxations must be non-empty and positive")
    if not 0 <= uniform_mix < 1:
        raise ValueError("uniform_mix must be in [0, 1)")

    visual = _normalize(layer_visual.float())
    tokens = _normalize(token_features.float())
    cost = 1.0 - torch.einsum("lkd,md->lkm", visual, tokens)
    raw = layer_attention.float().clamp_min(0).mean(dim=1)
    powered = raw.pow(attention_power)
    target = word_balanced_target_marginal(token_strings, cost.device)
    global_attention = raw.mean(dim=0)
    visual_tokens = cost.shape[-2]
    result: Dict[str, object] = {}

    for relaxation in relaxations:
        regions: Dict[str, object] = {}
        for raw_topk in region_topks:
            topk = min(int(raw_topk), max(1, visual_tokens // 2))
            positive_indices = global_attention.topk(topk).indices
            positive_cost = cost[:, positive_indices, :]
            positive_raw = powered[:, positive_indices]
            positive_total = positive_raw.sum(dim=-1, keepdim=True)
            positive_source = torch.where(
                positive_total > torch.finfo(torch.float32).tiny,
                positive_raw / positive_total.clamp_min(torch.finfo(torch.float32).tiny),
                torch.full_like(positive_raw, 1.0 / topk),
            )
            positive_source = (
                (1.0 - uniform_mix) * positive_source + uniform_mix / topk
            )

            # Same-image hard negative is kept as a zero-extra-forward
            # contrast.  The optional noisy-image path is added by the GPU
            # scorer and compared offline.
            affinity_cost = cost.mean(dim=(0, 2)).clone()
            affinity_cost[positive_indices] = float("inf")
            negative_indices = affinity_cost.topk(topk, largest=False).indices
            negative_cost = cost[:, negative_indices, :]
            negative_source = torch.full(
                (cost.shape[0], topk), 1.0 / topk,
                dtype=torch.float32, device=cost.device,
            )
            positive = _uot_summary(
                positive_cost, positive_source, target,
                epsilon, relaxation, sinkhorn_iters, sinkhorn_tolerance,
            )
            negative = _uot_summary(
                negative_cost, negative_source, target,
                epsilon, relaxation, sinkhorn_iters, sinkhorn_tolerance,
            )
            layer_costs = positive["normalized_cost"]
            layer_masses = positive["mass"]
            layer_margins = negative["normalized_cost"] - layer_costs
            regions[str(raw_topk)] = {
                "region_topk_effective": topk,
                "transport_mass": float(layer_masses.mean().item()),
                "normalized_cost": float(layer_costs.mean().item()),
                "hard_negative_mass": float(negative["mass"].mean().item()),
                "hard_negative_normalized_cost": float(
                    negative["normalized_cost"].mean().item()
                ),
                "normalized_margin": float(layer_margins.mean().item()),
                "layer_cost_std": float(layer_costs.std(unbiased=False).item()),
                "layer_mass_std": float(layer_masses.std(unbiased=False).item()),
                "positive_layer_fraction": float(
                    (layer_margins > 0).float().mean().item()
                ),
                "layer_transport_masses": layer_masses.cpu().tolist(),
                "layer_normalized_costs": layer_costs.cpu().tolist(),
                "layer_normalized_margins": layer_margins.cpu().tolist(),
            }
        result[f"{relaxation:g}"] = regions
    return {"relaxations": result}


def append_candidates(
    caption: str,
    candidates: Sequence[Dict[str, object]],
) -> str:
    """Append accepted phrases without changing a character of the OT draft."""
    if not candidates:
        return caption
    base = caption.rstrip()
    if base and base[-1] not in ".!?":
        base += "."

    rendered = []
    for candidate in candidates:
        phrase = str(candidate["phrase"]).strip()
        plural = bool(candidate.get("plural", False))
        if plural or phrase.lower().split()[0] in _DETERMINERS:
            rendered.append(phrase)
        else:
            article = "an" if phrase[0].lower() in "aeiou" else "a"
            rendered.append(f"{article} {phrase}")
    if len(rendered) == 1:
        addition = f"Also visible is {rendered[0]}."
    else:
        addition = "Also visible are " + ", ".join(rendered[:-1]) + f" and {rendered[-1]}."
    return f"{base} {addition}".strip()
