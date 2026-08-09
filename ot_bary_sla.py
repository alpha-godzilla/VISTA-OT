"""Optimal-transport-guided barycentric Self-Logits Augmentation.

This module is intentionally independent of the LLaVA generation code.  The
model integration only needs to cache projected visual tokens and pass the
selected early-layer logits plus the input embedding matrix to ``OTBarySLA``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]


def _normalize(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(eps)


def _pool_visual_tokens(
    visual_tokens: torch.Tensor,
    target_tokens: int,
) -> torch.Tensor:
    """Pool ``[B, K0, D]`` visual tokens to at most ``target_tokens`` tokens."""
    if visual_tokens.ndim != 3:
        raise ValueError(
            "Projected visual tokens must have shape [batch, tokens, hidden]; "
            f"got {tuple(visual_tokens.shape)}"
        )
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive; got {target_tokens}")

    token_count = visual_tokens.shape[1]
    if token_count <= target_tokens:
        return visual_tokens

    source_side = math.isqrt(token_count)
    target_side = math.isqrt(target_tokens)
    if (
        source_side * source_side == token_count
        and target_side * target_side == target_tokens
    ):
        pooled = F.adaptive_avg_pool2d(
            visual_tokens.transpose(1, 2).reshape(
                visual_tokens.shape[0],
                visual_tokens.shape[2],
                source_side,
                source_side,
            ),
            (target_side, target_side),
        )
        return pooled.flatten(2).transpose(1, 2)

    return F.adaptive_avg_pool1d(
        visual_tokens.transpose(1, 2),
        target_tokens,
    ).transpose(1, 2)


def log_sinkhorn(
    cost: torch.Tensor,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    epsilon: float,
    num_iters: int,
) -> torch.Tensor:
    """Solve a small entropic OT problem with log-domain Sinkhorn updates.

    Args:
        cost: Tensor with shape ``[..., source_nodes, target_nodes]``.
        source_marginal: Broadcastable tensor with shape ``[..., source_nodes]``.
        target_marginal: Broadcastable tensor with shape ``[..., target_nodes]``.
        epsilon: Positive entropic regularization coefficient.
        num_iters: Number of alternating Sinkhorn updates.

    Returns:
        A float32 transport plan with the same leading dimensions as ``cost``.
    """
    if cost.ndim < 2:
        raise ValueError(f"cost must have at least two dimensions; got {cost.ndim}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive; got {epsilon}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive; got {num_iters}")
    if source_marginal.shape[-1] != cost.shape[-2]:
        raise ValueError("source marginal size does not match the cost matrix")
    if target_marginal.shape[-1] != cost.shape[-1]:
        raise ValueError("target marginal size does not match the cost matrix")

    with torch.autocast(device_type=cost.device.type, enabled=False):
        cost_fp32 = cost.float()
        source_fp32 = source_marginal.to(
            device=cost.device,
            dtype=torch.float32,
        )
        target_fp32 = target_marginal.to(
            device=cost.device,
            dtype=torch.float32,
        )
        tiny = torch.finfo(torch.float32).tiny

        log_kernel = -cost_fp32 / epsilon
        log_source = source_fp32.clamp_min(tiny).log()
        log_target = target_fp32.clamp_min(tiny).log()
        log_u = torch.zeros_like(log_source)
        log_v = torch.zeros_like(log_target)

        for _ in range(num_iters):
            log_u = log_source - torch.logsumexp(
                log_kernel + log_v.unsqueeze(-2),
                dim=-1,
            )
            log_v = log_target - torch.logsumexp(
                log_kernel + log_u.unsqueeze(-1),
                dim=-2,
            )

        log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        return log_plan.exp()


class OTBarySLA:
    """Compute OMIT-style OT weights for VISTA's early-layer logits."""

    def __init__(
        self,
        topk: int = 8,
        visual_tokens: int = 36,
        epsilon: float = 0.05,
        sinkhorn_iters: int = 3,
        special_token_ids: Optional[Iterable[int]] = None,
        log_stats: bool = False,
        force_uniform: bool = False,
    ):
        if topk <= 0:
            raise ValueError(f"topk must be positive; got {topk}")
        if visual_tokens <= 0:
            raise ValueError(
                f"visual_tokens must be positive; got {visual_tokens}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive; got {epsilon}")
        if sinkhorn_iters <= 0:
            raise ValueError(
                f"sinkhorn_iters must be positive; got {sinkhorn_iters}"
            )

        self.topk = topk
        self.visual_tokens = visual_tokens
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.special_token_ids = tuple(
            sorted({int(token_id) for token_id in (special_token_ids or [])})
        )
        self.log_stats = log_stats
        self.force_uniform = force_uniform

        self._visual_extended: Optional[torch.Tensor] = None
        self._stats: Dict[str, torch.Tensor] = {}
        self._stats_steps = 0

    @property
    def has_visual_cache(self) -> bool:
        return self._visual_extended is not None

    def clear(self) -> None:
        self._visual_extended = None
        self._stats = {}
        self._stats_steps = 0

    @torch.no_grad()
    def cache_visual_features(self, projected_visual_tokens: TensorOrTensors) -> None:
        """Pool and cache projected image features for one generation sequence.

        ``projected_visual_tokens`` can be one ``[B, K0, D]`` tensor or a
        sequence of per-sample ``[Ki, D]`` tensors.  The sequence form supports
        samples whose unpooled visual-token counts differ.
        """
        if isinstance(projected_visual_tokens, torch.Tensor):
            if projected_visual_tokens.ndim == 2:
                projected_visual_tokens = projected_visual_tokens.unsqueeze(0)
            pooled = _pool_visual_tokens(
                projected_visual_tokens.detach().float(),
                self.visual_tokens,
            )
            local = _normalize(pooled)
        else:
            pooled_samples: List[torch.Tensor] = []
            for sample in projected_visual_tokens:
                if sample.ndim != 2:
                    raise ValueError(
                        "Each per-sample visual tensor must have shape "
                        f"[tokens, hidden]; got {tuple(sample.shape)}"
                    )
                pooled_sample = _pool_visual_tokens(
                    sample.detach().float().unsqueeze(0),
                    self.visual_tokens,
                ).squeeze(0)
                pooled_samples.append(pooled_sample)

            if not pooled_samples:
                raise ValueError("At least one visual sample is required")
            pooled_lengths = {sample.shape[0] for sample in pooled_samples}
            if len(pooled_lengths) != 1:
                raise ValueError(
                    "Visual samples must pool to the same token count; "
                    f"got {sorted(pooled_lengths)}"
                )
            local = _normalize(torch.stack(pooled_samples, dim=0))

        global_feature = _normalize(local.mean(dim=1, keepdim=True))
        self._visual_extended = torch.cat([local, global_feature], dim=1)
        self._stats = {}
        self._stats_steps = 0

    def _expanded_visual_features(self, batch_size: int) -> torch.Tensor:
        if self._visual_extended is None:
            raise RuntimeError(
                "OT-BarySLA visual features are not cached. The projected "
                "visual tokens must be cached during multimodal prefill."
            )

        visual = self._visual_extended
        if visual.shape[0] == batch_size:
            return visual
        if batch_size % visual.shape[0] != 0:
            raise ValueError(
                "Cannot align cached visual batch with logits batch: "
                f"{visual.shape[0]} vs {batch_size}"
            )
        return visual.repeat_interleave(batch_size // visual.shape[0], dim=0)

    def _candidate_ids(self, early_logits: torch.Tensor) -> torch.Tensor:
        vocab_size = early_logits.shape[-1]
        invalid_ids = [
            token_id
            for token_id in self.special_token_ids
            if 0 <= token_id < vocab_size
        ]
        if vocab_size - len(invalid_ids) < self.topk:
            raise ValueError(
                f"Only {vocab_size - len(invalid_ids)} valid vocabulary tokens "
                f"remain, fewer than ot_topk={self.topk}"
            )

        if invalid_ids:
            ranked_logits = early_logits.clone()
            invalid = torch.tensor(
                invalid_ids,
                dtype=torch.long,
                device=early_logits.device,
            )
            ranked_logits.index_fill_(-1, invalid, float("-inf"))
        else:
            ranked_logits = early_logits
        return ranked_logits.topk(self.topk, dim=-1).indices

    def _update_stats(
        self,
        layer_scores: torch.Tensor,
        layer_weights: torch.Tensor,
        transport_plan: torch.Tensor,
    ) -> None:
        if not self.log_stats:
            return

        # Only the visual side has a dustbin. All text columns are real
        # top-k candidates and therefore remain in the local transport mass.
        local_plan = transport_plan[..., :-1, :]
        entropy = -(
            layer_weights.float()
            * layer_weights.float().clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1)
        values = {
            "layer_scores": layer_scores.float().mean(dim=0),
            "layer_weights": layer_weights.float().mean(dim=0),
            "layer_weight_entropy": entropy.mean(),
            "local_transport_mass": local_plan.sum(dim=(-2, -1)).mean(),
            "dustbin_to_token_mass": transport_plan[..., -1, :]
            .sum(dim=-1)
            .mean(),
        }
        if not self._stats:
            self._stats = {name: value.detach().clone() for name, value in values.items()}
        else:
            for name, value in values.items():
                self._stats[name].add_(value.detach())
        self._stats_steps += 1

    def get_diagnostics(self) -> Dict[str, Union[int, float, List[float]]]:
        """Return generation-level statistics, synchronizing only once."""
        if not self.log_stats or self._stats_steps == 0:
            return {}

        result: Dict[str, Union[int, float, List[float]]] = {
            "steps": self._stats_steps
        }
        for name, total in self._stats.items():
            mean = (total / self._stats_steps).detach().float().cpu()
            result[f"mean_{name}"] = (
                mean.item() if mean.ndim == 0 else mean.tolist()
            )
        return result

    @torch.no_grad()
    def compute_layer_weights(
        self,
        early_logits: torch.Tensor,
        input_embedding_weight: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute dynamic weights for ``early_logits``.

        Args:
            early_logits: ``[B, w, vocab_size]``.
            input_embedding_weight: ``[vocab_size, hidden_dim]``.
            return_details: Also return the transport tensors used by tests and
                optional diagnostics.
        """
        if early_logits.ndim != 3:
            raise ValueError(
                "early_logits must have shape [batch, layers, vocab]; "
                f"got {tuple(early_logits.shape)}"
            )
        if input_embedding_weight.ndim != 2:
            raise ValueError(
                "input_embedding_weight must have shape [vocab, hidden]; "
                f"got {tuple(input_embedding_weight.shape)}"
            )
        if input_embedding_weight.shape[0] < early_logits.shape[-1]:
            raise ValueError(
                "Input embedding vocabulary is smaller than the logits "
                f"vocabulary: {input_embedding_weight.shape[0]} < "
                f"{early_logits.shape[-1]}"
            )

        batch_size, layer_count, _ = early_logits.shape
        visual = self._expanded_visual_features(batch_size).to(
            device=early_logits.device,
            dtype=torch.float32,
        )
        if visual.shape[-1] != input_embedding_weight.shape[-1]:
            raise ValueError(
                "Projected visual-token dimension and token-embedding "
                f"dimension differ: {visual.shape[-1]} vs "
                f"{input_embedding_weight.shape[-1]}"
            )

        candidate_ids = self._candidate_ids(early_logits)
        with torch.autocast(device_type=early_logits.device.type, enabled=False):
            candidate_features = F.embedding(
                candidate_ids,
                input_embedding_weight,
            ).float()
            text_local = _normalize(candidate_features)

            similarity = torch.einsum(
                "bkd,bwmd->bwkm",
                visual,
                text_local,
            )
            cost = 1.0 - similarity
            visual_nodes = visual.shape[-2]
            source_marginal = torch.full(
                (1, 1, visual_nodes),
                1.0 / visual_nodes,
                dtype=torch.float32,
                device=cost.device,
            )
            candidate_logits = torch.gather(
                early_logits.float(),
                dim=-1,
                index=candidate_ids,
            )
            target_marginal = F.softmax(candidate_logits, dim=-1)
            transport_plan = log_sinkhorn(
                cost,
                source_marginal,
                target_marginal,
                epsilon=self.epsilon,
                num_iters=self.sinkhorn_iters,
            )

            local_plan = transport_plan[..., :-1, :]
            local_similarity = similarity[..., :-1, :]
            layer_scores = (local_plan * local_similarity).sum(dim=(-2, -1))
            if self.force_uniform:
                layer_weights_fp32 = torch.full_like(
                    layer_scores,
                    1.0 / layer_count,
                )
            else:
                layer_weights_fp32 = torch.softmax(
                    layer_scores,
                    dim=-1,
                )

        self._update_stats(layer_scores, layer_weights_fp32, transport_plan)
        layer_weights = layer_weights_fp32.to(dtype=early_logits.dtype)
        if not return_details:
            return layer_weights

        details = {
            "candidate_ids": candidate_ids,
            "similarity": similarity,
            "transport_plan": transport_plan,
            "source_marginal": source_marginal,
            "target_marginal": target_marginal,
            "layer_scores": layer_scores,
        }
        return layer_weights, details

    @torch.no_grad()
    def aggregate(
        self,
        early_logits: torch.Tensor,
        final_logits: torch.Tensor,
        input_embedding_weight: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the reverse-KL barycentric logit mixture."""
        if final_logits.ndim != 2:
            raise ValueError(
                "final_logits must have shape [batch, vocab]; "
                f"got {tuple(final_logits.shape)}"
            )
        if early_logits.shape[0] != final_logits.shape[0]:
            raise ValueError("early and final logits batch sizes differ")
        if early_logits.shape[-1] != final_logits.shape[-1]:
            raise ValueError("early and final logits vocabulary sizes differ")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1]; got {gamma}")

        layer_weights, details = self.compute_layer_weights(
            early_logits,
            input_embedding_weight,
            return_details=True,
        )
        augmented = (
            early_logits * layer_weights.unsqueeze(-1)
        ).sum(dim=1)
        mixed = (1.0 - gamma) * final_logits + gamma * augmented
        details["layer_weights"] = layer_weights
        details["augmented_logits"] = augmented
        return mixed, details
