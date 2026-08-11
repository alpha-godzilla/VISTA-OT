"""Optimal-transport-guided barycentric Self-Logits Augmentation.

This module is intentionally independent of the LLaVA generation code.  The
The legacy integration caches projected visual tokens. Attention OT additionally
caches per-layer visual hidden states and compares them with ``lm_head`` token
directions while retaining the legacy OT2 behavior behind its original flag.
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
    tolerance: Optional[float] = None,
) -> torch.Tensor:
    """Solve a small entropic OT problem with log-domain Sinkhorn updates.

    Args:
        cost: Tensor with shape ``[..., source_nodes, target_nodes]``.
        source_marginal: Broadcastable tensor with shape ``[..., source_nodes]``.
        target_marginal: Broadcastable tensor with shape ``[..., target_nodes]``.
        epsilon: Positive entropic regularization coefficient.
        num_iters: Maximum number of alternating Sinkhorn updates.
        tolerance: Optional maximum marginal residual for early stopping.

    Returns:
        A float32 transport plan with the same leading dimensions as ``cost``.
    """
    if cost.ndim < 2:
        raise ValueError(f"cost must have at least two dimensions; got {cost.ndim}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive; got {epsilon}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive; got {num_iters}")
    if tolerance is not None and tolerance <= 0:
        raise ValueError(f"tolerance must be positive; got {tolerance}")
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

        log_plan = None
        for iteration in range(num_iters):
            log_u = log_source - torch.logsumexp(
                log_kernel + log_v.unsqueeze(-2),
                dim=-1,
            )
            log_v = log_target - torch.logsumexp(
                log_kernel + log_u.unsqueeze(-1),
                dim=-2,
            )

            should_check = tolerance is not None and (
                (iteration + 1) % 5 == 0 or iteration + 1 == num_iters
            )
            if should_check:
                log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
                plan = log_plan.exp()
                source_error = (
                    plan.sum(dim=-1) - source_fp32
                ).abs().amax()
                target_error = (
                    plan.sum(dim=-2) - target_fp32
                ).abs().amax()
                if torch.maximum(source_error, target_error).item() <= tolerance:
                    return plan

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
        sinkhorn_tolerance: Optional[float] = 1e-3,
        layer_temperature: float = 0.1,
        special_token_ids: Optional[Iterable[int]] = None,
        log_stats: bool = False,
        force_uniform: bool = False,
        attention_visual_marginal: bool = False,
        attention_power: float = 0.5,
        attention_uniform_mix: float = 0.02,
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
        if sinkhorn_tolerance is not None and sinkhorn_tolerance <= 0:
            raise ValueError("sinkhorn_tolerance must be positive")
        if layer_temperature <= 0:
            raise ValueError(
                "layer_temperature must be positive; got "
                f"{layer_temperature}"
            )
        if attention_power <= 0:
            raise ValueError("attention_power must be positive")
        if not 0.0 <= attention_uniform_mix < 1.0:
            raise ValueError("attention_uniform_mix must be in [0, 1)")

        self.topk = topk
        self.visual_tokens = visual_tokens
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tolerance = sinkhorn_tolerance
        self.layer_temperature = layer_temperature
        self.special_token_ids = tuple(
            sorted({int(token_id) for token_id in (special_token_ids or [])})
        )
        self.log_stats = log_stats
        self.force_uniform = force_uniform
        self.attention_visual_marginal = attention_visual_marginal
        self.attention_power = attention_power
        self.attention_uniform_mix = attention_uniform_mix

        self._visual_extended: Optional[torch.Tensor] = None
        self._visual_local: Optional[torch.Tensor] = None
        self._visual_attention_positions: Optional[torch.Tensor] = None
        self._layer_visual_features: Optional[torch.Tensor] = None
        self._stats: Dict[str, torch.Tensor] = {}
        self._stats_steps = 0

    @property
    def has_visual_cache(self) -> bool:
        return self._visual_extended is not None

    def clear(self) -> None:
        self._visual_extended = None
        self._visual_local = None
        self._visual_attention_positions = None
        self._layer_visual_features = None
        self._stats = {}
        self._stats_steps = 0

    @torch.no_grad()
    def cache_visual_features(self, projected_visual_tokens: TensorOrTensors) -> None:
        """Cache projected image features for one generation sequence.

        ``projected_visual_tokens`` can be one ``[B, K0, D]`` tensor or a
        sequence of per-sample ``[Ki, D]`` tensors.  The sequence form supports
        samples whose unpooled visual-token counts differ. Legacy OT2 pools
        these features; attention OT retains all tokens only to validate the
        layer-hidden and attention caches.
        """
        if isinstance(projected_visual_tokens, torch.Tensor):
            if projected_visual_tokens.ndim == 2:
                projected_visual_tokens = projected_visual_tokens.unsqueeze(0)
            local = projected_visual_tokens.detach().float()
            if not self.attention_visual_marginal:
                local = _pool_visual_tokens(local, self.visual_tokens)
            local = _normalize(local)
        else:
            pooled_samples: List[torch.Tensor] = []
            for sample in projected_visual_tokens:
                if sample.ndim != 2:
                    raise ValueError(
                        "Each per-sample visual tensor must have shape "
                        f"[tokens, hidden]; got {tuple(sample.shape)}"
                    )
                pooled_sample = sample.detach().float().unsqueeze(0)
                if not self.attention_visual_marginal:
                    pooled_sample = _pool_visual_tokens(
                        pooled_sample,
                        self.visual_tokens,
                    )
                pooled_sample = pooled_sample.squeeze(0)
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

        self._visual_local = local
        if self.attention_visual_marginal:
            self._visual_extended = local
        else:
            global_feature = _normalize(local.mean(dim=1, keepdim=True))
            self._visual_extended = torch.cat([local, global_feature], dim=1)
        self._stats = {}
        self._stats_steps = 0
        self._layer_visual_features = None

    @torch.no_grad()
    def cache_visual_attention_positions(self, positions: torch.Tensor) -> None:
        """Cache a prefill-sequence boolean mask for original image tokens."""
        if not self.attention_visual_marginal:
            return
        if self._visual_local is None:
            raise RuntimeError("Visual features must be cached before their positions")
        if positions.ndim != 2 or positions.dtype != torch.bool:
            raise ValueError("Visual attention positions must be a [batch, seq] bool mask")
        if positions.shape[0] != self._visual_local.shape[0]:
            raise ValueError("Visual feature and position-mask batch sizes differ")
        counts = positions.sum(dim=-1)
        if not torch.all(counts == counts[0]):
            raise ValueError("Every batch element must contain the same number of visual tokens")
        if counts[0].item() != self._visual_local.shape[1]:
            raise ValueError(
                "Visual position mask and unpooled projected features must "
                "contain the same number of tokens"
            )
        self._visual_attention_positions = positions.detach().clone()

    @property
    def has_layer_visual_cache(self) -> bool:
        return self._layer_visual_features is not None

    @torch.no_grad()
    def cache_layer_visual_features(
        self,
        layer_hidden_states: Sequence[torch.Tensor],
    ) -> None:
        """Cache unpooled hidden states at image positions for each SLA layer."""
        if not self.attention_visual_marginal:
            return
        if self._visual_attention_positions is None:
            raise RuntimeError("Visual token positions must be cached first")
        if not layer_hidden_states:
            raise ValueError("At least one layer hidden state is required")

        batch_size = self._visual_attention_positions.shape[0]
        per_layer = []
        for hidden in layer_hidden_states:
            if hidden.ndim != 3 or hidden.shape[0] != batch_size:
                raise ValueError(
                    "Layer hidden states must have shape [batch, sequence, hidden]"
                )
            positions = self._expanded_attention_positions(
                batch_size,
                hidden.shape[1],
            ).to(device=hidden.device)
            samples = [
                hidden[index][positions[index]]
                for index in range(batch_size)
            ]
            counts = {sample.shape[0] for sample in samples}
            if len(counts) != 1:
                raise ValueError("Visual hidden-state counts differ within the batch")
            per_layer.append(torch.stack(samples).detach().float())

        self._layer_visual_features = _normalize(
            torch.stack(per_layer, dim=1)
        )

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

    def _expanded_layer_visual_features(self, batch_size: int) -> torch.Tensor:
        if self._layer_visual_features is None:
            raise RuntimeError("Layer-aligned visual features are not cached")
        visual = self._layer_visual_features
        if visual.shape[0] == batch_size:
            return visual
        if batch_size % visual.shape[0] != 0:
            raise ValueError("Cannot align layer visual cache with logits batch")
        return visual.repeat_interleave(batch_size // visual.shape[0], dim=0)

    def _expanded_attention_positions(
        self,
        batch_size: int,
        key_length: int,
    ) -> torch.Tensor:
        if self._visual_attention_positions is None:
            raise RuntimeError(
                "Attention OT requires visual token positions cached during "
                "multimodal prefill."
            )
        positions = self._visual_attention_positions
        if positions.shape[0] != batch_size:
            if batch_size % positions.shape[0] != 0:
                raise ValueError("Cannot align visual attention masks with logits batch")
            positions = positions.repeat_interleave(batch_size // positions.shape[0], dim=0)
        if positions.shape[-1] > key_length:
            raise ValueError("Attention key length is shorter than the cached prefill")
        if positions.shape[-1] < key_length:
            positions = F.pad(positions, (0, key_length - positions.shape[-1]))
        return positions

    def _attention_source_marginal(
        self,
        attentions: Sequence[torch.Tensor],
        layer_indices: Sequence[int],
        batch_size: int,
        visual_tokens: int,
    ) -> torch.Tensor:
        """Turn current-query visual attention into per-layer OT marginals."""
        if len(attentions) <= max(layer_indices):
            raise ValueError("Attention outputs do not cover the requested SLA layers")
        layer_attention = [attentions[index] for index in layer_indices]
        if any(attention is None for attention in layer_attention):
            raise RuntimeError("Attention OT requires output_attentions=True")
        key_length = layer_attention[0].shape[-1]
        positions = self._expanded_attention_positions(batch_size, key_length).to(
            device=layer_attention[0].device,
        )
        raw_weights = []
        for attention in layer_attention:
            if attention.ndim != 4 or attention.shape[0] != batch_size:
                raise ValueError("Each attention tensor must have shape [batch, heads, query, key]")
            if attention.shape[-1] != key_length:
                raise ValueError("Attention key lengths differ across SLA layers")
            # Average raw head probabilities first. This preserves the lower
            # contribution of heads that do not attend to the image at all.
            current = attention.float().mean(dim=1)[:, -1, :]
            samples = [current[index][positions[index]] for index in range(batch_size)]
            counts = {sample.numel() for sample in samples}
            if len(counts) != 1:
                raise ValueError("Visual attention token counts differ within the batch")
            visual_attention = torch.stack(samples)
            if visual_attention.shape[-1] != visual_tokens:
                raise ValueError(
                    "Unpooled attention and layer visual-token counts differ"
                )
            raw_weights.append(visual_attention)

        source = torch.stack(raw_weights, dim=1)
        source = source.clamp_min(0).pow(self.attention_power)
        source_total = source.sum(dim=-1, keepdim=True)
        source = torch.where(
            source_total > torch.finfo(source.dtype).tiny,
            source / source_total.clamp_min(torch.finfo(source.dtype).tiny),
            torch.full_like(source, 1.0 / visual_tokens),
        )
        if self.attention_uniform_mix:
            source = (
                (1.0 - self.attention_uniform_mix) * source
                + self.attention_uniform_mix / visual_tokens
            )
        return source

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
        layer_costs: torch.Tensor,
        layer_weights: torch.Tensor,
        transport_plan: torch.Tensor,
    ) -> None:
        if not self.log_stats:
            return

        # Legacy OT has a visual dustbin. Attention OT transports only between
        # real visual and text nodes.
        local_plan = (
            transport_plan
            if self.attention_visual_marginal
            else transport_plan[..., :-1, :]
        )
        entropy = -(
            layer_weights.float()
            * layer_weights.float().clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1)
        values = {
            "layer_costs": layer_costs.float().mean(dim=0),
            "layer_weights": layer_weights.float().mean(dim=0),
            "layer_weight_entropy": entropy.mean(),
            "local_transport_mass": local_plan.sum(dim=(-2, -1)).mean(),
        }
        if not self.attention_visual_marginal:
            values["dustbin_to_token_mass"] = transport_plan[..., -1, :].sum(
                dim=-1
            ).mean()
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
        attentions: Optional[Sequence[torch.Tensor]] = None,
        attention_layer_indices: Optional[Sequence[int]] = None,
        output_embedding_weight: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute dynamic weights for ``early_logits``.

        Args:
            early_logits: ``[B, w, vocab_size]``.
            input_embedding_weight: Legacy OT2 token embeddings.
            output_embedding_weight: Attention OT ``lm_head.weight`` rows.
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
        text_embedding_weight = input_embedding_weight
        visual = self._expanded_visual_features(batch_size).to(
            device=early_logits.device, dtype=torch.float32,
        )
        if self.attention_visual_marginal:
            if attentions is None or attention_layer_indices is None:
                raise RuntimeError(
                    "Attention OT requires per-layer decoder attention outputs"
                )
            if len(attention_layer_indices) != layer_count:
                raise ValueError("Number of attention layers must match early logits")
            if output_embedding_weight is None:
                raise RuntimeError("Attention OT requires lm_head.weight")
            if output_embedding_weight.ndim != 2:
                raise ValueError("output_embedding_weight must have shape [vocab, hidden]")
            if output_embedding_weight.shape[0] < early_logits.shape[-1]:
                raise ValueError("Output embedding vocabulary is smaller than logits")
            text_embedding_weight = output_embedding_weight
            visual = self._expanded_layer_visual_features(batch_size).to(
                device=early_logits.device,
                dtype=torch.float32,
            )
            if visual.shape[1] != layer_count:
                raise ValueError("Layer visual cache and early logits layer counts differ")
        if visual.shape[-1] != text_embedding_weight.shape[-1]:
            raise ValueError(
                "Visual hidden-state and token-embedding "
                f"dimension differ: {visual.shape[-1]} vs "
                f"{text_embedding_weight.shape[-1]}"
            )

        candidate_ids = self._candidate_ids(early_logits)
        with torch.autocast(device_type=early_logits.device.type, enabled=False):
            candidate_features = F.embedding(
                candidate_ids,
                text_embedding_weight,
            ).float()
            text_local = _normalize(candidate_features)

            if self.attention_visual_marginal:
                similarity = torch.einsum(
                    "bwkd,bwmd->bwkm", visual, text_local,
                )
            else:
                similarity = torch.einsum(
                    "bkd,bwmd->bwkm", visual, text_local,
                )
            cost = 1.0 - similarity
            visual_nodes = visual.shape[-2]
            if self.attention_visual_marginal:
                source_marginal = self._attention_source_marginal(
                    attentions=attentions,
                    layer_indices=attention_layer_indices,
                    batch_size=batch_size,
                    visual_tokens=visual_nodes,
                ).to(device=cost.device)
            else:
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
                tolerance=self.sinkhorn_tolerance,
            )

            if self.attention_visual_marginal:
                local_plan = transport_plan
                local_cost = cost
            else:
                local_plan = transport_plan[..., :-1, :]
                local_cost = cost[..., :-1, :]
            # In the legacy method this normalizes away variable global-node
            # mass. Attention OT has no dustbin and its local mass is one.
            local_mass = local_plan.sum(dim=(-2, -1))
            layer_costs = (local_plan * local_cost).sum(dim=(-2, -1))
            layer_costs = layer_costs / local_mass.clamp_min(
                torch.finfo(torch.float32).tiny
            )
            if self.force_uniform:
                layer_weights_fp32 = torch.full_like(
                    layer_costs,
                    1.0 / layer_count,
                )
            else:
                layer_weights_fp32 = torch.softmax(
                    -layer_costs / self.layer_temperature,
                    dim=-1,
                )

        self._update_stats(layer_costs, layer_weights_fp32, transport_plan)
        layer_weights = layer_weights_fp32.to(dtype=early_logits.dtype)
        if not return_details:
            return layer_weights

        details = {
            "candidate_ids": candidate_ids,
            "similarity": similarity,
            "transport_plan": transport_plan,
            "source_marginal": source_marginal,
            "target_marginal": target_marginal,
            "layer_costs": layer_costs,
            # Kept as a compatibility alias for consumers of older diagnostic
            # details; it now denotes negative mean OT cost.
            "layer_scores": -layer_costs,
            "local_transport_mass": local_mass,
        }
        return layer_weights, details

    @torch.no_grad()
    def aggregate(
        self,
        early_logits: torch.Tensor,
        final_logits: torch.Tensor,
        input_embedding_weight: torch.Tensor,
        gamma: float,
        attentions: Optional[Sequence[torch.Tensor]] = None,
        attention_layer_indices: Optional[Sequence[int]] = None,
        output_embedding_weight: Optional[torch.Tensor] = None,
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
            attentions=attentions,
            attention_layer_indices=attention_layer_indices,
            output_embedding_weight=output_embedding_weight,
            return_details=True,
        )
        augmented = (
            early_logits * layer_weights.unsqueeze(-1)
        ).sum(dim=1)
        mixed = (1.0 - gamma) * final_logits + gamma * augmented
        details["layer_weights"] = layer_weights
        details["augmented_logits"] = augmented
        return mixed, details
