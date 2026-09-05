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


def log_unbalanced_sinkhorn(
    cost: torch.Tensor,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    epsilon: float,
    marginal_relaxation: float,
    num_iters: int,
    tolerance: Optional[float] = None,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    r"""Solve reference-KL, dustbin-free UOT.

    The optimized objective is

    ``<C,P> + epsilon KL(P || a tensor-product b)``
    ``+ rho KL(P 1 || a) + rho KL(P^T 1 || b)``.

    Including ``a tensor-product b`` in the Gibbs reference is important for
    the retention interpretation used below: for zero cost the unit-mass plan
    ``a tensor-product b`` is a fixed point, while positive costs may discard
    mass through the relaxed marginals.

    Convergence is measured on the dual updates: unlike balanced OT, an
    unbalanced plan is not expected to reproduce either input marginal.
    """
    if cost.ndim < 2:
        raise ValueError(f"cost must have at least two dimensions; got {cost.ndim}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive; got {epsilon}")
    if marginal_relaxation <= 0:
        raise ValueError(
            "marginal_relaxation must be positive; got "
            f"{marginal_relaxation}"
        )
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
            device=cost.device, dtype=torch.float32,
        )
        target_fp32 = target_marginal.to(
            device=cost.device, dtype=torch.float32,
        )
        tiny = torch.finfo(torch.float32).tiny
        log_source = source_fp32.clamp_min(tiny).log()
        log_target = target_fp32.clamp_min(tiny).log()
        log_kernel = (
            log_source.unsqueeze(-1)
            + log_target.unsqueeze(-2)
            - cost_fp32 / epsilon
        )
        log_u = torch.zeros_like(log_source)
        log_v = torch.zeros_like(log_target)
        exponent = marginal_relaxation / (marginal_relaxation + epsilon)
        final_update = torch.tensor(
            float("inf"), dtype=torch.float32, device=cost.device,
        )
        iterations_used = 0

        for iteration in range(num_iters):
            iterations_used = iteration + 1
            previous_u = log_u
            previous_v = log_v
            log_u = exponent * (
                log_source
                - torch.logsumexp(
                    log_kernel + log_v.unsqueeze(-2), dim=-1,
                )
            )
            log_v = exponent * (
                log_target
                - torch.logsumexp(
                    log_kernel + log_u.unsqueeze(-1), dim=-2,
                )
            )
            should_check = tolerance is not None and (
                (iteration + 1) % 5 == 0 or iteration + 1 == num_iters
            )
            if should_check:
                final_update = torch.maximum(
                    (log_u - previous_u).abs().amax(),
                    (log_v - previous_v).abs().amax(),
                )
                if final_update.item() <= tolerance:
                    break

        plan = (
            log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        ).exp()
        if not return_diagnostics:
            return plan
        return plan, {
            "iterations": torch.tensor(
                float(iterations_used), dtype=torch.float32, device=cost.device,
            ),
            "dual_residual": final_update,
        }


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
        trace_attention: bool = False,
        attention_coverage_beta: float = 0.0,
        attention_coverage_epsilon: float = 0.1,
        adaptive_alpha: bool = False,
        adaptive_alpha_min_ratio: float = 0.25,
        recall_reward_lambda: float = 0.0,
        recall_candidate_topk: int = 16,
        recall_temperature: float = 0.1,
        recall_coverage_decay: float = 1.0,
        recall_recovery_rho: float = 0.0,
        unbalanced: bool = False,
        marginal_relaxation: float = 0.5,
        mass_aware_layer_weights: bool = False,
        direction_aware_gating: bool = False,
        independent_uniform_layer_weights: bool = False,
        mass_centered_direction_gating: bool = False,
        bidirectional_timestep_gating: bool = False,
        shared_candidate_set: bool = False,
        final_norm_alignment: bool = False,
        head_aware_mode: str = "none",
        head_topk: int = 4,
        head_temperature: float = 0.1,
        head_uniform_mix: float = 0.0,
        head_mass_weight: float = 0.1,
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
        if attention_coverage_beta < 0:
            raise ValueError("attention_coverage_beta must be non-negative")
        if attention_coverage_epsilon <= 0:
            raise ValueError("attention_coverage_epsilon must be positive")
        if not 0.0 <= adaptive_alpha_min_ratio <= 1.0:
            raise ValueError("adaptive_alpha_min_ratio must be in [0, 1]")
        if recall_reward_lambda < 0:
            raise ValueError("recall_reward_lambda must be non-negative")
        if recall_candidate_topk <= 0:
            raise ValueError("recall_candidate_topk must be positive")
        if recall_temperature <= 0:
            raise ValueError("recall_temperature must be positive")
        if recall_coverage_decay < 0:
            raise ValueError("recall_coverage_decay must be non-negative")
        if not 0.0 <= recall_recovery_rho <= 1.0:
            raise ValueError("recall_recovery_rho must be in [0, 1]")
        if marginal_relaxation <= 0:
            raise ValueError("marginal_relaxation must be positive")
        if mass_aware_layer_weights and not unbalanced:
            raise ValueError("Mass-aware layer weights require unbalanced OT")
        if direction_aware_gating and not mass_aware_layer_weights:
            raise ValueError(
                "Direction-aware gating requires mass-aware layer weights"
            )
        if direction_aware_gating and not attention_visual_marginal:
            raise ValueError(
                "Direction-aware gating requires the attention visual marginal"
            )
        if independent_uniform_layer_weights and not direction_aware_gating:
            raise ValueError(
                "Independent uniform layer weights require direction-aware gating"
            )
        if mass_centered_direction_gating and not direction_aware_gating:
            raise ValueError(
                "Mass-centered direction gating requires direction-aware gating"
            )
        if bidirectional_timestep_gating and not mass_centered_direction_gating:
            raise ValueError(
                "Bidirectional timestep gating requires mass-centered "
                "direction-aware gating"
            )
        if bidirectional_timestep_gating and not shared_candidate_set:
            raise ValueError(
                "Bidirectional timestep gating requires a shared candidate set"
            )
        if bidirectional_timestep_gating and not final_norm_alignment:
            raise ValueError(
                "Bidirectional timestep gating requires final-norm alignment"
            )
        if head_aware_mode not in {
            "none", "mass", "topmass", "uot", "uot_uniform",
        }:
            raise ValueError(f"Unknown head-aware mode: {head_aware_mode}")
        if head_aware_mode != "none" and not attention_visual_marginal:
            raise ValueError(
                "Head-aware OT requires the attention visual marginal"
            )
        if head_topk <= 0:
            raise ValueError("head_topk must be positive")
        if head_temperature <= 0:
            raise ValueError("head_temperature must be positive")
        if not 0.0 <= head_uniform_mix < 1.0:
            raise ValueError("head_uniform_mix must be in [0, 1)")
        if head_aware_mode != "uot_uniform" and head_uniform_mix != 0.0:
            raise ValueError(
                "head_uniform_mix is only valid in uot_uniform mode"
            )
        if head_mass_weight < 0:
            raise ValueError("head_mass_weight must be non-negative")
        if direction_aware_gating and (
            adaptive_alpha
            or recall_reward_lambda > 0
            or recall_recovery_rho > 0
        ):
            raise ValueError(
                "Direction-aware gating is a clean standalone ablation and "
                "cannot be combined with adaptive alpha or recall recovery"
            )
        if recall_reward_lambda > 0 and recall_recovery_rho > 0:
            raise ValueError(
                "Additive recall reward and bounded recall recovery are "
                "mutually exclusive"
            )

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
        self.trace_attention = trace_attention
        self.attention_coverage_beta = attention_coverage_beta
        self.attention_coverage_epsilon = attention_coverage_epsilon
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_alpha_min_ratio = adaptive_alpha_min_ratio
        self.recall_reward_lambda = recall_reward_lambda
        self.recall_candidate_topk = recall_candidate_topk
        self.recall_temperature = recall_temperature
        self.recall_coverage_decay = recall_coverage_decay
        self.recall_recovery_rho = recall_recovery_rho
        self.unbalanced = unbalanced
        self.marginal_relaxation = marginal_relaxation
        self.mass_aware_layer_weights = mass_aware_layer_weights
        self.direction_aware_gating = direction_aware_gating
        self.independent_uniform_layer_weights = independent_uniform_layer_weights
        self.mass_centered_direction_gating = mass_centered_direction_gating
        self.bidirectional_timestep_gating = bidirectional_timestep_gating
        self.shared_candidate_set = shared_candidate_set
        self.final_norm_alignment = final_norm_alignment
        self.head_aware_mode = head_aware_mode
        self.head_topk = head_topk
        self.head_temperature = head_temperature
        self.head_uniform_mix = head_uniform_mix
        self.head_mass_weight = head_mass_weight

        self._visual_extended: Optional[torch.Tensor] = None
        self._visual_local: Optional[torch.Tensor] = None
        self._visual_attention_positions: Optional[torch.Tensor] = None
        self._layer_visual_features: Optional[torch.Tensor] = None
        self._stats: Dict[str, torch.Tensor] = {}
        self._stats_steps = 0
        self._attention_trace: List[Dict[str, object]] = []
        self._attention_coverage: Optional[torch.Tensor] = None

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
        self._attention_trace = []
        self._attention_coverage = None

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
        self._attention_trace = []
        self._attention_coverage = None

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
        if (
            self.attention_coverage_beta > 0
            and self._attention_coverage is not None
        ):
            coverage = self._attention_coverage.to(
                device=source.device, dtype=source.dtype,
            )
            coverage_total = coverage.sum(dim=-1, keepdim=True)
            relative_coverage = torch.where(
                coverage_total > torch.finfo(source.dtype).tiny,
                coverage * visual_tokens / coverage_total,
                torch.ones_like(coverage),
            )
            coverage_correction = (
                relative_coverage + self.attention_coverage_epsilon
            ).pow(-self.attention_coverage_beta)
            source = source * coverage_correction.unsqueeze(1)
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

    def _head_attention_source_marginals(
        self,
        attentions: Sequence[torch.Tensor],
        layer_indices: Sequence[int],
        batch_size: int,
        visual_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized per-head visual marginals and raw visual mass.

        The returned source has shape ``[B, W, H, K]``.  The second return
        value is the pre-normalization visual attention mass ``[B, W, H]``;
        this keeps heads that mostly attend to text from becoming equivalent
        to visually engaged heads merely because each head is normalized.
        """
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
            current = attention.float()[:, :, -1, :]
            samples = [
                current[index, :, positions[index]]
                for index in range(batch_size)
            ]
            counts = {sample.shape[-1] for sample in samples}
            if len(counts) != 1 or next(iter(counts)) != visual_tokens:
                raise ValueError(
                    "Unpooled attention and layer visual-token counts differ"
                )
            raw_weights.append(torch.stack(samples))

        raw = torch.stack(raw_weights, dim=1).clamp_min(0.0)
        visual_mass = raw.sum(dim=-1)
        source = raw.pow(self.attention_power)
        if (
            self.attention_coverage_beta > 0
            and self._attention_coverage is not None
        ):
            coverage = self._attention_coverage.to(
                device=source.device, dtype=source.dtype,
            )
            coverage_total = coverage.sum(dim=-1, keepdim=True)
            relative_coverage = torch.where(
                coverage_total > torch.finfo(source.dtype).tiny,
                coverage * visual_tokens / coverage_total,
                torch.ones_like(coverage),
            )
            correction = (
                relative_coverage + self.attention_coverage_epsilon
            ).pow(-self.attention_coverage_beta)
            source = source * correction[:, None, None, :]
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
        return source, visual_mass

    def _top_candidate_ids(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Return valid vocabulary candidates from logits with arbitrary prefix dims."""
        vocab_size = logits.shape[-1]
        invalid_ids = [
            token_id
            for token_id in self.special_token_ids
            if 0 <= token_id < vocab_size
        ]
        if vocab_size - len(invalid_ids) < topk:
            raise ValueError(
                f"Only {vocab_size - len(invalid_ids)} valid vocabulary tokens "
                f"remain, fewer than requested topk={topk}"
            )

        if invalid_ids:
            ranked_logits = logits.clone()
            invalid = torch.tensor(
                invalid_ids,
                dtype=torch.long,
                device=logits.device,
            )
            ranked_logits.index_fill_(-1, invalid, float("-inf"))
        else:
            ranked_logits = logits
        return ranked_logits.topk(topk, dim=-1).indices

    def _candidate_ids(self, early_logits: torch.Tensor) -> torch.Tensor:
        if self.shared_candidate_set:
            # A shared support makes conditional UOT costs comparable across
            # layers and aligns every layer's visual evidence with its
            # contribution to the augmented logit. Max pooled log-probability
            # retains a token proposed strongly by any selected early layer
            # without allowing layer-specific logit scales to dominate.
            pooled_log_probability = F.log_softmax(
                early_logits.float(), dim=-1,
            ).amax(dim=1)
            shared_ids = self._top_candidate_ids(
                pooled_log_probability, self.topk,
            )
            return shared_ids.unsqueeze(1).expand(
                -1, early_logits.shape[1], -1,
            )
        return self._top_candidate_ids(early_logits, self.topk)

    def _recall_reward(
        self,
        early_logits: torch.Tensor,
        final_logits: torch.Tensor,
        augmented_logits: torch.Tensor,
        layer_weights: torch.Tensor,
        output_embedding_weight: Optional[torch.Tensor],
        previous_coverage: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score candidate words supported by visual patches not yet covered.

        This is intentionally separate from the OT source marginal. OT still
        selects reliable layers from the current attention; this term only
        reranks a bounded union of textual candidates after OT aggregation.
        """
        recall_enabled = (
            self.recall_reward_lambda > 0 or self.recall_recovery_rho > 0
        )
        if not recall_enabled or not self.attention_visual_marginal:
            return torch.zeros_like(final_logits), torch.empty(
                final_logits.shape[0], 0, dtype=torch.long,
                device=final_logits.device,
            )
        if output_embedding_weight is None:
            raise RuntimeError("Recall reward requires lm_head.weight")
        candidate_topk = self.recall_candidate_topk
        candidate_ids = torch.cat(
            [
                self._top_candidate_ids(final_logits, candidate_topk),
                self._top_candidate_ids(augmented_logits, candidate_topk),
                self._top_candidate_ids(early_logits, candidate_topk).flatten(1),
            ],
            dim=1,
        )
        batch_size = final_logits.shape[0]
        visual = self._expanded_layer_visual_features(batch_size).to(
            device=final_logits.device, dtype=torch.float32,
        )
        token_features = _normalize(F.embedding(
            candidate_ids, output_embedding_weight,
        ).float())
        similarity = torch.einsum("bwkd,bnd->bwkn", visual, token_features)

        if previous_coverage is None:
            uncovered = torch.ones(
                batch_size, visual.shape[-2], dtype=torch.float32,
                device=final_logits.device,
            )
        else:
            coverage = previous_coverage.to(
                device=final_logits.device, dtype=torch.float32,
            )
            if coverage.shape != (batch_size, visual.shape[-2]):
                raise ValueError("Recall coverage and layer visual tokens differ")
            uncovered = torch.exp(-self.recall_coverage_decay * coverage)

        patch_distribution = torch.softmax(
            similarity / self.recall_temperature, dim=-2,
        )
        if self.recall_recovery_rho > 0:
            # Zero cosine is neutral, not positive visual evidence. Keeping
            # only positive similarity prevents the bounded path from acting
            # like an unconditional interpolation back toward uniform SLA.
            visual_support = similarity.clamp(min=0.0, max=1.0)
        else:
            # Preserve the original additive-reward experiment semantics.
            visual_support = (similarity.clamp(-1.0, 1.0) + 1.0) * 0.5
        reward_per_layer = (
            patch_distribution * visual_support * uncovered[:, None, :, None]
        ).sum(dim=-2)
        if self.recall_recovery_rho > 0:
            # Recovery must provide evidence independent of the OT-selected
            # layer distribution; otherwise the same confirmation loop that
            # caused suppression also gates the attempted recovery. The
            # uniform layer reference below supplies a conservative ceiling.
            reward = reward_per_layer.mean(dim=1)
        else:
            reward = (
                reward_per_layer * layer_weights.float().unsqueeze(-1)
            ).sum(dim=1)

        # A token can appear in several source top-k lists. Use max rather
        # than accumulation so duplicated candidates are not favored. This
        # deliberately avoids Tensor.scatter_reduce_, which is unavailable in
        # older PyTorch versions used by several VISTA environments.
        vocab_reward = torch.zeros_like(final_logits, dtype=torch.float32)
        for candidate_index in range(candidate_ids.shape[1]):
            token_ids = candidate_ids[:, candidate_index]
            candidate_reward = reward[:, candidate_index]
            current_reward = vocab_reward.gather(
                dim=-1, index=token_ids.unsqueeze(-1),
            ).squeeze(-1)
            vocab_reward.scatter_(
                dim=-1,
                index=token_ids.unsqueeze(-1),
                src=torch.maximum(current_reward, candidate_reward).unsqueeze(-1),
            )
        return vocab_reward.to(dtype=final_logits.dtype), candidate_ids

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

    def _update_recall_stats(
        self,
        support: torch.Tensor,
        recovery: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> None:
        """Accumulate compact diagnostics for a recall-enabled generation."""
        if not self.log_stats or candidate_ids.numel() == 0:
            return
        candidate_support = support.gather(dim=-1, index=candidate_ids)
        values = {
            "recall_candidate_support": candidate_support.float().mean(),
            "recall_recovery_mass": recovery.float().sum(dim=-1).mean(),
            "recall_recovered_token_count": (
                recovery > 0
            ).float().sum(dim=-1).mean(),
        }
        for name, value in values.items():
            if name not in self._stats:
                self._stats[name] = value.detach().clone()
            else:
                self._stats[name].add_(value.detach())

    def get_diagnostics(self) -> Dict[str, Union[int, float, List[float]]]:
        """Return generation-level statistics, synchronizing only once."""
        if (
            (not self.log_stats or self._stats_steps == 0)
            and not self._attention_trace
        ):
            return {}

        result: Dict[str, Union[int, float, List[float]]] = {}
        if self.log_stats and self._stats_steps:
            result["steps"] = self._stats_steps
            for name, total in self._stats.items():
                mean = (total / self._stats_steps).detach().float().cpu()
                result[f"mean_{name}"] = (
                    mean.item() if mean.ndim == 0 else mean.tolist()
                )
        if self._attention_trace:
            result["attention_trace"] = self._attention_trace
        return result

    def _record_attention_trace(self, details: Dict[str, torch.Tensor]) -> None:
        """Store compact per-step tensors for a single-image diagnosis run."""
        if not self.trace_attention or not self.attention_visual_marginal:
            return
        source = details["source_marginal"].detach().float().cpu()
        weights = details["layer_weights"].detach().float().cpu()
        effective_source = details["effective_source_marginal"].detach().float().cpu()
        trace_entry = {
                "source_marginal": source.tolist(),
                "effective_source_marginal": effective_source.tolist(),
                "layer_weights": weights.tolist(),
                "layer_costs": details["layer_costs"].detach().float().cpu().tolist(),
                "candidate_ids": details["candidate_ids"].detach().cpu().tolist(),
                "adaptive_alpha": details["adaptive_alpha"].detach().float().cpu().tolist(),
        }
        if details["promotion_gate"].numel():
            flat_candidates = details["candidate_ids"].flatten(1)
            trace_entry["candidate_promotion_gate"] = details[
                "promotion_gate"
            ].gather(-1, flat_candidates).detach().float().cpu().tolist()
            trace_entry["candidate_suppression_gate"] = details[
                "suppression_gate"
            ].gather(-1, flat_candidates).detach().float().cpu().tolist()
            trace_entry["uniform_layer_weights"] = details[
                "uniform_layer_weights"
            ].detach().float().cpu().tolist()
            if "timestep_promotion_strength" in details:
                trace_entry["timestep_promotion_strength"] = details[
                    "timestep_promotion_strength"
                ].detach().float().cpu().tolist()
                trace_entry["timestep_suppression_strength"] = details[
                    "timestep_suppression_strength"
                ].detach().float().cpu().tolist()
        if "uot_iterations" in details:
            trace_entry["uot_iterations"] = float(
                details["uot_iterations"].detach().cpu().item()
            )
            trace_entry["uot_dual_residual"] = float(
                details["uot_dual_residual"].detach().cpu().item()
            )
        if "uniform_uot_iterations" in details:
            trace_entry["uniform_uot_iterations"] = float(
                details["uniform_uot_iterations"].detach().cpu().item()
            )
            trace_entry["uniform_uot_dual_residual"] = float(
                details["uniform_uot_dual_residual"].detach().cpu().item()
            )
        self._attention_trace.append(trace_entry)

    def _update_attention_coverage(
        self,
        source_marginal: torch.Tensor,
        layer_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate the layer-weighted visual marginal for future steps."""
        effective_source = (source_marginal * layer_weights.unsqueeze(-1)).sum(dim=1)
        if self.attention_visual_marginal:
            if self._attention_coverage is None:
                self._attention_coverage = torch.zeros_like(effective_source)
            self._attention_coverage.add_(effective_source.detach())
        return effective_source

    def _effective_alpha(
        self,
        logits_alpha: float,
        effective_source: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce the OT mix when visual evidence is concentrated in few patches."""
        alpha = torch.full(
            (effective_source.shape[0],), logits_alpha,
            dtype=effective_source.dtype, device=effective_source.device,
        )
        if not self.adaptive_alpha or effective_source.shape[-1] <= 1:
            return alpha
        entropy = -(
            effective_source
            * effective_source.clamp_min(torch.finfo(effective_source.dtype).tiny).log()
        ).sum(dim=-1)
        coverage_score = entropy / math.log(effective_source.shape[-1])
        scale = self.adaptive_alpha_min_ratio + (
            1.0 - self.adaptive_alpha_min_ratio
        ) * coverage_score
        return alpha * scale

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
            head_sources = None
            head_visual_mass = None
            head_weights = None
            head_selected_indices = None
            head_scores = None
            if self.attention_visual_marginal:
                if self.head_aware_mode == "none":
                    source_marginal = self._attention_source_marginal(
                        attentions=attentions,
                        layer_indices=attention_layer_indices,
                        batch_size=batch_size,
                        visual_tokens=visual_nodes,
                    ).to(device=cost.device)
                else:
                    head_sources, head_visual_mass = (
                        self._head_attention_source_marginals(
                            attentions=attentions,
                            layer_indices=attention_layer_indices,
                            batch_size=batch_size,
                            visual_tokens=visual_nodes,
                        )
                    )
                    head_sources = head_sources.to(device=cost.device)
                    head_visual_mass = head_visual_mass.to(device=cost.device)
                    if self.head_aware_mode == "mass":
                        head_weights = torch.softmax(
                            head_visual_mass.clamp_min(
                                torch.finfo(torch.float32).tiny
                            ).log() / self.head_temperature,
                            dim=-1,
                        )
                        source_marginal = (
                            head_sources * head_weights.unsqueeze(-1)
                        ).sum(dim=2)
                    elif self.head_aware_mode == "topmass":
                        # This is deliberately not a temperature router:
                        # retain only the top-M visual-mass heads, then pool
                        # their *raw* visual mass before one UOT solve.  It
                        # isolates coverage loss from per-head UOT/fusion.
                        head_count = head_sources.shape[2]
                        selected_count = min(self.head_topk, head_count)
                        head_selected_indices = head_visual_mass.topk(
                            selected_count, dim=-1,
                        ).indices
                        selected_sources = torch.gather(
                            head_sources,
                            dim=2,
                            index=head_selected_indices.unsqueeze(-1).expand(
                                -1, -1, -1, visual_nodes,
                            ),
                        )
                        selected_mass = torch.gather(
                            head_visual_mass, dim=2,
                            index=head_selected_indices,
                        )
                        head_weights = selected_mass / selected_mass.sum(
                            dim=-1, keepdim=True,
                        ).clamp_min(torch.finfo(torch.float32).tiny)
                        source_marginal = (
                            selected_sources * head_weights.unsqueeze(-1)
                        ).sum(dim=2)
                    else:
                        # Filled after target marginals are available: UOT
                        # itself supplies the dynamic head reliability.
                        source_marginal = None
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
            # UOT retention uses an exact column-mass / target-mass ratio.
            # Keep every discrete target atom strictly positive even under
            # float32 softmax underflow, then restore unit total mass.
            target_marginal = target_marginal.clamp_min(
                torch.finfo(torch.float32).tiny
            )
            target_marginal = target_marginal / target_marginal.sum(
                dim=-1, keepdim=True,
            )
            transport_cost = cost.clamp(0.0, 2.0) if self.unbalanced else cost
            transport_diagnostics = None
            if self.head_aware_mode in {"uot", "uot_uniform"}:
                if head_sources is None or head_visual_mass is None:
                    raise RuntimeError("Head-UOT requires head attention sources")
                head_count = head_sources.shape[2]
                selected_count = min(self.head_topk, head_count)
                head_selected_indices = head_visual_mass.topk(
                    selected_count, dim=-1,
                ).indices
                selected_sources = torch.gather(
                    head_sources,
                    dim=2,
                    index=head_selected_indices.unsqueeze(-1).expand(
                        -1, -1, -1, visual_nodes,
                    ),
                )
                expanded_cost = transport_cost.unsqueeze(2).expand(
                    -1, -1, selected_count, -1, -1,
                )
                expanded_target = target_marginal.unsqueeze(2).expand(
                    -1, -1, selected_count, -1,
                )
                if self.unbalanced:
                    head_transport_plan, transport_diagnostics = (
                        log_unbalanced_sinkhorn(
                            expanded_cost,
                            selected_sources,
                            expanded_target,
                            epsilon=self.epsilon,
                            marginal_relaxation=self.marginal_relaxation,
                            num_iters=self.sinkhorn_iters,
                            tolerance=self.sinkhorn_tolerance,
                            return_diagnostics=True,
                        )
                    )
                else:
                    head_transport_plan = log_sinkhorn(
                        expanded_cost,
                        selected_sources,
                        expanded_target,
                        epsilon=self.epsilon,
                        num_iters=self.sinkhorn_iters,
                        tolerance=self.sinkhorn_tolerance,
                    )
                head_mass = head_transport_plan.sum(dim=(-2, -1))
                head_cost = (head_transport_plan * expanded_cost).sum(
                    dim=(-2, -1)
                ) / head_mass.clamp_min(torch.finfo(torch.float32).tiny)
                # Conditional cost alone can select an expert that transports
                # almost no mass. The log-mass term is a reliability factor,
                # not a semantic probability.
                head_scores = (
                    -head_cost
                    + self.head_mass_weight * head_mass.clamp_min(
                        torch.finfo(torch.float32).tiny
                    ).log()
                )
                head_weights = torch.softmax(
                    head_scores / self.head_temperature, dim=-1,
                )
                if self.head_aware_mode == "uot_uniform":
                    head_weights = (
                        (1.0 - self.head_uniform_mix) * head_weights
                        + self.head_uniform_mix / selected_count
                    )
                transport_plan = (
                    head_transport_plan * head_weights.unsqueeze(-1).unsqueeze(-1)
                ).sum(dim=2)
                source_marginal = (
                    selected_sources * head_weights.unsqueeze(-1)
                ).sum(dim=2)
            elif self.unbalanced:
                transport_plan, transport_diagnostics = log_unbalanced_sinkhorn(
                    transport_cost,
                    source_marginal,
                    target_marginal,
                    epsilon=self.epsilon,
                    marginal_relaxation=self.marginal_relaxation,
                    num_iters=self.sinkhorn_iters,
                    tolerance=self.sinkhorn_tolerance,
                    return_diagnostics=True,
                )
            else:
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
                local_cost = transport_cost
            else:
                local_plan = transport_plan[..., :-1, :]
                local_cost = transport_cost[..., :-1, :]
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
                layer_evidence = -layer_costs / self.layer_temperature
                if self.mass_aware_layer_weights:
                    # A low conditional cost is unreliable when UOT retains
                    # only negligible mass. Log mass provides a dimensionless
                    # multiplicative reliability correction; it is not
                    # treated as a calibrated semantic probability.
                    layer_evidence = layer_evidence + local_mass.clamp_min(
                        torch.finfo(torch.float32).tiny
                    ).log()
                layer_weights_fp32 = torch.softmax(layer_evidence, dim=-1)

            uniform_transport_plan = None
            uniform_layer_weights_fp32 = None
            uniform_transport_diagnostics = None
            if self.direction_aware_gating:
                uniform_source = torch.full(
                    (1, 1, visual_nodes),
                    1.0 / visual_nodes,
                    dtype=torch.float32,
                    device=cost.device,
                )
                (
                    uniform_transport_plan,
                    uniform_transport_diagnostics,
                ) = log_unbalanced_sinkhorn(
                    transport_cost,
                    uniform_source,
                    target_marginal,
                    epsilon=self.epsilon,
                    marginal_relaxation=self.marginal_relaxation,
                    num_iters=self.sinkhorn_iters,
                    tolerance=self.sinkhorn_tolerance,
                    return_diagnostics=True,
                )
                uniform_mass = uniform_transport_plan.sum(dim=(-2, -1))
                uniform_costs = (
                    uniform_transport_plan * transport_cost
                ).sum(dim=(-2, -1)) / uniform_mass.clamp_min(
                    torch.finfo(torch.float32).tiny
                )
                if self.force_uniform:
                    uniform_layer_weights_fp32 = torch.full_like(
                        uniform_costs, 1.0 / layer_count,
                    )
                else:
                    uniform_evidence = (
                        -uniform_costs / self.layer_temperature
                        + uniform_mass.clamp_min(
                            torch.finfo(torch.float32).tiny
                        ).log()
                    )
                    uniform_layer_weights_fp32 = torch.softmax(
                        uniform_evidence, dim=-1,
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
        if head_weights is not None:
            head_entropy = -(
                head_weights * head_weights.clamp_min(
                    torch.finfo(torch.float32).tiny
                ).log()
            ).sum(dim=-1)
            details["head_weights"] = head_weights
            details["head_effective_count"] = head_entropy.exp()
            details["head_visual_mass"] = (
                (
                    torch.gather(
                        head_visual_mass,
                        dim=2,
                        index=head_selected_indices,
                    )
                    if head_selected_indices is not None
                    else head_visual_mass
                ) * head_weights
            ).sum(dim=-1)
            if head_selected_indices is not None:
                details["head_selected_indices"] = head_selected_indices
            if head_scores is not None:
                details["head_scores"] = head_scores
        if uniform_transport_plan is not None:
            details["uniform_transport_plan"] = uniform_transport_plan
            details["uniform_layer_weights"] = uniform_layer_weights_fp32.to(
                dtype=early_logits.dtype,
            )
        if transport_diagnostics is not None:
            details["uot_iterations"] = transport_diagnostics["iterations"]
            details["uot_dual_residual"] = transport_diagnostics[
                "dual_residual"
            ]
        if uniform_transport_diagnostics is not None:
            details["uniform_uot_iterations"] = uniform_transport_diagnostics[
                "iterations"
            ]
            details["uniform_uot_dual_residual"] = (
                uniform_transport_diagnostics["dual_residual"]
            )
        return layer_weights, details

    @staticmethod
    def _scatter_candidate_retention(
        candidate_ids: torch.Tensor,
        retention: torch.Tensor,
        layer_weights: torch.Tensor,
        vocab_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate per-layer candidate retention without full-vocab costs."""
        batch_size = candidate_ids.shape[0]
        indices = candidate_ids.reshape(batch_size, -1)
        weighted_retention = (
            retention.float() * layer_weights.float().unsqueeze(-1)
        ).reshape(batch_size, -1)
        presence_weight = layer_weights.float().unsqueeze(-1).expand_as(
            retention
        ).reshape(batch_size, -1)
        numerator = torch.zeros(
            batch_size, vocab_size, dtype=torch.float32,
            device=candidate_ids.device,
        )
        denominator = torch.zeros_like(numerator)
        numerator.scatter_add_(dim=-1, index=indices, src=weighted_retention)
        denominator.scatter_add_(dim=-1, index=indices, src=presence_weight)
        present = denominator > 0
        score = torch.where(
            present,
            numerator / denominator.clamp_min(
                torch.finfo(torch.float32).tiny
            ),
            torch.zeros_like(numerator),
        )
        return score.clamp(0.0, 1.0), present

    def _direction_aware_mix(
        self,
        final_logits: torch.Tensor,
        augmented_logits: torch.Tensor,
        layer_weights: torch.Tensor,
        details: Dict[str, torch.Tensor],
        logits_alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply separate visual gates to promotion and suppression.

        Current-attention retention licenses promotion.  Whole-image uniform
        retention protects a candidate from suppression when a matching patch
        exists outside the current attention focus.
        """
        target = details["target_marginal"].float()
        tiny = torch.finfo(torch.float32).tiny
        attention_retention = (
            details["transport_plan"].float().sum(dim=-2)
            / target
        ).clamp_min(0.0)
        uniform_retention = (
            details["uniform_transport_plan"].float().sum(dim=-2)
            / target
        ).clamp_min(0.0)
        attention_mass = details["transport_plan"].float().sum(
            dim=(-2, -1), keepdim=False,
        ).unsqueeze(-1).clamp(0.0, 1.0)
        uniform_mass = details["uniform_transport_plan"].float().sum(
            dim=(-2, -1), keepdim=False,
        ).unsqueeze(-1).clamp(0.0, 1.0)
        details["attention_retention"] = attention_retention
        details["uniform_retention"] = uniform_retention
        details["attention_retention_abs_deviation"] = (
            target * (attention_retention - attention_mass).abs()
        ).sum(dim=-1)
        details["uniform_retention_abs_deviation"] = (
            target * (uniform_retention - uniform_mass).abs()
        ).sum(dim=-1)
        details["attention_uniform_retention_gap"] = (
            target * (attention_retention - uniform_retention).abs()
        ).sum(dim=-1)

        if self.mass_centered_direction_gating:
            # In UOT, sum_m b_m r_m equals the transported mass.  Subtracting
            # that identity-derived baseline removes the global shrinkage set
            # mainly by rho, leaving only candidate-specific evidence.
            attention_evidence = (
                (attention_retention - attention_mass).clamp_min(0.0)
                / (1.0 - attention_mass).clamp_min(tiny)
            ).clamp(0.0, 1.0)
            uniform_absence = (
                (uniform_mass - uniform_retention).clamp_min(0.0)
                / uniform_mass.clamp_min(tiny)
            ).clamp(0.0, 1.0)
        else:
            # Preserve the legacy raw-gate behavior independently of the
            # mathematically unbounded UOT retention ratio.
            attention_evidence = attention_retention.clamp(0.0, 1.0)
            uniform_absence = 1.0 - uniform_retention.clamp(0.0, 1.0)
        attention_support, present = self._scatter_candidate_retention(
            details["candidate_ids"], attention_evidence,
            layer_weights, final_logits.shape[-1],
        )
        uniform_absence_support, _ = self._scatter_candidate_retention(
            details["candidate_ids"], uniform_absence,
            (
                details["uniform_layer_weights"]
                if self.independent_uniform_layer_weights
                else layer_weights
            ),
            final_logits.shape[-1],
        )
        promotion_gate = attention_support
        suppression_gate = torch.where(
            present,
            uniform_absence_support,
            torch.zeros_like(uniform_absence_support),
        )
        delta = augmented_logits.float() - final_logits.float()
        positive_intervention = promotion_gate * delta.clamp_min(0.0)
        negative_intervention = suppression_gate * (-delta).clamp_min(0.0)
        if self.bidirectional_timestep_gating:
            # Restrict final-logit probabilities to candidates represented by
            # OT. Positive and negative evidence must remain separate: a
            # likely hallucinated candidate can have no promotion evidence yet
            # still need whole-image absence suppression.
            candidate_logits = final_logits.float().masked_fill(
                ~present, float("-inf"),
            )
            candidate_probability = torch.softmax(candidate_logits, dim=-1)
            timestep_promotion_strength = (
                candidate_probability * promotion_gate
            ).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            timestep_suppression_strength = (
                candidate_probability * suppression_gate
            ).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            intervention = (
                timestep_promotion_strength * positive_intervention
                - timestep_suppression_strength * negative_intervention
            )
            details["timestep_promotion_strength"] = (
                timestep_promotion_strength.squeeze(-1)
            )
            details["timestep_suppression_strength"] = (
                timestep_suppression_strength.squeeze(-1)
            )
        else:
            intervention = positive_intervention - negative_intervention
        mixed = final_logits.float() + logits_alpha * intervention
        return (
            mixed.to(dtype=final_logits.dtype),
            promotion_gate,
            suppression_gate,
        )

    @torch.no_grad()
    def aggregate(
        self,
        early_logits: torch.Tensor,
        final_logits: torch.Tensor,
        input_embedding_weight: torch.Tensor,
        logits_alpha: float,
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
        if not 0.0 <= logits_alpha <= 1.0:
            raise ValueError(
                "logits_alpha must be in [0, 1]; "
                f"got {logits_alpha}"
            )

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
        previous_coverage = self._attention_coverage
        if self.attention_visual_marginal:
            effective_source = (
                details["source_marginal"] * layer_weights.unsqueeze(-1)
            ).sum(dim=1)
        else:
            effective_source = torch.empty(
                early_logits.shape[0], 0,
                dtype=early_logits.dtype, device=early_logits.device,
            )
        effective_alpha = self._effective_alpha(logits_alpha, effective_source)
        promotion_gate = torch.empty(
            final_logits.shape[0], 0,
            dtype=torch.float32, device=final_logits.device,
        )
        suppression_gate = promotion_gate
        if self.direction_aware_gating:
            mixed, promotion_gate, suppression_gate = self._direction_aware_mix(
                final_logits=final_logits,
                augmented_logits=augmented,
                layer_weights=layer_weights,
                details=details,
                logits_alpha=logits_alpha,
            )
        else:
            mixed = (
                (1.0 - effective_alpha.unsqueeze(-1)) * final_logits
                + effective_alpha.unsqueeze(-1) * augmented
            )
        recall_reward, recall_candidate_ids = self._recall_reward(
            early_logits=early_logits,
            final_logits=final_logits,
            augmented_logits=augmented,
            layer_weights=layer_weights,
            output_embedding_weight=output_embedding_weight,
            previous_coverage=previous_coverage,
        )
        pre_recovery_logits = mixed
        mixed = mixed + self.recall_reward_lambda * recall_reward
        uniform_augmented = early_logits.mean(dim=1)
        uniform_reference = (
            (1.0 - effective_alpha.unsqueeze(-1)) * final_logits
            + effective_alpha.unsqueeze(-1) * uniform_augmented
        )
        suppressed_by_ot = (uniform_reference - mixed).clamp_min(0)
        recall_recovery = recall_reward * suppressed_by_ot
        mixed = mixed + self.recall_recovery_rho * recall_recovery
        self._update_recall_stats(
            recall_reward,
            self.recall_recovery_rho * recall_recovery,
            recall_candidate_ids,
        )
        if self.attention_visual_marginal:
            self._update_attention_coverage(
                details["source_marginal"], layer_weights,
            )
        details["layer_weights"] = layer_weights
        details["augmented_logits"] = augmented
        details["effective_source_marginal"] = effective_source
        details["adaptive_alpha"] = effective_alpha
        details["recall_reward"] = recall_reward
        details["recall_candidate_ids"] = recall_candidate_ids
        details["pre_recovery_logits"] = pre_recovery_logits
        details["uniform_reference_logits"] = uniform_reference
        details["recall_recovery"] = recall_recovery
        details["promotion_gate"] = promotion_gate
        details["suppression_gate"] = suppression_gate
        if self.log_stats and "uot_iterations" in details:
            solver_values = {
                "uot_iterations": details["uot_iterations"].float(),
                "uot_dual_residual": details["uot_dual_residual"].float(),
            }
            if "uniform_uot_iterations" in details:
                solver_values.update({
                    "uniform_uot_iterations": details[
                        "uniform_uot_iterations"
                    ].float(),
                    "uniform_uot_dual_residual": details[
                        "uniform_uot_dual_residual"
                    ].float(),
                })
            for name, value in solver_values.items():
                if name not in self._stats:
                    self._stats[name] = value.detach().clone()
                else:
                    self._stats[name].add_(value.detach())
        if self.log_stats and promotion_gate.numel():
            flat_candidates = details["candidate_ids"].flatten(1)
            direction_values = {
                "candidate_promotion_gate": promotion_gate.gather(
                    -1, flat_candidates,
                ).float().mean(),
                "candidate_suppression_gate": suppression_gate.gather(
                    -1, flat_candidates,
                ).float().mean(),
                "uniform_transport_mass": details[
                    "uniform_transport_plan"
                ].float().sum(dim=(-2, -1)).mean(),
                "uniform_layer_weights": details[
                    "uniform_layer_weights"
                ].float().mean(dim=0),
                "attention_retention_abs_deviation": details[
                    "attention_retention_abs_deviation"
                ].float().mean(),
                "uniform_retention_abs_deviation": details[
                    "uniform_retention_abs_deviation"
                ].float().mean(),
                "attention_uniform_retention_gap": details[
                    "attention_uniform_retention_gap"
                ].float().mean(),
            }
            if "timestep_promotion_strength" in details:
                direction_values.update({
                    "timestep_promotion_strength": details[
                        "timestep_promotion_strength"
                    ].float().mean(),
                    "timestep_suppression_strength": details[
                        "timestep_suppression_strength"
                    ].float().mean(),
                })
            for name, value in direction_values.items():
                if name not in self._stats:
                    self._stats[name] = value.detach().clone()
                else:
                    self._stats[name].add_(value.detach())
        if self.log_stats and "head_weights" in details:
            head_values = {
                "head_effective_count": details["head_effective_count"].float().mean(),
                "head_visual_mass": details["head_visual_mass"].float().mean(),
                "head_max_weight": details["head_weights"].float().amax(dim=-1).mean(),
            }
            for name, value in head_values.items():
                if name not in self._stats:
                    self._stats[name] = value.detach().clone()
                else:
                    self._stats[name].add_(value.detach())
        self._record_attention_trace(details)
        return mixed, details
