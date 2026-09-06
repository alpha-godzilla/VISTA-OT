"""Non-invasive tracing of visual/prompt/generated attention retrieval.

The tracer is intentionally implemented with module hooks.  It never replaces
``LlamaAttention.forward`` and therefore does not switch SDPA/FlashAttention
to eager attention.  When detached from a model it has zero inference cost.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


METRIC_NAMES = (
    "C_V", "C_P", "C_G", "L_GV", "compat_GV", "compat_PV",
    "query_effect", "new_K_effect", "m_V", "m_P", "m_G",
    "reconstruction_error", "L_GV_identity_error",
)
VALID_NAMES = ("has_V", "has_P", "has_G", "query_effect_valid", "new_K_effect_valid")


def _safe_group_logmeanexp(logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (log mean exp, nonempty) without NaNs for an empty group."""
    count = mask.sum()
    if int(count.item()) == 0:
        return logits.new_zeros(logits.shape[:-1]), torch.zeros((), dtype=torch.bool, device=logits.device)
    return torch.logsumexp(logits[..., mask], dim=-1) - count.float().log(), torch.ones((), dtype=torch.bool, device=logits.device)


def retrieval_group_metrics(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    visual_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    generated_mask: torch.Tensor,
    original_output: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Exact grouped softmax decomposition for one query per head.

    Args use ``[H, D]`` query and ``[H, K, D]`` key/value tensors.  Empty
    groups are represented by zero-valued metrics plus explicit validity flags,
    never NaN.
    """
    if query.ndim != 2 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("Expected q=[H,D], k/v=[H,K,D]")
    if keys.shape[1] != visual_mask.numel() or keys.shape[1] != prompt_mask.numel() or keys.shape[1] != generated_mask.numel():
        raise ValueError("Group masks must cover the complete KV sequence")
    logits = torch.einsum("hd,hkd->hk", query.float(), keys.float()) / math.sqrt(query.shape[-1])
    c_v, has_v = _safe_group_logmeanexp(logits, visual_mask)
    c_p, has_p = _safe_group_logmeanexp(logits, prompt_mask)
    c_g, has_g = _safe_group_logmeanexp(logits, generated_mask)
    lse_v = c_v + visual_mask.sum().float().clamp_min(1).log()
    lse_g = c_g + generated_mask.sum().float().clamp_min(1).log()
    logits_prob = torch.softmax(logits, dim=-1)

    def group_readout(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mass = logits_prob[:, mask].sum(dim=-1)
        if int(mask.sum().item()) == 0:
            return mass, values.new_zeros((values.shape[0], values.shape[-1]), dtype=torch.float32)
        conditional = logits_prob[:, mask] / mass.clamp_min(torch.finfo(logits_prob.dtype).tiny).unsqueeze(-1)
        return mass, torch.einsum("hk,hkd->hd", conditional.float(), values[:, mask].float())

    m_v, mu_v = group_readout(visual_mask)
    m_p, mu_p = group_readout(prompt_mask)
    m_g, mu_g = group_readout(generated_mask)
    reconstructed = m_v.unsqueeze(-1) * mu_v + m_p.unsqueeze(-1) * mu_p + m_g.unsqueeze(-1) * mu_g
    if original_output is None:
        original_output = torch.einsum("hk,hkd->hd", logits_prob.float(), values.float())
    reconstruction_error = (original_output.float() - reconstructed).norm(dim=-1)
    valid_gv = has_g & has_v
    l_gv = torch.where(valid_gv, lse_g - lse_v, torch.zeros_like(c_v))
    compat_gv = torch.where(valid_gv, c_g - c_v, torch.zeros_like(c_v))
    identity_rhs = (
        (generated_mask.sum().float().clamp_min(1) / visual_mask.sum().float().clamp_min(1)).log()
        + compat_gv
    )
    return {
        "C_V": c_v, "C_P": c_p, "C_G": c_g,
        "L_GV": l_gv,
        "compat_GV": compat_gv,
        "compat_PV": torch.where(has_p & has_v, c_p - c_v, torch.zeros_like(c_v)),
        "m_V": m_v, "m_P": m_p, "m_G": m_g,
        "reconstruction_error": reconstruction_error,
        "L_GV_identity_error": torch.where(valid_gv, (l_gv - identity_rhs).abs(), torch.zeros_like(c_v)),
        "has_V": has_v.expand_as(c_v), "has_P": has_p.expand_as(c_v), "has_G": has_g.expand_as(c_v),
        "logits": logits,
    }


@dataclass
class _ForwardContext:
    prefill: bool
    token_id: int
    token_text: str
    timestep: int


class RetrievalShiftTracer:
    """Hook-based per-layer/head retrieval-shift measurement for batch size one."""

    def __init__(self, model, tokenizer, output_dir: Path, debug: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.debug = self._parse_debug(debug)
        self.handles = []
        self.pending: Optional[_ForwardContext] = None
        self.prefix_visual_mask: Optional[torch.Tensor] = None
        self.prefix_length: Optional[int] = None
        self.sample_id: Optional[int] = None
        self.rows = []
        self.token_rows = []
        self.state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.prev_query: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _parse_debug(value):
        if not value:
            return None
        # sample_id:layer:head:timestep, all fields are exact selectors.
        parts = value.split(":")
        if len(parts) != 4:
            raise ValueError("--retrieval-shift-debug must be sample_id:layer:head:timestep")
        return tuple(int(x) for x in parts)

    def install(self):
        if self.handles:
            raise RuntimeError("Tracer is already installed")
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            self.handles.append(attn.register_forward_pre_hook(self._attention_pre_hook(layer_idx), with_kwargs=True))
            self.handles.append(attn.register_forward_hook(self._attention_post_hook(layer_idx), with_kwargs=True))
            self.handles.append(attn.o_proj.register_forward_pre_hook(self._output_pre_hook(layer_idx)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.state.clear()

    def start_sample(self, sample_id: int):
        if self.sample_id is not None:
            raise RuntimeError("finish_sample must be called before start_sample")
        self.sample_id = int(sample_id)
        self.rows = []
        self.token_rows = []
        self.prefix_visual_mask = None
        self.prefix_length = None
        self.prev_query = {}

    def begin_forward(self, input_ids, visual_position_mask, is_prefill: bool):
        """Called by LLaVA immediately before the decoder layers."""
        if self.sample_id is None:
            return
        if input_ids is not None and input_ids.shape[0] != 1:
            raise ValueError("RetrievalShiftTracer currently requires batch_size=1")
        if is_prefill:
            if visual_position_mask is None:
                raise RuntimeError("Missing expanded visual-position mask on multimodal prefill")
            if visual_position_mask.shape[0] != 1:
                raise ValueError("RetrievalShiftTracer currently requires batch_size=1")
            self.prefix_visual_mask = visual_position_mask[0].detach().bool()
            self.prefix_length = int(self.prefix_visual_mask.numel())
            self.pending = _ForwardContext(True, -1, "<prefill_last_prompt>", 0)
            # The prefill's final prompt query predicts the first generated
            # token.  It is the unique valid decoding measurement with G empty.
            self.token_rows.append((-1, "<prefill_last_prompt>"))
            return
        if self.prefix_length is None:
            return
        token_id = int(input_ids[0, -1].item()) if input_ids is not None else -1
        token_text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False) if token_id >= 0 else ""
        self.pending = _ForwardContext(False, token_id, token_text, len(self.token_rows))
        self.token_rows.append((token_id, token_text))

    def _attention_pre_hook(self, layer_idx):
        def hook(attn, args, kwargs):
            context = self.pending
            if context is None or self.sample_id is None:
                return
            if getattr(attn.config, "pretraining_tp", 1) != 1:
                raise NotImplementedError(
                    "RetrievalShiftTracer requires pretraining_tp=1 so its "
                    "instrumented Q/K/V projections exactly match attention"
                )
            hidden = args[0]
            if hidden.shape[0] != 1:
                raise ValueError("RetrievalShiftTracer currently requires batch_size=1")
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                raise RuntimeError("Llama attention must receive explicit position_ids")
            q_pre = attn.q_proj(hidden)
            k_pre = attn.k_proj(hidden)
            v_pre = attn.v_proj(hidden)
            bsz, q_len, _ = hidden.shape
            q_pre = q_pre.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k_pre = k_pre.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            v_pre = v_pre.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            cache = kwargs.get("past_key_value")
            past_length = 0
            if cache is not None:
                if hasattr(cache, "get_usable_length"):
                    past_length = int(cache.get_usable_length(q_len, attn.layer_idx))
                elif isinstance(cache, (tuple, list)) and cache:
                    past_length = int(cache[0][0].shape[-2])
            cos, sin = attn.rotary_emb(v_pre, seq_len=past_length + q_len)
            q_post, k_post = apply_rotary_pos_emb(q_pre, k_pre, cos, sin, position_ids)
            self.state[layer_idx] = {
                "q_pre": q_pre.detach(), "k_pre": k_pre.detach(), "v_pre": v_pre.detach(),
                "q_post": q_post.detach(), "k_post": k_post.detach(), "past_length": torch.tensor(past_length),
            }
        return hook

    def _output_pre_hook(self, layer_idx):
        def hook(_, args):
            state = self.state.get(layer_idx)
            if state is not None:
                state["attention_output"] = args[0].detach()
        return hook

    @staticmethod
    def _cache_kv(cache, layer_idx):
        if cache is None:
            raise RuntimeError("Expected a KV cache while tracing generation")
        if hasattr(cache, "key_cache"):
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
        if isinstance(cache, (tuple, list)):
            return cache[layer_idx][0], cache[layer_idx][1]
        raise TypeError(f"Unsupported cache type: {type(cache)!r}")

    def _attention_post_hook(self, layer_idx):
        def hook(attn, args, kwargs, output):
            context = self.pending
            state = self.state.pop(layer_idx, None)
            if context is None or state is None or self.prefix_length is None:
                return
            cache = output[2]
            keys, values = self._cache_kv(cache, layer_idx)
            keys = repeat_kv(keys, attn.num_key_value_groups)[0].detach()
            values = repeat_kv(values, attn.num_key_value_groups)[0].detach()
            total = keys.shape[-2]
            if total != self.prefix_length + context.timestep:
                raise RuntimeError(
                    f"KV/group alignment failure: cache={total}, prefix={self.prefix_length}, timestep={context.timestep}"
                )
            q = state["q_post"][0, :, -1, :]
            key = keys
            value = values
            visual = torch.zeros(total, dtype=torch.bool, device=key.device)
            visual[:self.prefix_length] = self.prefix_visual_mask.to(key.device)
            prompt = torch.zeros(total, dtype=torch.bool, device=key.device)
            prompt[:self.prefix_length] = ~self.prefix_visual_mask.to(key.device)
            generated = torch.zeros(total, dtype=torch.bool, device=key.device)
            generated[self.prefix_length:] = True
            # Prefix padding, if any, is excluded through the prefill mask's length.
            original = state.get("attention_output")
            if original is None:
                raise RuntimeError("Missing o_proj pre-hook capture")
            original = original[0, -1].view(attn.num_heads, attn.head_dim)
            metrics = retrieval_group_metrics(q, key, value, visual, prompt, generated, original)
            prior_generated = generated.clone()
            if context.timestep > 0:
                prior_generated[-1] = False
            has_prior_generated = bool(prior_generated.any().item())
            if context.timestep > 0:
                old = retrieval_group_metrics(q, key[:, :-1], value[:, :-1], visual[:-1], prompt[:-1], prior_generated[:-1])
            else:
                old = None
            valid_effect = has_prior_generated and layer_idx in self.prev_query
            if valid_effect:
                prev = retrieval_group_metrics(self.prev_query[layer_idx], key[:, :-1], value[:, :-1], visual[:-1], prompt[:-1], prior_generated[:-1])
                query_effect = old["compat_GV"] - prev["compat_GV"]
                new_k_effect = metrics["compat_GV"] - old["compat_GV"]
            else:
                query_effect = torch.zeros_like(metrics["compat_GV"])
                new_k_effect = torch.zeros_like(metrics["compat_GV"])
            self.prev_query[layer_idx] = q.detach()
            row = {name: metrics[name].float().cpu().numpy() for name in METRIC_NAMES if name in metrics}
            row["query_effect"] = query_effect.float().cpu().numpy()
            row["new_K_effect"] = new_k_effect.float().cpu().numpy()
            row.update({name: metrics[name].bool().cpu().numpy() for name in ("has_V", "has_P", "has_G")})
            row["query_effect_valid"] = np.full(attn.num_heads, valid_effect, dtype=np.bool_)
            row["new_K_effect_valid"] = np.full(attn.num_heads, valid_effect, dtype=np.bool_)
            row.update({"timestep": context.timestep, "layer": layer_idx, "n_V": int(visual.sum()), "n_P": int(prompt.sum()), "n_G": int(generated.sum())})
            self.rows.append(row)
            self._maybe_dump_debug(layer_idx, context, state, q, key, value, visual, prompt, generated)
        return hook

    def _maybe_dump_debug(self, layer_idx, context, state, q, key, value, visual, prompt, generated):
        if self.debug is None or self.debug[0] != self.sample_id or self.debug[1] != layer_idx or self.debug[3] != context.timestep:
            return
        head = self.debug[2]
        if not 0 <= head < q.shape[0]:
            raise ValueError(f"Debug head {head} is outside [0, {q.shape[0]})")
        out = self.output_dir / "debug"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / f"sample{self.sample_id}_layer{layer_idx}_head{head}_t{context.timestep}.npz",
            q_pre=state["q_pre"][0, head, -1].float().cpu().numpy(),
            q_post=q[head].float().cpu().numpy(),
            k_pre=state["k_pre"][0, min(head, state["k_pre"].shape[1] - 1), -1].float().cpu().numpy(),
            k_post=state["k_post"][0, min(head, state["k_post"].shape[1] - 1), -1].float().cpu().numpy(),
            cached_k=key[head].float().cpu().numpy(), cached_v=value[head].float().cpu().numpy(),
            visual_mask=visual.cpu().numpy(), prompt_mask=prompt.cpu().numpy(), generated_mask=generated.cpu().numpy(),
        )

    def finish_sample(self):
        if self.sample_id is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"sample_{self.sample_id}.npz"
        if self.rows:
            timesteps = max(row["timestep"] for row in self.rows) + 1
            layers = len(self.model.model.layers)
            heads = self.model.config.num_attention_heads
            arrays = {name: np.zeros((timesteps, layers, heads), dtype=np.float32) for name in METRIC_NAMES}
            validity = {name: np.zeros((timesteps, layers, heads), dtype=np.bool_) for name in VALID_NAMES}
            counts = {name: np.zeros((timesteps, layers), dtype=np.int32) for name in ("n_V", "n_P", "n_G")}
            for row in self.rows:
                t, layer = row["timestep"], row["layer"]
                for name in METRIC_NAMES:
                    arrays[name][t, layer] = row[name]
                for name in VALID_NAMES:
                    validity[name][t, layer] = row[name]
                for name in counts:
                    counts[name][t, layer] = row[name]
            token_ids = np.asarray([x[0] for x in self.token_rows], dtype=np.int64)
            token_text = np.asarray([x[1] for x in self.token_rows], dtype="U")
            np.savez_compressed(output, token_ids=token_ids, token_text=token_text, **arrays, **validity, **counts)
            sidecar = output.with_suffix(".json")
            sidecar.write_text(json.dumps({"sample_id": self.sample_id, "prefix_length": self.prefix_length, "format": "[timestep, layer, head]", "metric_names": METRIC_NAMES, "validity_names": VALID_NAMES}, indent=2))
        self.sample_id = None
        self.pending = None
        return output
