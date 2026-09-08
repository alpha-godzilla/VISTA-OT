"""Inference-only VSV dose laws and numerically stable geometry utilities.

This module contains no CHAIR/COCO access.  Features are computed from the
positive/negative prefill states (and, optionally, first-token logits); labels
are deliberately absent from the controller interface.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import math
import torch
import torch.nn.functional as F


def safe_normalize(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def legacy_lambda_sim(c):
    return 1.0 + torch.clamp(-c, min=0.0)


def steering_angle(c, lam, sim_gate="legacy"):
    """Angle of normalize(u + lambda*g*d) relative to u."""
    c = torch.as_tensor(c, dtype=torch.float64).clamp(-1.0, 1.0)
    lam = torch.as_tensor(lam, dtype=torch.float64).clamp_min(0.0)
    gate = legacy_lambda_sim(c) if sim_gate == "legacy" else torch.ones_like(c)
    a = lam * gate
    perpendicular = torch.sqrt(torch.clamp(1.0 - c * c, min=0.0))
    return torch.atan2(a * perpendicular, 1.0 + a * c)


def direct_steering_angle(u, d, lam, sim_gate="legacy"):
    u, d = safe_normalize(u), safe_normalize(d)
    if sim_gate == "legacy":
        gate = legacy_lambda_sim(F.cosine_similarity(u, d, dim=-1))
    else:
        gate = torch.ones_like(lam if torch.is_tensor(lam) else torch.as_tensor(lam))
    y = safe_normalize(u + torch.as_tensor(lam, device=u.device)[..., None] * gate[..., None] * d)
    return torch.acos(torch.sum(u * y, dim=-1).clamp(-1.0, 1.0))


def solve_lambda_for_angle(c, target, sim_gate="legacy", lo=0.0, hi=0.30, steps=64):
    """Monotone bisection with explicit bracket/clamp status."""
    target = float(target)
    high_angle = float(steering_angle(c, hi, sim_gate))
    if target <= 0: return lo, "zero_target"
    if target >= high_angle: return hi, "upper_clamped"
    left, right = float(lo), float(hi)
    for _ in range(steps):
        mid = (left + right) / 2.0
        if float(steering_angle(c, mid, sim_gate)) < target: left = mid
        else: right = mid
    return (left + right) / 2.0, "solved"


def summarize_angles(cosines, lam, sim_gate="legacy", late_start=0.75):
    c = torch.as_tensor(cosines, dtype=torch.float64)
    theta = steering_angle(c, lam, sim_gate)
    late = theta[int(len(theta) * late_start):] if len(theta) else theta
    def stat(x, q=None):
        if not len(x): return float("nan")
        return float(torch.quantile(x, q) if q is not None else x.mean())
    return {"theta_mean": stat(theta), "theta_median": stat(theta, .5),
            "theta_std": float(theta.std(unbiased=False)) if len(theta) else float("nan"),
            "theta_max": float(theta.max()) if len(theta) else float("nan"),
            "theta_late_mean": stat(late), "theta_late_median": stat(late, .5)}


def layer_consistency(vectors, late_start=0.75):
    v = safe_normalize(torch.as_tensor(vectors, dtype=torch.float32))
    if v.shape[0] < 2:
        return {"adjacent_cos_mean": 1.0, "adjacent_cos_std": 0.0,
                "adjacent_cos_min": 1.0, "negative_adjacent_fraction": 0.0,
                "late_adjacent_cos_mean": 1.0, "late_adjacent_cos_std": 0.0,
                "normalized_direction_smoothness": 0.0}
    cos = (v[:-1] * v[1:]).sum(-1)
    late = cos[int(len(cos) * late_start):]
    return {"adjacent_cos_mean": float(cos.mean()), "adjacent_cos_std": float(cos.std(unbiased=False)),
            "adjacent_cos_min": float(cos.min()), "negative_adjacent_fraction": float((cos < 0).float().mean()),
            "late_adjacent_cos_mean": float(late.mean()), "late_adjacent_cos_std": float(late.std(unbiased=False)),
            "normalized_direction_smoothness": float((v[1:] - v[:-1]).norm(dim=-1).mean())}


@dataclass
class VSVFeatureRecord:
    raw_diff_norm_mean: float
    raw_diff_norm_cv: float
    raw_diff_norm_slope: float
    relative_raw_diff_norm_mean: float
    cos_pos_neg_mean: float
    angular_pos_neg_mean: float
    official_raw_cos_mean: float
    layer_consistency: Dict[str, float]
    geometry_cos_mean: float
    geometry_cos_median: float

    def to_dict(self):
        output = asdict(self); consistency = output.pop("layer_consistency")
        output.update({f"layer_{key}": value for key, value in consistency.items()})
        return output


def build_vsv_features(hidden_states, official_vsv):
    """Build diagnostics from [style(neg,pos), layer, hidden] prefill states."""
    neg, pos = hidden_states[0], hidden_states[1]
    raw = pos - neg
    raw_norm = raw.norm(dim=-1)
    rel = raw_norm / neg.norm(dim=-1).clamp_min(1e-8)
    cos_np = F.cosine_similarity(pos, neg, dim=-1).clamp(-1, 1)
    angular = torch.acos(cos_np)
    official_raw = F.cosine_similarity(raw, official_vsv, dim=-1)
    positions = torch.arange(len(raw), dtype=torch.float32)
    centered = positions - positions.mean()
    slope = float((centered * (raw_norm.float() - raw_norm.float().mean())).sum() /
                  centered.square().sum().clamp_min(1e-8))
    return VSVFeatureRecord(
        raw_diff_norm_mean=float(raw_norm.mean()), raw_diff_norm_cv=float(raw_norm.std(unbiased=False) / raw_norm.mean().clamp_min(1e-8)),
        raw_diff_norm_slope=slope, relative_raw_diff_norm_mean=float(rel.mean()),
        cos_pos_neg_mean=float(cos_np.mean()), angular_pos_neg_mean=float(angular.mean()),
        official_raw_cos_mean=float(official_raw.mean()), layer_consistency=layer_consistency(official_vsv),
        geometry_cos_mean=float(F.cosine_similarity(safe_normalize(pos), safe_normalize(official_vsv), dim=-1).mean()),
        geometry_cos_median=float(F.cosine_similarity(safe_normalize(pos), safe_normalize(official_vsv), dim=-1).median()),
    )


def law_dose(feature, law, scale=0.17, eps=1e-6, min_lambda=0.0, max_lambda=0.30):
    """Simple interpretable laws; scale is fitted on calibration only."""
    x = float(feature)
    if law in ("visual_deficit_inverse", "sensitivity_inverse"):
        value = scale / max(abs(x), eps)
    elif law in ("visual_deficit_direct", "sensitivity_direct", "consistency"):
        value = scale * x
    else:
        raise ValueError(f"unknown law: {law}")
    return max(min_lambda, min(max_lambda, value))
