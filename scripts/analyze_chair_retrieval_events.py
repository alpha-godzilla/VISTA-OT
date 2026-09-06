#!/usr/bin/env python3
"""Event-level, position-controlled CHAIR retrieval-shift analysis.

This is descriptive measurement only.  It reports associations with CHAIR
object labels; no result from this script is a causal hallucination claim.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_METRICS = (
    "size_effect", "query_effect", "query_content_effect", "query_position_effect",
    "new_K_effect", "compat_GV", "compat_PV", "m_V", "m_P", "m_G", "mass_ratio_log",
)
DISPLAY_METRICS = (
    "size_effect", "query_content_effect", "query_position_effect", "new_K_effect",
    "net_compat_effect", "total_mass_effect", "compat_GV", "compat_PV",
    "m_V", "m_G", "log_mass_ratio_GV", "semantic_compensation_margin",
)


def read_events(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bootstrap_cluster(values, clusters, rng, iterations):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    if not len(values):
        return (np.nan, np.nan, np.nan)
    unique = np.unique(clusters)
    groups = [np.flatnonzero(clusters == group) for group in unique]
    means = np.empty(iterations, dtype=float)
    for index in range(iterations):
        picked = rng.integers(len(groups), size=len(groups))
        selection = np.concatenate([groups[item] for item in picked])
        means[index] = values[selection].mean()
    return values.mean(), *np.quantile(means, [0.025, 0.975])


def effect_size(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def match_events(events, same_window, cross_window, order_window, relative_position_window):
    """Greedily construct non-reused grounded controls, preferring same image."""
    hall = [index for index, row in enumerate(events) if row["event_type"] == "hallucinated"]
    ground = [index for index, row in enumerate(events) if row["event_type"] == "grounded"]
    used, pairs = set(), []
    def distance(left, right):
        timestep = abs(int(events[left]["decision_query_timestep"]) - int(events[right]["decision_query_timestep"]))
        order = abs(int(events[left]["event_order"]) - int(events[right]["event_order"]))
        left_rel = int(events[left]["decision_query_timestep"]) / max(1, int(events[left]["caption_token_count"]))
        right_rel = int(events[right]["decision_query_timestep"]) / max(1, int(events[right]["caption_token_count"]))
        return timestep, order, abs(left_rel - right_rel)
    for hall_index in sorted(hall, key=lambda x: (int(events[x]["sample_id"]), int(events[x]["decision_query_timestep"]))):
        candidates = [
            ground_index for ground_index in ground if ground_index not in used
            and events[ground_index]["sample_id"] == events[hall_index]["sample_id"]
            and distance(hall_index, ground_index)[0] <= same_window
        ]
        matching_type = "same_image"
        if not candidates:
            candidates = [
                ground_index for ground_index in ground if ground_index not in used
                and distance(hall_index, ground_index)[0] <= cross_window
                and distance(hall_index, ground_index)[1] <= order_window
                and distance(hall_index, ground_index)[2] <= relative_position_window
            ]
            matching_type = "cross_image"
        if not candidates:
            continue
        ground_index = min(candidates, key=lambda item: distance(hall_index, item))
        used.add(ground_index)
        timestep_distance, _, _ = distance(hall_index, ground_index)
        pairs.append({
            "matched_pair_id": len(pairs), "hall_event_id": events[hall_index]["event_id"],
            "ground_event_id": events[ground_index]["event_id"], "matching_type": matching_type,
            "position_distance": timestep_distance,
        })
    return pairs


def auc_roc(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    if positives == 0 or negatives == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float); ranks[order] = np.arange(1, len(scores) + 1)
    # Average ties exactly.
    for value in np.unique(scores):
        tied = np.flatnonzero(scores == value)
        ranks[tied] = ranks[tied].mean()
    return (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    positives = labels.sum()
    if positives == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def balanced_accuracy(labels, scores):
    labels, predicted = np.asarray(labels, int), np.asarray(scores) >= .5
    positives, negatives = labels == 1, labels == 0
    if not positives.any() or not negatives.any():
        return np.nan
    return .5 * (predicted[positives].mean() + (~predicted[negatives]).mean())


def fit_logistic(train_x, train_y, test_x, l2=1e-3, steps=600, learning_rate=.1):
    mean, std = train_x.mean(0), train_x.std(0)
    std[std < 1e-8] = 1.
    x, test = (train_x - mean) / std, (test_x - mean) / std
    x = np.c_[np.ones(len(x)), x]; test = np.c_[np.ones(len(test)), test]
    weights = np.zeros(x.shape[1])
    regularizer = np.r_[0., np.full(x.shape[1] - 1, l2)]
    for _ in range(steps):
        logits = np.clip(x @ weights, -30, 30)
        probability = 1 / (1 + np.exp(-logits))
        weights -= learning_rate * ((x.T @ (probability - train_y) / len(x)) + regularizer * weights)
    return 1 / (1 + np.exp(-np.clip(test @ weights, -30, 30)))


def grouped_cv_predictions(features, labels, image_ids, folds=5):
    groups = np.unique(image_ids)
    if len(groups) < 4:
        return np.full(len(labels), np.nan)
    fold_count = min(folds, len(groups))
    # Deterministic image-group split without requiring sklearn.
    group_fold = {group: index % fold_count for index, group in enumerate(sorted(groups))}
    prediction = np.full(len(labels), np.nan)
    for fold in range(fold_count):
        test = np.array([group_fold[group] == fold for group in image_ids])
        train = ~test
        if len(np.unique(labels[train])) < 2:
            continue
        prediction[test] = fit_logistic(features[train], labels[train], features[test])
    return prediction


def image_bootstrap_metric(labels, scores, images, metric, rng, iterations):
    unique = np.unique(images)
    groups = [np.flatnonzero(images == image) for image in unique]
    values = []
    for _ in range(iterations):
        chosen = np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
        score = metric(labels[chosen], scores[chosen])
        if np.isfinite(score): values.append(score)
    return np.quantile(values, [.025, .975]).tolist() if values else [np.nan, np.nan]


def plot_trajectory(summary, output, metrics):
    figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.4), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, metric in zip(axes, metrics):
        for kind, color in (("grounded", "#1f77b4"), ("hallucinated", "#d62728")):
            rows = [row for row in summary if row["analysis"] == "unmatched" and row["metric"] == metric and row["event_type"] == kind]
            if not rows: continue
            x = np.array([int(row["relative_t"]) for row in rows]); y = np.array([float(row["mean"]) for row in rows])
            lo = np.array([float(row["ci_low"]) for row in rows]); hi = np.array([float(row["ci_high"]) for row in rows])
            axis.plot(x, y, label=kind, color=color); axis.fill_between(x, lo, hi, color=color, alpha=.18)
        axis.axvline(0, color="black", linewidth=.7); axis.set_title(metric); axis.set_xlabel("relative timestep")
    axes[0].set_ylabel("event-level layer/head mean"); axes[-1].legend(); figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--same-image-window", type=int, default=5)
    parser.add_argument("--cross-image-window", type=int, default=2)
    parser.add_argument("--order-window", type=int, default=2)
    parser.add_argument("--relative-position-window", type=float, default=.15)
    args = parser.parse_args()
    events = read_events(args.events)
    by_id = {int(row["event_id"]): row for row in events}
    with np.load(args.windows, allow_pickle=False) as windows:
        window_ids = windows["event_ids"].astype(int)
        missing = set(window_ids) - set(by_id)
        if missing: raise RuntimeError(f"Windows reference unknown events: {sorted(missing)[:5]}")
        events = [by_id[event_id] for event_id in window_ids]
        relative = windows["relative_t"].astype(int)
        valid = windows["valid"].astype(bool)
        source = {metric: windows[metric].astype(np.float32) for metric in BASE_METRICS}
        counts = {metric: windows[metric].astype(np.float32) for metric in ("n_V", "n_P", "n_G")}
    # Preserve full layer/head arrays in the input NPZ; only this primary
    # statistical view averages them within an event.
    values = {metric: source[metric].mean(axis=(2, 3)) for metric in BASE_METRICS}
    values["net_compat_effect"] = values["query_effect"] + values["new_K_effect"]
    values["total_mass_effect"] = (values["size_effect"] + values["query_content_effect"] + values["query_position_effect"] + values["new_K_effect"])
    values["log_mass_ratio_GV"] = values["mass_ratio_log"]
    values["semantic_compensation_margin"] = -values["query_content_effect"] - values["new_K_effect"]
    count_values = {metric: counts[metric].mean(axis=(2, 3)) for metric in counts}
    labels = np.asarray([row["event_type"] == "hallucinated" for row in events], int)
    images = np.asarray([int(row["sample_id"]) for row in events])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = match_events(events, args.same_image_window, args.cross_image_window, args.order_window, args.relative_position_window)
    with (args.output_dir / "matched_event_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("matched_pair_id", "hall_event_id", "ground_event_id", "matching_type", "position_distance")); writer.writeheader(); writer.writerows(pairs)
    event_index = {int(row["event_id"]): index for index, row in enumerate(events)}
    summary, rng = [], np.random.default_rng(args.seed)
    for analysis in ("unmatched", "matched"):
        for rel_index, rel in enumerate(relative):
            for metric in DISPLAY_METRICS:
                if analysis == "unmatched":
                    for kind, label in (("grounded", 0), ("hallucinated", 1)):
                        mask = (labels == label) & valid[:, rel_index]
                        item, cluster = values[metric][mask, rel_index], images[mask]
                        mean, lo, hi = bootstrap_cluster(item, cluster, rng, args.bootstrap)
                        summary.append({"analysis": analysis, "event_type": kind, "relative_t": rel, "metric": metric, "event_count": len(item), "unique_images": len(np.unique(cluster)), "mean": mean, "median": np.median(item) if len(item) else np.nan, "std": np.std(item, ddof=1) if len(item) > 1 else np.nan, "ci_low": lo, "ci_high": hi, "effect_size_hall_minus_ground": np.nan})
                else:
                    hall, ground, pair_cluster = [], [], []
                    for pair in pairs:
                        h, g = event_index[int(pair["hall_event_id"])], event_index[int(pair["ground_event_id"])]
                        if valid[h, rel_index] and valid[g, rel_index]:
                            hall.append(values[metric][h, rel_index]); ground.append(values[metric][g, rel_index]); pair_cluster.append(f"{images[h]}|{images[g]}")
                    difference = np.asarray(hall) - np.asarray(ground)
                    mean, lo, hi = bootstrap_cluster(difference, pair_cluster, rng, args.bootstrap)
                    summary.append({"analysis": analysis, "event_type": "hall_minus_ground", "relative_t": rel, "metric": metric, "event_count": len(difference), "unique_images": len(set(pair_cluster)), "mean": mean, "median": np.median(difference) if len(difference) else np.nan, "std": np.std(difference, ddof=1) if len(difference) > 1 else np.nan, "ci_low": lo, "ci_high": hi, "effect_size_hall_minus_ground": effect_size(hall, ground)})
    fields = ("analysis", "event_type", "relative_t", "metric", "event_count", "unique_images", "mean", "median", "std", "ci_low", "ci_high", "effect_size_hall_minus_ground")
    with (args.output_dir / "event_level_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    plot_trajectory(summary, args.output_dir / "figure1_component_trajectories.png", DISPLAY_METRICS[:4])
    plot_trajectory(summary, args.output_dir / "figure2_log_mass_ratio.png", ("log_mass_ratio_GV",))
    plot_trajectory(summary, args.output_dir / "figure3_semantic_compensation.png", ("semantic_compensation_margin",))
    matched_components = ("size_effect", "query_content_effect", "query_position_effect", "new_K_effect")
    figure, axes = plt.subplots(1, len(matched_components), figsize=(4 * len(matched_components), 3.4), sharex=True)
    for axis, metric in zip(np.atleast_1d(axes), matched_components):
        rows = [row for row in summary if row["analysis"] == "matched" and row["metric"] == metric]
        if rows:
            x = np.array([int(row["relative_t"]) for row in rows]); y = np.array([float(row["mean"]) for row in rows])
            lo = np.array([float(row["ci_low"]) for row in rows]); hi = np.array([float(row["ci_high"]) for row in rows])
            axis.plot(x, y, color="#6a3d9a"); axis.fill_between(x, lo, hi, color="#6a3d9a", alpha=.18)
        axis.axhline(0, color="black", linewidth=.7); axis.axvline(0, color="black", linewidth=.7)
        axis.set_title(f"matched Δ {metric}"); axis.set_xlabel("relative timestep")
    axes[0].set_ylabel("hallucinated − grounded"); figure.tight_layout(); figure.savefig(args.output_dir / "figure6_position_matched.png", dpi=180); plt.close(figure)
    # Pre-event cumulative event features and matched bar plot.
    pre = np.flatnonzero((relative >= -5) & (relative <= -1))
    complete = valid[:, pre].all(axis=1)
    feature = {
        "A_size": values["size_effect"][:, pre].sum(1),
        "A_q_content": values["query_content_effect"][:, pre].sum(1),
        "A_q_position": values["query_position_effect"][:, pre].sum(1),
        "A_newK": values["new_K_effect"][:, pre].sum(1),
        "A_query": values["query_effect"][:, pre].sum(1),
        "A_compat": values["net_compat_effect"][:, pre].sum(1),
        "A_total": values["total_mass_effect"][:, pre].sum(1),
        "A_semantic_comp_margin": values["semantic_compensation_margin"][:, pre].sum(1),
        "pre_event_mean_compat_GV": values["compat_GV"][:, pre].mean(1),
        "pre_event_mean_log_mass_ratio_GV": values["log_mass_ratio_GV"][:, pre].mean(1),
        "pre_event_mean_m_V": values["m_V"][:, pre].mean(1),
        "pre_event_mean_m_G": values["m_G"][:, pre].mean(1),
        "generation_timestep": np.asarray([int(row["decision_query_timestep"]) for row in events]),
        "n_G": count_values["n_G"][:, -1],
    }
    cumulative_rows = []
    for name, feature_values in feature.items():
        for kind, label in (("grounded", 0), ("hallucinated", 1)):
            mask = (labels == label) & complete
            mean, lo, hi = bootstrap_cluster(feature_values[mask], images[mask], rng, args.bootstrap)
            cumulative_rows.append({"event_type": kind, "metric": name, "event_count": int(mask.sum()), "unique_images": len(np.unique(images[mask])), "mean": mean, "ci_low": lo, "ci_high": hi})
    with (args.output_dir / "cumulative_pre_event.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("event_type", "metric", "event_count", "unique_images", "mean", "ci_low", "ci_high")); writer.writeheader(); writer.writerows(cumulative_rows)
    component_names = ("A_size", "A_q_content", "A_q_position", "A_newK")
    figure, axis = plt.subplots(figsize=(7, 3.5)); x = np.arange(len(component_names)); width = .35
    for offset, (kind, label, color) in enumerate((("grounded", 0, "#1f77b4"), ("hallucinated", 1, "#d62728"))):
        means = [feature[name][(labels == label) & complete].mean() for name in component_names]
        axis.bar(x + (offset - .5) * width, means, width, label=kind, color=color)
    axis.set_xticks(x, component_names); axis.set_ylabel("pre-event cumulative mean"); axis.legend(); figure.tight_layout(); figure.savefig(args.output_dir / "figure4_cumulative_components.png", dpi=180); plt.close(figure)
    # Localization keeps layer/head dimensions rather than treating them as observations.
    localization_metrics = ("query_content_effect", "query_position_effect", "new_K_effect", "mass_ratio_log", "m_V", "m_G")
    head_effect_rows = []
    for metric in localization_metrics:
        hall = source[metric][labels == 1]; ground = source[metric][labels == 0]
        hall_valid, ground_valid = valid[labels == 1], valid[labels == 0]
        layer_time = np.zeros((len(relative), hall.shape[2]), dtype=float)
        for rel_index in range(len(relative)):
            hmask, gmask = hall_valid[:, rel_index], ground_valid[:, rel_index]
            if hmask.any() and gmask.any():
                layer_time[rel_index] = hall[hmask, rel_index].mean(axis=(0, 2)) - ground[gmask, rel_index].mean(axis=(0, 2))
        figure, axis = plt.subplots(figsize=(10, 3)); image = axis.imshow(layer_time, aspect="auto", cmap="coolwarm"); axis.set_yticks(range(len(relative)), relative); axis.set_xlabel("layer"); axis.set_ylabel("relative timestep"); axis.set_title(f"Hallucinated − grounded: {metric} (head mean)"); figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(args.output_dir / f"{metric}_relative_t_layer_heatmap.png", dpi=180); plt.close(figure)
        if pre.size:
            hmask, gmask = complete & (labels == 1), complete & (labels == 0)
            if hmask.any() and gmask.any():
                head_layer = source[metric][hmask][:, pre].mean(axis=(0, 1)) - source[metric][gmask][:, pre].mean(axis=(0, 1))
                figure, axis = plt.subplots(figsize=(10, 6)); image = axis.imshow(head_layer, aspect="auto", cmap="coolwarm"); axis.set_xlabel("head"); axis.set_ylabel("layer"); axis.set_title(f"Pre-event hallucinated − grounded: {metric}"); figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(args.output_dir / f"{metric}_layer_head_heatmap.png", dpi=180); plt.close(figure)
                for layer in range(head_layer.shape[0]):
                    for head in range(head_layer.shape[1]):
                        head_effect_rows.append({"metric": metric, "layer": layer, "head": head, "pre_event_mean_difference": float(head_layer[layer, head])})
    with (args.output_dir / "head_localization_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "layer", "head", "pre_event_mean_difference")); writer.writeheader(); writer.writerows(head_effect_rows)
    # Grouped-CV predictive sanity check.  Scores are out-of-fold, then CIs
    # bootstrap images with predictions fixed (uncertainty of held-out scores).
    usable = complete
    feature_sets = {
        "position_only": ("generation_timestep", "n_G"),
        "retrieval_only": ("A_size", "A_q_content", "A_q_position", "A_newK", "A_total", "pre_event_mean_compat_GV", "pre_event_mean_log_mass_ratio_GV", "pre_event_mean_m_V", "pre_event_mean_m_G"),
        "position_plus_retrieval": ("generation_timestep", "n_G", "A_size", "A_q_content", "A_q_position", "A_newK", "A_total", "pre_event_mean_compat_GV", "pre_event_mean_log_mass_ratio_GV", "pre_event_mean_m_V", "pre_event_mean_m_G"),
    }
    predictive_rows = []
    for name, names in feature_sets.items():
        x = np.column_stack([feature[item][usable] for item in names]); y, group = labels[usable], images[usable]
        prediction = grouped_cv_predictions(x, y, group)
        keep = np.isfinite(prediction)
        for metric_name, function in (("AUROC", auc_roc), ("AUPRC", average_precision), ("balanced_accuracy", balanced_accuracy)):
            if keep.any():
                point = function(y[keep], prediction[keep])
                lo, hi = image_bootstrap_metric(y[keep], prediction[keep], group[keep], function, rng, args.bootstrap)
            else:
                point, lo, hi = np.nan, np.nan, np.nan
            predictive_rows.append({"model": name, "metric": metric_name, "events": int(keep.sum()), "images": len(np.unique(group[keep])), "mean": point, "ci_low": lo, "ci_high": hi})
    with (args.output_dir / "predictive_sanity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "metric", "events", "images", "mean", "ci_low", "ci_high")); writer.writeheader(); writer.writerows(predictive_rows)
    figure, axis = plt.subplots(figsize=(6, 3.5)); models = list(feature_sets); x = np.arange(len(models));
    for offset, metric_name in enumerate(("AUROC", "AUPRC", "balanced_accuracy")):
        rows = [row for row in predictive_rows if row["metric"] == metric_name]
        axis.bar(x + (offset - 1) * .23, [row["mean"] for row in rows], .23, label=metric_name)
    axis.set_xticks(x, models, rotation=15); axis.set_ylim(0, 1); axis.legend(); figure.tight_layout(); figure.savefig(args.output_dir / "figure7_predictive_sanity.png", dpi=180); plt.close(figure)
    report = {
        "numerical_scope": "Descriptive association only; no intervention and no causal claim.",
        "events": len(events), "grounded_events": int((labels == 0).sum()), "hallucinated_events": int((labels == 1).sum()),
        "unique_images": len(np.unique(images)), "events_without_full_pre_window": int((~complete).sum()),
        "same_image_pairs": sum(pair["matching_type"] == "same_image" for pair in pairs),
        "cross_image_pairs": sum(pair["matching_type"] == "cross_image" for pair in pairs),
        "window_npz": str(args.windows),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
