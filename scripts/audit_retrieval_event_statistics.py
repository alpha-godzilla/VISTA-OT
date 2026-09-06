#!/usr/bin/env python3
"""Strict statistical re-audit of existing retrieval event windows.

This script uses no model and never treats layers/heads as independent samples.
Matched differences are clustered by hallucinated image; same-image pairs are
reported separately.  It intentionally leaves spatial/within-V questions for
the teacher-forced replay gate.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "size_effect", "query_content_effect", "query_position_effect", "new_K_effect",
    "compat_GV", "m_V", "m_G", "mass_ratio_log",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cluster_bootstrap(values, clusters, rng, repetitions):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    if not len(values):
        return np.nan, np.nan, np.nan
    unique = np.unique(clusters)
    members = [np.flatnonzero(clusters == item) for item in unique]
    samples = np.empty(repetitions)
    for index in range(repetitions):
        selected = np.concatenate([members[item] for item in rng.integers(len(members), size=len(members))])
        samples[index] = values[selected].mean()
    return values.mean(), *np.quantile(samples, [.025, .975])


def auc(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    if not positives or not negatives: return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        indices = np.flatnonzero(scores == value); ranks[indices] = ranks[indices].mean()
    return (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def ap(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    positives = labels.sum()
    if not positives: return np.nan
    ranked = labels[np.argsort(-scores, kind="mergesort")]
    return float((ranked * (np.cumsum(ranked) / np.arange(1, len(ranked) + 1))).sum() / positives)


def balanced_accuracy(labels, scores, threshold):
    labels, predicted = np.asarray(labels, int), np.asarray(scores) >= threshold
    positive, negative = labels == 1, labels == 0
    if not positive.any() or not negative.any(): return np.nan
    return .5 * (predicted[positive].mean() + (~predicted[negative]).mean())


def fit_balanced_logistic(train_x, train_y, test_x, steps=900, learning_rate=.12, l2=1e-3):
    mean, std = train_x.mean(0), train_x.std(0)
    std[std < 1e-8] = 1.
    x = np.c_[np.ones(len(train_x)), (train_x - mean) / std]
    test = np.c_[np.ones(len(test_x)), (test_x - mean) / std]
    count0, count1 = max(1, (train_y == 0).sum()), max(1, (train_y == 1).sum())
    weight = np.where(train_y == 1, len(train_y) / (2 * count1), len(train_y) / (2 * count0))
    parameter = np.zeros(x.shape[1]); regularizer = np.r_[0., np.full(x.shape[1] - 1, l2)]
    for _ in range(steps):
        probability = 1 / (1 + np.exp(-np.clip(x @ parameter, -30, 30)))
        gradient = x.T @ ((probability - train_y) * weight) / weight.sum() + regularizer * parameter
        parameter -= learning_rate * gradient
    train_score = 1 / (1 + np.exp(-np.clip(x @ parameter, -30, 30)))
    thresholds = np.unique(train_score)
    threshold = max(thresholds, key=lambda value: balanced_accuracy(train_y, train_score, value))
    return 1 / (1 + np.exp(-np.clip(test @ parameter, -30, 30))), threshold


def grouped_predictions(features, labels, images, folds=5):
    groups = sorted(np.unique(images)); folds = min(folds, len(groups))
    prediction, threshold = np.full(len(labels), np.nan), np.full(len(labels), np.nan)
    group_fold = {group: index % folds for index, group in enumerate(groups)}
    for fold in range(folds):
        test = np.array([group_fold[item] == fold for item in images]); train = ~test
        if len(np.unique(labels[train])) < 2: continue
        prediction[test], threshold[test] = fit_balanced_logistic(features[train], labels[train], features[test])
    return prediction, threshold


def image_bootstrap_predictions(labels, images, prediction_map, threshold_map, rng, repetitions):
    groups = np.unique(images); members = [np.flatnonzero(images == item) for item in groups]
    output = defaultdict(list)
    for _ in range(repetitions):
        selected = np.concatenate([members[item] for item in rng.integers(len(members), size=len(members))])
        scores = {name: values[selected] for name, values in prediction_map.items()}
        output["position_AUROC"].append(auc(labels[selected], scores["position_only"]))
        output["retrieval_AUROC"].append(auc(labels[selected], scores["retrieval_only"]))
        output["combined_AUROC"].append(auc(labels[selected], scores["position_plus_retrieval"]))
        output["delta_AUROC"].append(output["combined_AUROC"][-1] - output["position_AUROC"][-1])
        output["position_AUPRC"].append(ap(labels[selected], scores["position_only"]))
        output["retrieval_AUPRC"].append(ap(labels[selected], scores["retrieval_only"]))
        output["combined_AUPRC"].append(ap(labels[selected], scores["position_plus_retrieval"]))
        output["delta_AUPRC"].append(output["combined_AUPRC"][-1] - output["position_AUPRC"][-1])
        for name, scores in scores.items():
            output[f"{name}_balanced_accuracy"].append(
                balanced_accuracy(labels[selected], scores, threshold_map[name][selected])
            )
    return {key: (np.mean(value), *np.quantile(value, [.025, .975])) for key, value in output.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args(); rng = np.random.default_rng(args.seed)
    event_by_id = {int(row["event_id"]): row for row in read_csv(args.events)}
    pairs = read_csv(args.pairs)
    with np.load(args.windows, allow_pickle=False) as raw:
        ids, relative, valid = raw["event_ids"].astype(int), raw["relative_t"].astype(int), raw["valid"].astype(bool)
        event_index = {event_id: index for index, event_id in enumerate(ids)}
        metric = {name: raw[name].astype(np.float32).mean(axis=(2, 3)) for name in METRICS}
    requested = [-2, -1, 0]
    if not set(requested).issubset(relative): raise ValueError("Expected relative timesteps -2,-1,0")
    rel_index = {item: int(np.flatnonzero(relative == item)[0]) for item in requested}
    pair_rows, output = [], []
    for pair in pairs:
        hall_id, ground_id = int(pair["hall_event_id"]), int(pair["ground_event_id"])
        hall, ground = event_by_id[hall_id], event_by_id[ground_id]
        hidx, gidx = event_index[hall_id], event_index[ground_id]
        for rel in requested:
            index = rel_index[rel]
            if not (valid[hidx, index] and valid[gidx, index]): continue
            for name in METRICS:
                hval, gval = float(metric[name][hidx, index]), float(metric[name][gidx, index])
                pair_rows.append({"matched_pair_id": pair["matched_pair_id"], "matching_type": pair["matching_type"], "hall_image_id": hall["sample_id"], "ground_image_id": ground["sample_id"], "relative_t": rel, "metric": name, "hall_value": hval, "ground_value": gval, "difference": hval - gval})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "strict_pair_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)
    # Count controls at the matching-pair level, not once per metric/timestep.
    ground_reuse = Counter(
        event_by_id[int(pair["ground_event_id"])]["sample_id"]
        for pair in pairs if pair["matching_type"] == "cross_image"
    )
    with (args.output_dir / "cross_control_image_reuse.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ground_image_id", "pair_count")); writer.writeheader(); writer.writerows({"ground_image_id": item, "pair_count": count} for item, count in ground_reuse.items())
    analyses = {
        "same_image": lambda row: row["matching_type"] == "same_image",
        "cross_image_hall_image_cluster": lambda row: row["matching_type"] == "cross_image",
        "cross_unique_control_image": lambda row: row["matching_type"] == "cross_image" and ground_reuse[row["ground_image_id"]] == 1,
        "combined_hall_image_cluster": lambda row: True,
    }
    for analysis, keep in analyses.items():
        for rel in requested:
            for name in METRICS:
                subset = [row for row in pair_rows if keep(row) and row["relative_t"] == rel and row["metric"] == name]
                values = [float(row["difference"]) for row in subset]
                clusters = [row["hall_image_id"] for row in subset]
                mean, low, high = cluster_bootstrap(values, clusters, rng, args.bootstrap)
                output.append({"analysis": analysis, "relative_t": rel, "metric": name, "pair_count": len(values), "hall_image_clusters": len(set(clusters)), "hall_mean": np.mean([float(row["hall_value"]) for row in subset]) if subset else np.nan, "ground_mean": np.mean([float(row["ground_value"]) for row in subset]) if subset else np.nan, "difference_mean": mean, "difference_median": np.median(values) if values else np.nan, "ci_low": low, "ci_high": high})
    fields = ("analysis", "relative_t", "metric", "pair_count", "hall_image_clusters", "hall_mean", "ground_mean", "difference_mean", "difference_median", "ci_low", "ci_high")
    with (args.output_dir / "strict_matched_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output)

    # Image-grouped, class-balanced OOF predictive audit.  Features use only
    # the existing event windows and no test-fold fitting.
    complete = np.all(valid[:, [np.where(relative == item)[0][0] for item in range(-5, 0)]], axis=1)
    labels = np.asarray([event_by_id[item]["event_type"] == "hallucinated" for item in ids], int)
    images = np.asarray([int(event_by_id[item]["sample_id"]) for item in ids])
    pre = np.flatnonzero((relative >= -5) & (relative <= -1))
    features = {
        "generation_timestep": np.asarray([int(event_by_id[item]["decision_query_timestep"]) for item in ids]),
        "n_G": np.asarray([int(event_by_id[item]["decision_query_timestep"]) for item in ids]),
        "A_size": metric["size_effect"][:, pre].sum(1),
        "A_q_content": metric["query_content_effect"][:, pre].sum(1),
        "A_q_position": metric["query_position_effect"][:, pre].sum(1),
        "A_newK": metric["new_K_effect"][:, pre].sum(1),
        "pre_compat_GV": metric["compat_GV"][:, pre].mean(1),
        "pre_m_V": metric["m_V"][:, pre].mean(1),
        "pre_m_G": metric["m_G"][:, pre].mean(1),
        "pre_log_mass_ratio": metric["mass_ratio_log"][:, pre].mean(1),
    }
    sets = {"position_only": ("generation_timestep", "n_G"), "retrieval_only": tuple(name for name in features if name not in {"generation_timestep", "n_G"}), "position_plus_retrieval": tuple(features)}
    keep = complete
    oof, thresholds = {}, {}
    for name, columns in sets.items():
        oof[name], thresholds[name] = grouped_predictions(np.column_stack([features[column][keep] for column in columns]), labels[keep], images[keep])
    oof_path = args.output_dir / "strict_predictive_oof.csv"
    with oof_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("event_id", "image_id", "label", *sets, *(f"{name}_threshold" for name in sets))); writer.writeheader()
        for local, global_index in enumerate(np.flatnonzero(keep)):
            writer.writerow({"event_id": int(ids[global_index]), "image_id": int(images[global_index]), "label": int(labels[global_index]), **{name: oof[name][local] for name in sets}, **{f"{name}_threshold": thresholds[name][local] for name in sets}})
    prediction_bootstrap = image_bootstrap_predictions(labels[keep], images[keep], oof, thresholds, rng, args.bootstrap)
    predictive = []
    for name in sets:
        for metric_name, function in (("AUROC", auc), ("AUPRC", ap), ("balanced_accuracy", None)):
            if function is None:
                value = balanced_accuracy(labels[keep], oof[name], thresholds[name])
                low, high = np.quantile(prediction_bootstrap[f"{name}_balanced_accuracy"], [.025, .975])
            else:
                key = f"{name.split('_')[0]}_{metric_name}" if name != "position_plus_retrieval" else f"combined_{metric_name}"
                value, low, high = prediction_bootstrap[key]
            predictive.append({"model": name, "metric": metric_name, "events": int(keep.sum()), "images": len(np.unique(images[keep])), "mean": value, "ci_low": low, "ci_high": high})
    for metric_name in ("AUROC", "AUPRC"):
        value, low, high = prediction_bootstrap[f"delta_{metric_name}"]
        predictive.append({"model": "position_plus_retrieval_minus_position", "metric": f"delta_{metric_name}", "events": int(keep.sum()), "images": len(np.unique(images[keep])), "mean": value, "ci_low": low, "ci_high": high})
    with (args.output_dir / "strict_predictive_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "metric", "events", "images", "mean", "ci_low", "ci_high")); writer.writeheader(); writer.writerows(predictive)
    report = {"same_pairs": sum(row["matching_type"] == "same_image" for row in pairs), "cross_pairs": sum(row["matching_type"] == "cross_image" for row in pairs), "cross_control_images": len(ground_reuse), "cross_control_images_reused": sum(count > 1 for count in ground_reuse.values()), "statistical_unit": "hallucinated image cluster (same-image pairs also preserve pair structure)"}
    (args.output_dir / "strict_stats_audit.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
