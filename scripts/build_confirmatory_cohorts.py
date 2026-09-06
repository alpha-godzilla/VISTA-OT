#!/usr/bin/env python3
"""Pre-registered one-to-one cohorts for the retrieval confirmatory study."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import heapq


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assignment(halls, grounds, tolerance, forbid_same_image=False):
    """Maximum-cardinality then minimum-distance deterministic assignment."""
    if not halls or not grounds: return []
    # Min-cost maximum-flow avoids an undeclared SciPy dependency on remote
    # evaluation nodes.  We augment until no path remains, hence cardinality
    # is maximized first; each path cost is its position distance.
    n, m = len(halls), len(grounds)
    graph = [[] for _ in range(n + m + 2)]
    source, sink = n + m, n + m + 1
    def add(u, v, cap, cost):
        graph[u].append([v, len(graph[v]), cap, cost])
        graph[v].append([u, len(graph[u]) - 1, 0, -cost])
    for i in range(n): add(source, i, 1, 0)
    for j in range(m): add(n + j, sink, 1, 0)
    candidates = []
    for i, hall in enumerate(halls):
        for j, ground in enumerate(grounds):
            distance = abs(int(hall["decision_query_timestep"]) - int(ground["decision_query_timestep"]))
            if distance <= tolerance and (not forbid_same_image or hall["sample_id"] != ground["sample_id"]):
                # Integer tie-breaker is deterministic but never outweighs
                # a one-token distance difference.
                cost = distance * (n * m + 1) + i * max(1, m) + j
                add(i, n + j, 1, cost); candidates.append((i, j))
    potential = [0] * len(graph)
    while True:
        dist = [10**30] * len(graph); prev = [None] * len(graph); dist[source] = 0
        queue = [(0, source)]
        while queue:
            current, node = heapq.heappop(queue)
            if current != dist[node]: continue
            for edge_index, edge in enumerate(graph[node]):
                target, _, cap, cost = edge
                if not cap: continue
                candidate = current + cost + potential[node] - potential[target]
                if candidate < dist[target]:
                    dist[target] = candidate; prev[target] = (node, edge_index); heapq.heappush(queue, (candidate, target))
        if prev[sink] is None: break
        for node, value in enumerate(dist):
            if value < 10**30: potential[node] += value
        node = sink
        while node != source:
            parent, edge_index = prev[node]; edge = graph[parent][edge_index]
            edge[2] -= 1; graph[node][edge[1]][2] += 1; node = parent
    result = []
    for i, j in candidates:
        # Forward edge consumed means its flow is one.
        if not any(edge[0] == n + j and edge[2] == 0 for edge in graph[i]): continue
        result.append((halls[i], grounds[j]))
    return result


def make_same(events, tolerance):
    by_image = defaultdict(list)
    for row in events: by_image[row["sample_id"]].append(row)
    pairs = []
    for image, rows in sorted(by_image.items(), key=lambda item: int(item[0])):
        hall = [row for row in rows if row["event_type"] == "hallucinated"]
        ground = [row for row in rows if row["event_type"] == "grounded"]
        for h, g in assignment(hall, ground, tolerance):
            pairs.append({"image_id": image, "hall_event_id": h["event_id"], "ground_event_id": g["event_id"],
                          "position_distance": abs(int(h["decision_query_timestep"]) - int(g["decision_query_timestep"]))})
    return pairs


def make_category(events, tolerance):
    by_category = defaultdict(list)
    for row in events: by_category[row["object_category"]].append(row)
    pairs = []
    for category, rows in sorted(by_category.items()):
        hall = [row for row in rows if row["event_type"] == "hallucinated"]
        ground = [row for row in rows if row["event_type"] == "grounded"]
        for h, g in assignment(hall, ground, tolerance, forbid_same_image=True):
            pairs.append({"hall_image_id": h["sample_id"], "ground_image_id": g["sample_id"],
                          "hall_event_id": h["event_id"], "ground_event_id": g["event_id"], "category": category,
                          "surface_form_hall": h["object_text"], "surface_form_ground": g["object_text"],
                          "same_surface_form": h["object_text"].lower() == g["object_text"].lower(),
                          "position_distance": abs(int(h["decision_query_timestep"]) - int(g["decision_query_timestep"]))})
    return pairs


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ("hall_event_id", "ground_event_id", "position_distance")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--events", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); events = read_csv(args.events)
    aligned = [row for row in events if row["decision_query_timestep"] != ""]
    types_by_image = defaultdict(set)
    for row in aligned: types_by_image[row["sample_id"]].add(row["event_type"])
    mixed = sorted(int(image) for image, types in types_by_image.items() if {"grounded", "hallucinated"} <= types)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mixed_images.csv").write_text("image_id\n" + "".join(f"{item}\n" for item in mixed))
    summary = []
    for tolerance in (2, 5, 10):
        same, category = make_same(aligned, tolerance), make_category(aligned, tolerance)
        for index, row in enumerate(same): row["matched_pair_id"] = index
        for index, row in enumerate(category): row["matched_pair_id"] = index
        write(args.output_dir / f"same_image_pairs_tolerance{tolerance}.csv", same)
        write(args.output_dir / f"same_category_pairs_tolerance{tolerance}.csv", category)
        summary.append({"tolerance": tolerance, "same_image_pairs": len(same), "same_image_clusters": len({row["image_id"] for row in same}),
                        "same_category_pairs": len(category), "same_category_hall_clusters": len({row["hall_image_id"] for row in category}),
                        "same_category_controls": len({row["ground_image_id"] for row in category}),
                        "same_surface_category_pairs": sum(bool(row["same_surface_form"]) for row in category)})
    write(args.output_dir / "cohort_yield.csv", summary)
    print(f"mixed_images={len(mixed)}")


if __name__ == "__main__": main()
