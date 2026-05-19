"""Test different bbox consecutive miss thresholds."""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vectorizer import vectorize
from index import IVFIndex, K_NEIGHBORS, _INT64_MAX

INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.bin")
ERRORS_PATH = os.path.join(os.path.dirname(__file__), "errors_detail.json")

index = IVFIndex(INDEX_PATH)

with open(ERRORS_PATH) as f:
    errors = json.load(f)

# Manually implement search_adaptive with configurable bbox_misses threshold
def search_adaptive_custom(idx, query, nprobe, repair_min, repair_max, max_repair, max_bbox_misses):
    q = idx._query_i32
    np.copyto(q, query, casting="unsafe")
    q_sq = int(q @ q)

    qc = idx.centroids_i32 @ q
    centroid_dists = idx.centroids_sq + q_sq - 2 * qc

    total_probe = min(nprobe + max_repair, idx.k)
    if total_probe >= idx.k:
        top_sorted = np.argsort(centroid_dists)
    else:
        top_clusters = np.argpartition(centroid_dists, total_probe)[:total_probe]
        top_sorted = top_clusters[np.argsort(centroid_dists[top_clusters])]

    top5_d = idx._top5_dists
    top5_l = idx._top5_labels
    top5_d.fill(_INT64_MAX)
    top5_l.fill(0)
    md = idx._merge_dists
    ml = idx._merge_labels

    offsets = idx.offsets
    vectors = idx.vectors
    vector_sq = idx.vector_sq
    labels = idx.labels

    for c_idx in top_sorted[:nprobe]:
        c = int(c_idx)
        s = int(offsets[c]); e = int(offsets[c + 1])
        if s >= e: continue
        dot = vectors[s:e] @ q
        dists = vector_sq[s:e] + q_sq
        np.subtract(dists, np.multiply(dot, 2, dtype=np.int64), out=dists)
        worst = top5_d.max()
        mask = dists < worst
        if not mask.any(): continue
        cand_dists = dists[mask]; cand_labels = labels[s:e][mask]
        n_cand = len(cand_dists)
        if n_cand > K_NEIGHBORS:
            ix = np.argpartition(cand_dists, K_NEIGHBORS)[:K_NEIGHBORS]
            cand_dists = cand_dists[ix]; cand_labels = cand_labels[ix]; n_cand = K_NEIGHBORS
        md[:K_NEIGHBORS] = top5_d; ml[:K_NEIGHBORS] = top5_l
        md[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_dists
        ml[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_labels
        ix = md[:K_NEIGHBORS + n_cand].argsort()[:K_NEIGHBORS]
        top5_d[:] = md[ix]; top5_l[:] = ml[ix]

    fraud_count = int(top5_l.sum())
    if fraud_count < repair_min or fraud_count > repair_max:
        return fraud_count

    bbox_misses = 0
    for c_idx in top_sorted[nprobe:]:
        c = int(c_idx)
        bmin = idx.bbox_min_i32[c]; bmax = idx.bbox_max_i32[c]
        below = bmin - q; above = q - bmax
        d = np.maximum(below, 0) + np.maximum(above, 0)
        if int(np.sum(d * d)) >= top5_d.max():
            bbox_misses += 1
            if max_bbox_misses > 0 and bbox_misses >= max_bbox_misses:
                break
            continue
        bbox_misses = 0
        s = int(offsets[c]); e = int(offsets[c + 1])
        if s >= e: continue
        dot = vectors[s:e] @ q
        dists = vector_sq[s:e] + q_sq
        np.subtract(dists, np.multiply(dot, 2, dtype=np.int64), out=dists)
        worst = top5_d.max()
        mask = dists < worst
        if not mask.any(): continue
        cand_dists = dists[mask]; cand_labels = labels[s:e][mask]
        n_cand = len(cand_dists)
        if n_cand > K_NEIGHBORS:
            ix = np.argpartition(cand_dists, K_NEIGHBORS)[:K_NEIGHBORS]
            cand_dists = cand_dists[ix]; cand_labels = cand_labels[ix]; n_cand = K_NEIGHBORS
        md[:K_NEIGHBORS] = top5_d; ml[:K_NEIGHBORS] = top5_l
        md[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_dists
        ml[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_labels
        ix = md[:K_NEIGHBORS + n_cand].argsort()[:K_NEIGHBORS]
        top5_d[:] = md[ix]; top5_l[:] = ml[ix]

    return int(top5_l.sum())


thresholds = [1, 4, 8, 16, 32, 64, 0]  # 0 = no limit (pure continue)

for thresh in thresholds:
    label = f"unlimited" if thresh == 0 else f"{thresh}"
    misses = 0
    for err in errors:
        q = vectorize(err["request"]).copy()
        fraud_count = search_adaptive_custom(index, q, 9, 1, 4, 91, thresh)
        approved = (fraud_count / 5.0) < 0.6
        if approved != err["expected_approved"]:
            misses += 1
    print(f"bbox_misses_limit={label:>9}: {misses} errors")
