"""Accuracy test: compare IVF search vs brute force KNN (ground truth).

Loads references.json.gz, vectorizes example payloads, runs brute force k=5
euclidean, then compares with IVF results at different nprobe values.

Usage: python3 scripts/accuracy_test.py
"""

import gzip
import json
import os
import sys
import time

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vectorizer import vectorize

REFS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "rinha-de-backend-2026", "resources", "references.json.gz")
PAYLOADS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "rinha-de-backend-2026", "resources", "example-payloads.json")
INDEX_PATH = os.environ.get("INDEX_PATH", None)

Q_SCALE = 10000.0
K = 5


def load_references(path):
    print(f"Loading references from {path}...")
    t0 = time.time()
    with gzip.open(path, "rb") as f:
        data = json.loads(f.read())

    vectors = np.array([e["vector"] for e in data], dtype=np.float32)
    labels = np.array([1 if e["label"] == "fraud" else 0 for e in data], dtype=np.uint8)
    print(f"  Loaded {len(labels)} refs in {time.time() - t0:.1f}s")
    return vectors, labels


def brute_force_knn(query_f32, ref_vectors, ref_labels, k=5):
    """Exact KNN with euclidean distance (brute force). Returns fraud_count."""
    # ||q - r||^2 = ||q||^2 + ||r||^2 - 2*q.r
    q_sq = np.sum(query_f32 ** 2)
    dots = ref_vectors @ query_f32
    r_sq = np.sum(ref_vectors ** 2, axis=1)
    dists = q_sq + r_sq - 2 * dots

    top_k = np.argpartition(dists, k)[:k]
    fraud_count = int(ref_labels[top_k].sum())
    return fraud_count, dists[top_k]


def ivf_search(query_int16, index, nprobe):
    """Run IVF search with given nprobe."""
    return index.search(query_int16, nprobe=nprobe)


def main():
    # Load references
    ref_vectors, ref_labels = load_references(REFS_PATH)

    # Pre-compute ref squared norms for brute force
    print("Pre-computing reference norms...")
    ref_sq = np.sum(ref_vectors ** 2, axis=1)

    # Load example payloads
    with open(PAYLOADS_PATH) as f:
        payloads = json.load(f)
    print(f"Loaded {len(payloads)} example payloads\n")

    # Vectorize all payloads
    queries_f32 = []
    queries_int16 = []
    for p in payloads:
        q_int16 = vectorize(p)
        q_f32 = q_int16.astype(np.float32) / Q_SCALE
        queries_f32.append(q_f32)
        queries_int16.append(q_int16)

    # Brute force (ground truth)
    print("Running brute force KNN (ground truth)...")
    t0 = time.time()
    bf_results = []
    for i, q in enumerate(queries_f32):
        fraud_count, _ = brute_force_knn(q, ref_vectors, ref_labels)
        score = fraud_count / 5.0
        approved = score < 0.6
        bf_results.append({"fraud_count": fraud_count, "score": score, "approved": approved})
    bf_time = time.time() - t0
    print(f"  Done in {bf_time:.1f}s ({bf_time/len(payloads)*1000:.0f}ms per query)\n")

    # Try loading IVF index
    index = None
    if INDEX_PATH and os.path.exists(INDEX_PATH):
        from index import IVFIndex
        index = IVFIndex(INDEX_PATH)
        print(f"Loaded IVF index: {index.n} vectors, {index.k} clusters\n")
    else:
        # Try to find it in a running container
        print("No local index found. Will compare brute force only.\n")
        print("To test IVF: INDEX_PATH=/path/to/index.bin python3 scripts/accuracy_test.py\n")

    # Print brute force results
    bf_approved = sum(1 for r in bf_results if r["approved"])
    bf_denied = len(bf_results) - bf_approved
    print(f"Brute force results: {bf_approved} approved, {bf_denied} denied")
    print()

    if index is None:
        # Just print brute force results
        print(f"{'#':>3} {'ID':>16} {'BF fraud':>8} {'BF score':>8} {'BF approved':>11}")
        print("-" * 55)
        for i, (p, bf) in enumerate(zip(payloads, bf_results)):
            print(f"{i:3d} {p['id']:>16} {bf['fraud_count']:8d} {bf['score']:8.1f} {str(bf['approved']):>11}")
        return

    # Compare IVF at different nprobe values
    nprobe_values = [1, 2, 3, 4, 6, 8, 12, 16]

    print(f"{'nprobe':>6} | {'Matches':>7} | {'FP':>3} | {'FN':>3} | {'Errors':>6} | {'Weighted':>8} | {'Avg ms':>6}")
    print("-" * 65)

    for nprobe in nprobe_values:
        matches = 0
        fp = 0  # false positive: BF says legit, IVF says fraud
        fn = 0  # false negative: BF says fraud, IVF says legit
        t0 = time.time()

        for i, q in enumerate(queries_int16):
            ivf_fraud = ivf_search(q, index, nprobe)
            ivf_approved = (ivf_fraud / 5.0) < 0.6
            bf_approved = bf_results[i]["approved"]

            if ivf_approved == bf_approved:
                matches += 1
            elif bf_approved and not ivf_approved:
                fp += 1  # legit blocked
            else:
                fn += 1  # fraud approved

        elapsed = time.time() - t0
        avg_ms = elapsed / len(payloads) * 1000
        weighted = fp * 1 + fn * 3
        n = len(payloads)
        print(f"{nprobe:6d} | {matches:3d}/{n:<3d} | {fp:3d} | {fn:3d} | {fp+fn:6d} | {weighted:8d} | {avg_ms:6.1f}")

    # Detailed comparison for default nprobe
    print(f"\n--- Detailed comparison (nprobe=3 vs brute force) ---\n")
    print(f"{'#':>3} {'ID':>16} {'BF':>4} {'IVF':>4} {'Match':>5}")
    print("-" * 50)
    mismatches = 0
    for i, (p, bf) in enumerate(zip(payloads, bf_results)):
        ivf_fraud = ivf_search(queries_int16[i], index, 3)
        ivf_score = ivf_fraud / 5.0
        bf_score = bf["score"]
        match = "OK" if (ivf_score < 0.6) == (bf_score < 0.6) else "MISS"
        if match == "MISS":
            mismatches += 1
        print(f"{i:3d} {p['id']:>16} {bf_score:4.1f} {ivf_score:4.1f} {match:>5}")

    print(f"\nTotal mismatches: {mismatches}/{len(payloads)}")


if __name__ == "__main__":
    main()
