"""Diagnose error cases: vectorizer bug vs quantization error.

Loads the 8 error payloads, vectorizes them, does brute force KNN on
float32 references, and compares with expected_approved.

If brute force matches expected → problem is quantization (int16 rounding)
If brute force mismatches → problem is in the vectorizer
"""

import gzip
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vectorizer import vectorize

REFS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "rinha-de-backend-2026", "resources", "references.json.gz")
ERRORS_PATH = os.path.join(os.path.dirname(__file__), "errors_detail.json")
Q_SCALE = 10000.0
K = 5


def main():
    # Load references
    print("Loading references...")
    t0 = time.time()
    with gzip.open(REFS_PATH, "rb") as f:
        ref_data = json.loads(f.read())

    ref_vectors_f32 = np.array([e["vector"] for e in ref_data], dtype=np.float32)
    ref_labels = np.array([1 if e["label"] == "fraud" else 0 for e in ref_data], dtype=np.uint8)
    ref_sq = np.sum(ref_vectors_f32 ** 2, axis=1)
    print(f"  Loaded {len(ref_labels)} refs in {time.time() - t0:.1f}s")

    # Also prepare int16 quantized refs (same as index)
    ref_vectors_i16 = np.clip(np.round(ref_vectors_f32 * Q_SCALE), -10000, 10000).astype(np.int16)
    ref_vectors_i32 = ref_vectors_i16.astype(np.int32)
    ref_sq_i32 = np.sum(ref_vectors_i32 ** 2, axis=1)

    # Load error cases
    with open(ERRORS_PATH) as f:
        errors = json.load(f)

    print(f"\nDiagnosing {len(errors)} error cases...\n")
    print(f"{'#':>2} {'Index':>6} {'Type':>4} {'Expected':>8} | {'BF-f32':>6} {'BF-i16':>6} {'Our vec':>7} | {'Diagnosis'}")
    print("-" * 80)

    for err in errors:
        payload = err["request"]
        expected_approved = err["expected_approved"]
        error_type = err.get("type", "?")
        idx = err["index"]

        # Vectorize with our vectorizer (int16)
        q_i16 = vectorize(payload).copy()
        q_f32 = q_i16.astype(np.float32) / Q_SCALE

        # Brute force on float32 references with our float32 query
        dots_f32 = ref_vectors_f32 @ q_f32
        dists_f32 = np.sum(q_f32 ** 2) + ref_sq - 2 * dots_f32
        top5_f32 = np.argpartition(dists_f32, K)[:K]
        fraud_count_f32 = int(ref_labels[top5_f32].sum())
        approved_f32 = (fraud_count_f32 / 5.0) < 0.6

        # Brute force on int16 references with our int16 query (same math as IVF)
        q_i32 = q_i16.astype(np.int32)
        q_sq = int(q_i32 @ q_i32)
        dots_i32 = ref_vectors_i32 @ q_i32
        dists_i32 = ref_sq_i32 + q_sq - 2 * dots_i32.astype(np.int64)
        top5_i32 = np.argpartition(dists_i32, K)[:K]
        fraud_count_i32 = int(ref_labels[top5_i32].sum())
        approved_i32 = (fraud_count_i32 / 5.0) < 0.6

        # Diagnosis
        if approved_f32 == expected_approved and approved_i32 == expected_approved:
            diag = "IVF miss (search bug)"
        elif approved_f32 == expected_approved and approved_i32 != expected_approved:
            diag = "QUANTIZATION error (int16 rounding)"
        elif approved_f32 != expected_approved:
            diag = "VECTORIZER bug"
        else:
            diag = "???"

        # Show the fraud counts for detail
        score_f32 = fraud_count_f32 / 5.0
        score_i32 = fraud_count_i32 / 5.0

        print(f"{error_type:>2} {idx:6d} {error_type:>4} exp={'T' if expected_approved else 'F':>1} | "
              f"f32={fraud_count_f32}({score_f32:.1f}) "
              f"i16={fraud_count_i32}({score_i32:.1f}) "
              f"api={err['fraud_score']:.1f} | {diag}")

        # Show neighbor details for mismatches
        if diag != "IVF miss (search bug)":
            # Show top-5 neighbors in both spaces
            top5_f32_sorted = top5_f32[np.argsort(dists_f32[top5_f32])]
            top5_i32_sorted = top5_i32[np.argsort(dists_i32[top5_i32])]
            print(f"     f32 neighbors: {list(top5_f32_sorted)} labels={list(ref_labels[top5_f32_sorted])}")
            print(f"     i16 neighbors: {list(top5_i32_sorted)} labels={list(ref_labels[top5_i32_sorted])}")

            # Also show 6th-10th neighbors to see how close the boundary is
            top10_f32 = np.argpartition(dists_f32, 10)[:10]
            top10_f32_sorted = top10_f32[np.argsort(dists_f32[top10_f32])]
            print(f"     f32 top-10: idx={list(top10_f32_sorted)} labels={list(ref_labels[top10_f32_sorted])}")
            dists_top10 = dists_f32[top10_f32_sorted]
            print(f"     f32 dists:  {[f'{d:.6f}' for d in dists_top10]}")


if __name__ == "__main__":
    main()
