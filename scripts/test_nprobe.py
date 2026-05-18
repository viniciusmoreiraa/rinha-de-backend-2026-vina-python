"""Test error cases with different nprobe values using local index."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vectorizer import vectorize
from index import IVFIndex

INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.bin")
ERRORS_PATH = os.path.join(os.path.dirname(__file__), "errors_detail.json")

index = IVFIndex(INDEX_PATH)
print(f"Index: {index.n} vectors, {index.k} clusters\n")

with open(ERRORS_PATH) as f:
    errors = json.load(f)

nprobe_values = [5, 9, 20, 50, 100, 200, 500]

# Header
header = f"{'#':>2} {'Idx':>6} {'Type':>4} {'Exp':>3}"
for np_val in nprobe_values:
    header += f" | np={np_val:>3}"
print(header)
print("-" * len(header))

for err in errors:
    payload = err["request"]
    expected_approved = err["expected_approved"]
    expected_score = 0.6 if not expected_approved else 0.4  # approximate

    q = vectorize(payload).copy()

    line = f"{err.get('type','?'):>2} {err['index']:6d} {err.get('type','?'):>4} {'F' if not expected_approved else 'T':>3}"

    for np_val in nprobe_values:
        fraud_count = index.search(q, nprobe=np_val)
        score = fraud_count / 5.0
        approved = score < 0.6
        match = "OK" if approved == expected_approved else "X "
        line += f" | {fraud_count}/5={score:.1f} {match}"

    print(line)

# Summary
print(f"\n--- Summary ---")
for np_val in nprobe_values:
    misses = 0
    for err in errors:
        q = vectorize(err["request"]).copy()
        fraud_count = index.search(q, nprobe=np_val)
        approved = (fraud_count / 5.0) < 0.6
        if approved != err["expected_approved"]:
            misses += 1
    print(f"nprobe={np_val:>3}: {misses} errors out of {len(errors)}")
