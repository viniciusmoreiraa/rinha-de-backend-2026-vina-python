"""Test error cases with search_adaptive after the break→continue fix."""

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

configs = [
    ("np=9 plain", 9, False, 0, 0, 0),
    ("np=9 adapt(1-4,mr=91)", 9, True, 1, 4, 91),
    ("np=20 plain", 20, False, 0, 0, 0),
    ("np=50 plain", 50, False, 0, 0, 0),
    ("np=100 plain", 100, False, 0, 0, 0),
]

for label, nprobe, adaptive, rmin, rmax, max_rep in configs:
    misses = 0
    details = []
    for err in errors:
        q = vectorize(err["request"]).copy()
        if adaptive:
            fraud_count = index.search_adaptive(q, nprobe, rmin, rmax, max_rep)
        else:
            fraud_count = index.search(q, nprobe)
        approved = (fraud_count / 5.0) < 0.6
        ok = approved == err["expected_approved"]
        if not ok:
            misses += 1
            details.append(f"  idx={err['index']} got={fraud_count}/5 exp={'F' if not err['expected_approved'] else 'T'}")

    status = "PERFECT" if misses == 0 else f"{misses} errors"
    print(f"{label:>30}: {status}")
    for d in details:
        print(d)
