"""Send all test-data entries and report misclassified cases."""

import json
import sys
import time
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9999"
TEST_DATA = "../../rinha-de-backend-2026/test/test-data.json"

print(f"Loading {TEST_DATA}...")
with open(TEST_DATA) as f:
    data = json.load(f)

entries = data["entries"]
total = len(entries)
errors = []

print(f"Sending {total} requests to {BASE_URL}/fraud-score ...")
t0 = time.time()

for i, entry in enumerate(entries):
    payload = json.dumps(entry["request"]).encode()
    expected_approved = entry["expected_approved"]

    req = urllib.request.Request(
        f"{BASE_URL}/fraud-score",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
    except Exception as e:
        errors.append({"index": i, "error": str(e), "expected_approved": expected_approved})
        continue

    actual_approved = body["approved"]
    if actual_approved != expected_approved:
        error_type = "FP" if expected_approved and not actual_approved else "FN"
        errors.append({
            "index": i,
            "type": error_type,
            "expected_approved": expected_approved,
            "actual_approved": actual_approved,
            "fraud_score": body["fraud_score"],
        })

    if (i + 1) % 5000 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{total} ({len(errors)} errors so far) [{elapsed:.1f}s]")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")
print(f"Total: {total}, Errors: {len(errors)}")

if errors:
    fp = sum(1 for e in errors if e.get("type") == "FP")
    fn = sum(1 for e in errors if e.get("type") == "FN")
    print(f"  FP: {fp}, FN: {fn}")
    for e in errors:
        print(json.dumps(e))
else:
    print("Perfect accuracy!")
