"""Smoke test: validates API contract against example payloads."""

import json
import sys
import time
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9999"


def test_ready():
    """GET /ready must return 2xx."""
    url = f"{BASE_URL}/ready"
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        assert 200 <= resp.status < 300, f"/ready returned {resp.status}"
        print(f"  /ready: {resp.status} OK")
    except Exception as e:
        print(f"  /ready: FAIL - {e}")
        return False
    return True


def test_fraud_score(payload: dict, idx: int):
    """POST /fraud-score must return valid response."""
    url = f"{BASE_URL}/fraud-score"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=10)
        latency_ms = (time.time() - t0) * 1000
        assert resp.status == 200, f"Status {resp.status}"

        result = json.loads(resp.read())

        # Validate response format
        assert "approved" in result, "Missing 'approved' field"
        assert "fraud_score" in result, "Missing 'fraud_score' field"
        assert isinstance(result["approved"], bool), f"'approved' not bool: {type(result['approved'])}"
        assert isinstance(result["fraud_score"], (int, float)), f"'fraud_score' not number"
        assert 0.0 <= result["fraud_score"] <= 1.0, f"fraud_score out of range: {result['fraud_score']}"

        # Validate consistency
        if result["fraud_score"] < 0.6:
            assert result["approved"] is True, f"fraud_score {result['fraud_score']} < 0.6 but approved=false"
        else:
            assert result["approved"] is False, f"fraud_score {result['fraud_score']} >= 0.6 but approved=true"

        print(f"  payload[{idx}] (id={payload['id']}): "
              f"approved={result['approved']}, score={result['fraud_score']}, "
              f"latency={latency_ms:.1f}ms OK")
        return True

    except Exception as e:
        print(f"  payload[{idx}] (id={payload.get('id', '?')}): FAIL - {e}")
        return False


def main():
    print(f"Smoke test against {BASE_URL}\n")

    # Load example payloads
    paths = [
        "../rinha-de-backend-2026/resources/example-payloads.json",
        "../../rinha-de-backend-2026/resources/example-payloads.json",
        "resources/example-payloads.json",
        "../resources/example-payloads.json",
    ]
    payloads = None
    for p in paths:
        try:
            with open(p) as f:
                payloads = json.load(f)
            break
        except FileNotFoundError:
            continue
    if payloads is None:
        print("ERROR: example-payloads.json not found")
        sys.exit(1)

    passed = 0
    failed = 0

    # Test /ready
    print("[1/2] Testing GET /ready")
    if test_ready():
        passed += 1
    else:
        failed += 1

    # Test /fraud-score with each payload
    print(f"\n[2/2] Testing POST /fraud-score ({len(payloads)} payloads)")
    for i, payload in enumerate(payloads):
        if test_fraud_score(payload, i):
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
