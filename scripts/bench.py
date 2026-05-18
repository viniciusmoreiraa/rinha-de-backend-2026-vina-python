"""Simple benchmark: measures latency distribution for /fraud-score."""

import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9999"
N_REQUESTS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 4


def send_request(payload: dict) -> tuple:
    """Send a single request. Returns (latency_ms, success, status_code)."""
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
        latency = (time.time() - t0) * 1000
        resp.read()
        return (latency, True, resp.status)
    except Exception as e:
        latency = (time.time() - t0) * 1000
        return (latency, False, str(e))


def main():
    print(f"Benchmark: {N_REQUESTS} requests, concurrency={CONCURRENCY}")
    print(f"Target: {BASE_URL}\n")

    # Load payloads
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

    # Warmup
    print("Warmup (5 requests)...")
    for i in range(min(5, len(payloads))):
        send_request(payloads[i % len(payloads)])

    # Benchmark
    print(f"Running {N_REQUESTS} requests...\n")
    latencies = []
    errors = 0

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = []
        for i in range(N_REQUESTS):
            payload = payloads[i % len(payloads)]
            futures.append(pool.submit(send_request, payload))

        for f in as_completed(futures):
            latency, success, status = f.result()
            if success:
                latencies.append(latency)
            else:
                errors += 1

    total_time = time.time() - t_start

    if not latencies:
        print("All requests failed!")
        sys.exit(1)

    latencies.sort()
    n = len(latencies)

    print(f"Results ({n} successful, {errors} errors):")
    print(f"  Total time:  {total_time:.2f}s")
    print(f"  Throughput:   {n / total_time:.1f} req/s")
    print(f"  Mean:         {statistics.mean(latencies):.2f}ms")
    print(f"  Median (p50): {latencies[n // 2]:.2f}ms")
    print(f"  p90:          {latencies[int(n * 0.9)]:.2f}ms")
    print(f"  p95:          {latencies[int(n * 0.95)]:.2f}ms")
    print(f"  p99:          {latencies[int(n * 0.99)]:.2f}ms")
    print(f"  Min:          {latencies[0]:.2f}ms")
    print(f"  Max:          {latencies[-1]:.2f}ms")

    # Score estimate
    p99 = latencies[int(n * 0.99)]
    if p99 > 2000:
        score_p99 = -3000
    elif p99 < 1:
        score_p99 = 3000
    else:
        import math
        score_p99 = 1000 * math.log10(1000 / max(p99, 1))

    print(f"\n  Estimated p99 score: {score_p99:.0f} (p99={p99:.1f}ms)")


if __name__ == "__main__":
    main()
