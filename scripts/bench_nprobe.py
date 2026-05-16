"""Benchmark different nprobe values for latency comparison."""

import json
import sys
import time
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9999"
N_REQUESTS = 1000
CONCURRENCY = 10

from concurrent.futures import ThreadPoolExecutor, as_completed


def send_request(payload):
    url = f"{BASE_URL}/fraud-score"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=10)
        latency = (time.time() - t0) * 1000
        data = json.loads(resp.read())
        return (latency, True, data)
    except Exception as e:
        return ((time.time() - t0) * 1000, False, str(e))


def run_bench(payloads):
    latencies = []
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = []
        for i in range(N_REQUESTS):
            futures.append(pool.submit(send_request, payloads[i % len(payloads)]))
        for f in as_completed(futures):
            lat, ok, data = f.result()
            if ok:
                latencies.append(lat)
                results.append(data)
    latencies.sort()
    n = len(latencies)
    return {
        "n": n,
        "mean": sum(latencies) / n,
        "median": latencies[n // 2],
        "p90": latencies[int(n * 0.9)],
        "p95": latencies[int(n * 0.95)],
        "p99": latencies[int(n * 0.99)],
        "results": results,
    }


def main():
    payloads_path = "../rinha-de-backend-2026/resources/example-payloads.json"
    try:
        with open(payloads_path) as f:
            payloads = json.load(f)
    except FileNotFoundError:
        payloads_path = "../../rinha-de-backend-2026/resources/example-payloads.json"
        with open(payloads_path) as f:
            payloads = json.load(f)

    # Warmup
    for i in range(10):
        send_request(payloads[i % len(payloads)])

    stats = run_bench(payloads)
    n = stats["n"]
    print(f"  {n} ok | mean={stats['mean']:.1f}ms | med={stats['median']:.1f}ms | "
          f"p90={stats['p90']:.1f}ms | p95={stats['p95']:.1f}ms | p99={stats['p99']:.1f}ms")

    # Count approvals for accuracy comparison
    approved = sum(1 for r in stats["results"] if r.get("approved"))
    denied = n - approved
    print(f"  approved={approved} denied={denied} ratio={denied/n*100:.1f}%")


if __name__ == "__main__":
    main()
