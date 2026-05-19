// Stress test: ramps from 1 to 1200 req/s over 60 seconds

import http from 'k6/http';
import { Counter } from 'k6/metrics';
import { SharedArray } from 'k6/data';
import exec from 'k6/execution';

const testData = new SharedArray('test-data', function () {
    return JSON.parse(open('../../rinha-de-backend-2026/test/test-data.json')).entries;
});

const fpCount = new Counter('fp_count');
const fnCount = new Counter('fn_count');
const errorCount = new Counter('error_count');

export const options = {
    summaryTrendStats: ['p(99)', 'p(95)', 'p(50)', 'avg', 'min', 'max'],
    scenarios: {
        default: {
            executor: 'ramping-arrival-rate',
            startRate: 1,
            timeUnit: '1s',
            preAllocatedVUs: 150,
            maxVUs: 300,
            gracefulStop: '10s',
            stages: [
                { duration: '60s', target: 1200 },
            ],
        },
    },
};

export default function () {
    const idx = exec.scenario.iterationInTest % testData.length;
    const entry = testData[idx];

    const res = http.post(
        'http://localhost:9999/fraud-score',
        JSON.stringify(entry.request),
        { headers: { 'Content-Type': 'application/json' }, timeout: '2001ms' }
    );

    if (res.status === 200) {
        const body = JSON.parse(res.body);
        if (entry.expected_approved !== body.approved) {
            if (body.approved) fnCount.add(1);
            else fpCount.add(1);
        }
    } else {
        errorCount.add(1);
    }
}

export function handleSummary(data) {
    const d = data.metrics.http_req_duration.values;
    const p99 = d['p(99)'];
    const p95 = d['p(95)'];
    const p50 = d['p(50)'];
    const avg = d['avg'];
    const min = d['min'];
    const max = d['max'];
    const fp = data.metrics.fp_count ? data.metrics.fp_count.values.count : 0;
    const fn = data.metrics.fn_count ? data.metrics.fn_count.values.count : 0;
    const errs = data.metrics.error_count ? data.metrics.error_count.values.count : 0;
    const iters = data.metrics.iterations.values.count;
    const duration = data.state.testRunDurationMs / 1000;
    const rps = (iters / duration).toFixed(0);

    // Score estimation (same formula as official test)
    const N = iters;
    const E = (fp * 1) + (fn * 3) + (errs * 5);
    const epsilon = N > 0 ? E / N : 0;
    const failures = fp + fn + errs;
    const failureRate = N > 0 ? failures / N : 0;

    let p99Score = p99 > 2000 ? -3000 : 1000 * Math.log10(1000 / Math.max(p99, 1));
    let detScore;
    if (failureRate > 0.15) {
        detScore = -3000;
    } else {
        const rate = 1000 * Math.log10(1 / Math.max(epsilon, 0.001));
        const penalty = -300 * Math.log10(1 + E);
        detScore = rate + penalty;
    }
    const totalScore = p99Score + detScore;

    const summary = `
=== STRESS 1200 (1→1200 req/s, 60s) ===
Requests:   ${iters} (${rps} req/s avg)
Duration:   ${duration.toFixed(1)}s

Latency:
  min:      ${min.toFixed(2)}ms
  p50:      ${p50.toFixed(2)}ms
  avg:      ${avg.toFixed(2)}ms
  p95:      ${p95.toFixed(2)}ms
  p99:      ${p99.toFixed(2)}ms
  max:      ${max.toFixed(2)}ms

Accuracy:
  FP:       ${fp}
  FN:       ${fn}
  HTTP Err: ${errs}

Score (estimated):
  p99:      ${p99Score.toFixed(2)}
  detect:   ${detScore.toFixed(2)}
  TOTAL:    ${totalScore.toFixed(2)}
`;
    return { stdout: summary };
}
