import http from 'k6/http';
import { SharedArray } from 'k6/data';

const testData = new SharedArray('test-data', function () {
    return JSON.parse(open('../../rinha-de-backend-2026/test/test-data.json')).entries;
});

export const options = {
    summaryTrendStats: ['avg', 'p(90)', 'p(95)', 'p(99)'],
    scenarios: {
        stress: {
            executor: 'constant-arrival-rate',
            rate: 1500,
            timeUnit: '1s',
            duration: '30s',
            preAllocatedVUs: 200,
            maxVUs: 500,
        },
    },
};

export default function () {
    const idx = Math.floor(Math.random() * testData.length);
    const entry = testData[idx];

    http.post(
        'http://localhost:9999/fraud-score',
        JSON.stringify(entry.request),
        { headers: { 'Content-Type': 'application/json' }, timeout: '2001ms' }
    );
}
