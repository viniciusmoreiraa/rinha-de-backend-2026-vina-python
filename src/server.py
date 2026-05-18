"""Raw ASGI app for Rinha de Backend 2026 fraud detection API."""

import os
import orjson
from index import IVFIndex
from vectorizer import vectorize

# Configuration
INDEX_PATH = os.environ.get("INDEX_PATH", "/index/index.bin")
NPROBE = int(os.environ.get("NPROBE", "2"))
USE_ADAPTIVE = os.environ.get("ADAPTIVE", "1") == "1"
REPAIR_MIN = int(os.environ.get("REPAIR_MIN", "1"))
REPAIR_MAX = int(os.environ.get("REPAIR_MAX", "4"))
MAX_REPAIR = int(os.environ.get("MAX_REPAIR", "4"))
DEBUG_ERRORS = os.environ.get("DEBUG_ERRORS", "0") == "1"

# Load index at module level BEFORE uvicorn creates socket
# (same as silent-index: index loads first, then listen)
INDEX = IVFIndex(INDEX_PATH)

# Resolve search function once at startup
if USE_ADAPTIVE:
    def _search(query):
        return INDEX.search_adaptive(query, NPROBE, REPAIR_MIN, REPAIR_MAX, MAX_REPAIR)
else:
    def _search(query):
        return INDEX.search(query, NPROBE)

# Pre-computed responses (only 6 possible outcomes)
BODIES = [
    b'{"approved":true,"fraud_score":0.0}',
    b'{"approved":true,"fraud_score":0.2}',
    b'{"approved":true,"fraud_score":0.4}',
    b'{"approved":false,"fraud_score":0.6}',
    b'{"approved":false,"fraud_score":0.8}',
    b'{"approved":false,"fraud_score":1.0}',
]

READY_BODY = b'{"status":"ok"}'

# Fallback: approved=false, score=0.6 (FP costs 1 vs FN costs 3)
FALLBACK_IDX = 3


def _make_start(body: bytes):
    return {
        "type": "http.response.start",
        "status": 200,
        "headers": [
            [b"content-type", b"application/json"],
            [b"content-length", str(len(body)).encode()],
        ],
    }


# Pre-computed ASGI start events — zero allocation per request
STARTS = [_make_start(b) for b in BODIES]
READY_START = _make_start(READY_BODY)

# Pre-computed body events
BODY_EVENTS = [{"type": "http.response.body", "body": b} for b in BODIES]
READY_BODY_EVENT = {"type": "http.response.body", "body": READY_BODY}


async def _read_body(receive) -> bytes:
    message = await receive()
    body = message.get("body", b"")
    if not message.get("more_body", False):
        return body
    parts = [body]
    while True:
        message = await receive()
        parts.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(parts)


async def app(scope, receive, send):
    if scope["type"] != "http":
        return

    path = scope["path"]
    method = scope["method"]

    # GET /ready
    if method == "GET" and path == "/ready":
        await send(READY_START)
        await send(READY_BODY_EVENT)
        return

    # POST /fraud-score
    if method == "POST" and path == "/fraud-score":
        try:
            body = await _read_body(receive)
            data = orjson.loads(body)
            query = vectorize(data)

            fraud_count = _search(query)

            await send(STARTS[fraud_count])
            await send(BODY_EVENTS[fraud_count])
        except Exception:
            if DEBUG_ERRORS:
                raise
            await send(STARTS[FALLBACK_IDX])
            await send(BODY_EVENTS[FALLBACK_IDX])
        return

    # Any other route: return fallback
    await send(STARTS[FALLBACK_IDX])
    await send(BODY_EVENTS[FALLBACK_IDX])
