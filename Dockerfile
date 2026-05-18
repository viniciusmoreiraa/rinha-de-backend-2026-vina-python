### Stage 1: Build the IVF index offline
FROM --platform=linux/amd64 python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
COPY requirements-build.txt .
RUN pip install --no-cache-dir -r requirements-build.txt

# Copy build script
COPY src/build_index.py .

# Download reference data
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /build/references.json.gz \
    https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz

# Build index (K=4096 clusters, sample=80000)
RUN mkdir -p /index && \
    python build_index.py /build/references.json.gz /index/index.bin 4096 80000

### Stage 2: Runtime
FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/

# Copy pre-built index
COPY --from=builder /index/index.bin /index/index.bin

# Environment
ENV INDEX_PATH=/index/index.bin
ENV NPROBE=5
ENV ADAPTIVE=1
ENV REPAIR_MIN=2
ENV REPAIR_MAX=3
ENV MAX_REPAIR=3
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=1

CMD ["uvicorn", "server:app", "--uds", "/sockets/api.sock", \
     "--workers", "1", \
     "--no-access-log", "--log-level", "error", \
     "--loop", "uvloop", "--http", "httptools"]
