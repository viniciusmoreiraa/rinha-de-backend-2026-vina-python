"""Build IVF index from references.json.gz.

Usage: python build_index.py <references.json.gz> <output_index.bin> [n_clusters] [sample_size] [n_init_iters]

Runs offline during `docker build`. Produces a binary index file for mmap at runtime.
"""

import gzip
import json
import os
import struct
import sys
import time

import numpy as np

# Index file magic and version
MAGIC = b"RIVF"
VERSION = 1
DIMS = 14
Q_SCALE = 10000.0


def parse_references(path: str):
    """Parse references.json.gz into vectors and labels arrays."""
    print(f"Parsing {path}...")
    t0 = time.time()

    vectors = []
    labels = []

    with gzip.open(path, "rb") as f:
        data = json.loads(f.read())

    for entry in data:
        vectors.append(entry["vector"])
        labels.append(1 if entry["label"] == "fraud" else 0)

    vectors = np.array(vectors, dtype=np.float32)
    labels = np.array(labels, dtype=np.uint8)

    print(f"  Parsed {len(labels)} references in {time.time() - t0:.1f}s")
    print(f"  Fraud: {labels.sum()}, Legit: {len(labels) - labels.sum()}")
    return vectors, labels


def quantize(vectors: np.ndarray) -> np.ndarray:
    """Quantize float32 vectors to int16 (scale 10000)."""
    return np.clip(np.round(vectors * Q_SCALE), -10000, 10000).astype(np.int16)


def train_kmeans(vectors_f32: np.ndarray, n_clusters: int, sample_size: int):
    """Train K-means on a sample of vectors. Returns centroids as float32."""
    from sklearn.cluster import MiniBatchKMeans

    n = len(vectors_f32)
    if sample_size < n:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=sample_size, replace=False)
        sample = vectors_f32[indices]
    else:
        sample = vectors_f32

    print(f"Training K-means with K={n_clusters} on {len(sample)} samples...")
    t0 = time.time()

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        max_iter=20,
        n_init=1,
        random_state=42,
        verbose=0,
    )
    kmeans.fit(sample)
    print(f"  K-means done in {time.time() - t0:.1f}s")
    return kmeans.cluster_centers_.astype(np.float32)


def assign_clusters(vectors_q: np.ndarray, centroids_q: np.ndarray, n_clusters: int):
    """Assign each vector to its nearest centroid. Batch processing to limit memory."""
    print("Assigning vectors to clusters...")
    t0 = time.time()

    n = len(vectors_q)
    assignments = np.empty(n, dtype=np.int32)

    # Batch size scaled by K to avoid OOM: batch * K * 4 bytes < ~400MB
    batch_size = max(1000, min(50000, 400_000_000 // (n_clusters * 4)))
    centroids_i32 = centroids_q.astype(np.int32)
    b_sq = np.sum(centroids_i32 * centroids_i32, axis=1, keepdims=True).T  # (1, k)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = vectors_q[start:end].astype(np.int32)
        a_sq = np.sum(batch * batch, axis=1, keepdims=True)  # (batch, 1)
        ab = batch @ centroids_i32.T  # (batch, k)
        dists = a_sq + b_sq - 2 * ab  # (batch, k)
        assignments[start:end] = np.argmin(dists, axis=1)
        del batch, ab, dists

        if (start // batch_size) % 20 == 0:
            print(f"  Assigned {end}/{n}...")

    print(f"  Assignment done in {time.time() - t0:.1f}s")
    return assignments


def build_index(vectors_q: np.ndarray, labels: np.ndarray, centroids_q: np.ndarray,
                assignments: np.ndarray, n_clusters: int, output_path: str):
    """Build and write binary index file."""
    print(f"Building index to {output_path}...")
    t0 = time.time()

    n = len(vectors_q)

    # Sort vectors by cluster assignment
    order = np.argsort(assignments, kind="stable")
    sorted_vectors = vectors_q[order]
    sorted_labels = labels[order]
    sorted_assignments = assignments[order]

    # Compute cluster offsets and counts
    offsets = np.zeros(n_clusters + 1, dtype=np.uint32)
    counts = np.zeros(n_clusters, dtype=np.uint32)

    for c in range(n_clusters):
        mask = sorted_assignments == c
        counts[c] = mask.sum()

    # Cumulative offsets
    offsets[1:] = np.cumsum(counts)

    # Compute bounding boxes per cluster
    bbox_min = np.full((n_clusters, DIMS), 10000, dtype=np.int16)
    bbox_max = np.full((n_clusters, DIMS), -10000, dtype=np.int16)

    for c in range(n_clusters):
        start = offsets[c]
        end = offsets[c + 1]
        if end > start:
            cluster_vecs = sorted_vectors[start:end]
            bbox_min[c] = cluster_vecs.min(axis=0)
            bbox_max[c] = cluster_vecs.max(axis=0)

    # Pre-compute squared norms per vector: ||v||² (int32, fits since max = 14 * 10000² = 1.4B < 2.1B)
    vector_sq = np.sum(sorted_vectors.astype(np.int32) ** 2, axis=1).astype(np.int32)
    print(f"  Pre-computed {len(vector_sq)} squared norms")

    # Write binary file
    with open(output_path, "wb") as f:
        # Header: magic(4) + version(4) + n(4) + k(4) + dims(4) + flags(4) + padding(8) = 32 bytes
        # flags=1 means vector_sq is present
        header = struct.pack("<4sIIIII8x", MAGIC, VERSION, n, n_clusters, DIMS, 1)
        f.write(header)

        # Centroids: k * 14 * int16
        f.write(centroids_q.tobytes())

        # Bbox min: k * 14 * int16
        f.write(bbox_min.tobytes())

        # Bbox max: k * 14 * int16
        f.write(bbox_max.tobytes())

        # Offsets: (k+1) * uint32
        f.write(offsets.tobytes())

        # Vector squared norms: n * int32
        f.write(vector_sq.tobytes())

        # Labels: n * uint8 (ordered by cluster)
        f.write(sorted_labels.tobytes())

        # Vectors: n * 14 * int16 (ordered by cluster)
        f.write(sorted_vectors.tobytes())

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Index written: {file_size_mb:.1f} MB in {time.time() - t0:.1f}s")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <references.json.gz> <output.bin> [n_clusters] [sample_size]")
        sys.exit(1)

    refs_path = sys.argv[1]
    output_path = sys.argv[2]
    n_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    sample_size = int(sys.argv[4]) if len(sys.argv) > 4 else 50000

    total_t0 = time.time()

    # 1. Parse references
    vectors_f32, labels = parse_references(refs_path)

    # 2. Train K-means (on float32, before quantization for better centroids)
    centroids_f32 = train_kmeans(vectors_f32, n_clusters, sample_size)

    # 3. Quantize everything to int16
    vectors_q = quantize(vectors_f32)
    centroids_q = quantize(centroids_f32)

    # Free float32 vectors
    del vectors_f32, centroids_f32

    # 4. Assign all vectors to clusters
    assignments = assign_clusters(vectors_q, centroids_q, n_clusters)

    # 5. Build and write index
    build_index(vectors_q, labels, centroids_q, assignments, n_clusters, output_path)

    print(f"\nTotal build time: {time.time() - total_t0:.1f}s")


if __name__ == "__main__":
    main()
