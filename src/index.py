"""IVF index loader and search via mmap + NumPy. Optimized for low p99."""

import mmap
import os
import struct

import numpy as np

MAGIC = b"RIVF"
DIMS = 14
HEADER_SIZE = 32
K_NEIGHBORS = 5
_INT64_MAX = np.iinfo(np.int64).max


class IVFIndex:
    """Memory-mapped IVF index for K-NN search."""

    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDONLY)
        file_size = os.fstat(self.fd).st_size
        self.mm = mmap.mmap(self.fd, file_size, access=mmap.ACCESS_READ)

        # Parse header
        magic, _, n, k, dims, flags = struct.unpack_from("<4sIIIII", self.mm, 0)
        assert magic == MAGIC, f"Bad magic: {magic}"
        assert dims == DIMS, f"Bad dims: {dims}"

        self.n = n
        self.k = k

        offset = HEADER_SIZE

        # Centroids: k * 14 * int16 → int32
        self.centroids_i32 = np.frombuffer(
            self.mm, dtype=np.int16, count=k * DIMS, offset=offset
        ).reshape(k, DIMS).astype(np.int32)
        offset += k * DIMS * 2

        # Bbox min/max: k * 14 * int16 → int32
        bbox_bytes = k * DIMS * 2
        self.bbox_min_i32 = np.frombuffer(
            self.mm, dtype=np.int16, count=k * DIMS, offset=offset
        ).reshape(k, DIMS).astype(np.int32)
        offset += bbox_bytes
        self.bbox_max_i32 = np.frombuffer(
            self.mm, dtype=np.int16, count=k * DIMS, offset=offset
        ).reshape(k, DIMS).astype(np.int32)
        offset += bbox_bytes

        # Offsets: (k+1) * uint32
        self.offsets = np.frombuffer(
            self.mm, dtype=np.uint32, count=k + 1, offset=offset
        ).copy()
        offset += (k + 1) * 4

        # Vector squared norms: n * int32 → int64 (avoid per-request cast)
        self.vector_sq = np.frombuffer(
            self.mm, dtype=np.int32, count=n, offset=offset
        ).astype(np.int64)
        offset += n * 4

        # Labels: n * uint8 (mmap view)
        self.labels = np.frombuffer(self.mm, dtype=np.uint8, count=n, offset=offset)
        offset += n

        # Vectors: n * 14 * int16 (mmap view)
        self.vectors = np.frombuffer(
            self.mm, dtype=np.int16, count=n * DIMS, offset=offset
        ).reshape(n, DIMS)

        # Pre-compute centroid squared norms
        self.centroids_sq = np.sum(self.centroids_i32 * self.centroids_i32, axis=1)

        # Reusable per-request buffers
        self._query_i32 = np.empty(DIMS, dtype=np.int32)
        self._top5_dists = np.empty(K_NEIGHBORS, dtype=np.int64)
        self._top5_labels = np.empty(K_NEIGHBORS, dtype=np.uint8)
        self._merge_dists = np.empty(K_NEIGHBORS * 2, dtype=np.int64)
        self._merge_labels = np.empty(K_NEIGHBORS * 2, dtype=np.uint8)

    def search(self, query: np.ndarray, nprobe: int = 7) -> int:
        """Find 5 nearest neighbors. Inlined scan, minimal allocations."""
        # Query setup
        q = self._query_i32
        np.copyto(q, query, casting="unsafe")
        q_sq = int(q @ q)

        # Find nearest clusters
        qc = self.centroids_i32 @ q
        centroid_dists = self.centroids_sq + q_sq - 2 * qc
        best_clusters = np.argpartition(centroid_dists, nprobe)[:nprobe]

        # Reset top-5
        top5_d = self._top5_dists
        top5_l = self._top5_labels
        top5_d.fill(_INT64_MAX)
        top5_l.fill(0)
        md = self._merge_dists
        ml = self._merge_labels

        offsets = self.offsets
        vectors = self.vectors
        vector_sq = self.vector_sq
        labels = self.labels

        # Scan each cluster — inlined for fewer Python→C transitions
        for c_idx in best_clusters:
            c = int(c_idx)
            s = int(offsets[c])
            e = int(offsets[c + 1])
            if s >= e:
                continue

            # Matmul: int16 @ int32, NumPy casts internally
            dot = vectors[s:e] @ q

            # Distance: ||a-b||² = ||a||² + ||b||² - 2*a·b
            # Avoid dot.astype(int64) allocation — use np.multiply with dtype
            dists = vector_sq[s:e] + q_sq
            np.subtract(dists, np.multiply(dot, 2, dtype=np.int64), out=dists)

            # Filter candidates below worst
            worst = top5_d.max()
            mask = dists < worst
            if not mask.any():
                continue

            cand_dists = dists[mask]
            cand_labels = labels[s:e][mask]

            # Pre-reduce to at most 5
            n_cand = len(cand_dists)
            if n_cand > K_NEIGHBORS:
                idx = np.argpartition(cand_dists, K_NEIGHBORS)[:K_NEIGHBORS]
                cand_dists = cand_dists[idx]
                cand_labels = cand_labels[idx]
                n_cand = K_NEIGHBORS

            # Merge with top-5 using pre-allocated buffers + argsort (faster for ≤10 items)
            md[:K_NEIGHBORS] = top5_d
            ml[:K_NEIGHBORS] = top5_l
            md[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_dists
            ml[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_labels
            total = K_NEIGHBORS + n_cand
            idx = md[:total].argsort()[:K_NEIGHBORS]
            top5_d[:] = md[idx]
            top5_l[:] = ml[idx]

        return int(top5_l.sum())

    def search_adaptive(self, query: np.ndarray, nprobe: int = 2,
                        repair_min: int = 1, repair_max: int = 4,
                        max_repair: int = 4) -> int:
        """Search with adaptive repair for borderline cases."""
        q = self._query_i32
        np.copyto(q, query, casting="unsafe")
        q_sq = int(q @ q)

        qc = self.centroids_i32 @ q
        centroid_dists = self.centroids_sq + q_sq - 2 * qc

        # Phase 1: only sort the nprobe closest clusters (cheap)
        best_clusters = np.argpartition(centroid_dists, nprobe)[:nprobe]

        top5_d = self._top5_dists
        top5_l = self._top5_labels
        top5_d.fill(_INT64_MAX)
        top5_l.fill(0)
        md = self._merge_dists
        ml = self._merge_labels

        offsets = self.offsets
        vectors = self.vectors
        vector_sq = self.vector_sq
        labels = self.labels

        # Initial probe (unsorted — order doesn't matter for correctness)
        for c_idx in best_clusters:
            c = int(c_idx)
            s = int(offsets[c])
            e = int(offsets[c + 1])
            if s >= e:
                continue

            dot = vectors[s:e] @ q
            dists = vector_sq[s:e] + q_sq
            np.subtract(dists, np.multiply(dot, 2, dtype=np.int64), out=dists)

            worst = top5_d.max()
            mask = dists < worst
            if not mask.any():
                continue

            cand_dists = dists[mask]
            cand_labels = labels[s:e][mask]
            n_cand = len(cand_dists)
            if n_cand > K_NEIGHBORS:
                idx = np.argpartition(cand_dists, K_NEIGHBORS)[:K_NEIGHBORS]
                cand_dists = cand_dists[idx]
                cand_labels = cand_labels[idx]
                n_cand = K_NEIGHBORS

            md[:K_NEIGHBORS] = top5_d
            ml[:K_NEIGHBORS] = top5_l
            md[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_dists
            ml[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_labels
            idx = md[:K_NEIGHBORS + n_cand].argsort()[:K_NEIGHBORS]
            top5_d[:] = md[idx]
            top5_l[:] = ml[idx]

        fraud_count = int(top5_l.sum())
        if fraud_count < repair_min or fraud_count > repair_max:
            return fraud_count

        # Phase 2 (repair): bbox filter ALL clusters at once (no argpartition)
        # Vectorized bbox lower-bound distances for all k clusters
        below_all = self.bbox_min_i32 - q
        above_all = q - self.bbox_max_i32
        d_all = np.maximum(below_all, 0) + np.maximum(above_all, 0)
        bbox_dists = np.sum(d_all * d_all, axis=1)

        # Exclude initial probe clusters and filter by worst distance
        bbox_dists[best_clusters] = np.iinfo(bbox_dists.dtype).max
        worst = top5_d.max()
        candidates = np.where(bbox_dists < worst)[0]

        for c_idx in candidates:
            c = int(c_idx)
            s = int(offsets[c])
            e = int(offsets[c + 1])
            if s >= e:
                continue

            dot = vectors[s:e] @ q
            dists = vector_sq[s:e] + q_sq
            np.subtract(dists, np.multiply(dot, 2, dtype=np.int64), out=dists)

            worst = top5_d.max()
            mask = dists < worst
            if not mask.any():
                continue

            cand_dists = dists[mask]
            cand_labels = labels[s:e][mask]
            n_cand = len(cand_dists)
            if n_cand > K_NEIGHBORS:
                idx = np.argpartition(cand_dists, K_NEIGHBORS)[:K_NEIGHBORS]
                cand_dists = cand_dists[idx]
                cand_labels = cand_labels[idx]
                n_cand = K_NEIGHBORS

            md[:K_NEIGHBORS] = top5_d
            ml[:K_NEIGHBORS] = top5_l
            md[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_dists
            ml[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_labels
            idx = md[:K_NEIGHBORS + n_cand].argsort()[:K_NEIGHBORS]
            top5_d[:] = md[idx]
            top5_l[:] = ml[idx]

        return int(top5_l.sum())

    def close(self):
        self.mm.close()
        os.close(self.fd)
