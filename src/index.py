"""IVF index loader and search via mmap + NumPy."""

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

        # Centroids: k * 14 * int16
        self.centroids = np.frombuffer(self.mm, dtype=np.int16, count=k * DIMS, offset=offset).reshape(k, DIMS).copy()
        offset += k * DIMS * 2

        # Bbox min/max: k * 14 * int16 each → load as int32
        bbox_bytes = k * DIMS * 2
        self.bbox_min_i32 = np.frombuffer(self.mm, dtype=np.int16, count=k * DIMS, offset=offset).reshape(k, DIMS).astype(np.int32)
        offset += bbox_bytes
        self.bbox_max_i32 = np.frombuffer(self.mm, dtype=np.int16, count=k * DIMS, offset=offset).reshape(k, DIMS).astype(np.int32)
        offset += bbox_bytes

        # Offsets: (k+1) * uint32
        offsets_count = k + 1
        self.offsets = np.frombuffer(self.mm, dtype=np.uint32, count=offsets_count, offset=offset).copy()
        offset += offsets_count * 4

        # Vector squared norms: n * int32 → load as int64 to avoid per-request cast
        vsq_i32 = np.frombuffer(self.mm, dtype=np.int32, count=n, offset=offset)
        self.vector_sq = vsq_i32.astype(np.int64)
        offset += n * 4

        # Labels: n * uint8 (mmap view)
        self.labels = np.frombuffer(self.mm, dtype=np.uint8, count=n, offset=offset)
        offset += n

        # Vectors: n * 14 * int16 (mmap view)
        self.vectors = np.frombuffer(self.mm, dtype=np.int16, count=n * DIMS, offset=offset).reshape(n, DIMS)

        # Pre-compute centroids as int32
        self.centroids_i32 = self.centroids.astype(np.int32)
        self.centroids_sq = np.sum(self.centroids_i32 * self.centroids_i32, axis=1)

        # Reusable per-request buffers (single-threaded uvicorn)
        self._top5_dists = np.empty(K_NEIGHBORS, dtype=np.int64)
        self._top5_labels = np.empty(K_NEIGHBORS, dtype=np.uint8)
        self._query_i32 = np.empty(DIMS, dtype=np.int32)
        self._merge_dists = np.empty(K_NEIGHBORS * 2, dtype=np.int64)
        self._merge_labels = np.empty(K_NEIGHBORS * 2, dtype=np.uint8)

    def search(self, query: np.ndarray, nprobe: int = 2) -> int:
        """Find 5 nearest neighbors and return fraud count."""
        q = self._query_i32
        np.copyto(q, query, casting="unsafe")
        q_sq = q @ q

        # Find nearest clusters
        qc = self.centroids_i32 @ q
        centroid_dists = q_sq + self.centroids_sq - 2 * qc

        if nprobe >= self.k:
            best_clusters = np.arange(self.k)
        else:
            best_clusters = np.argpartition(centroid_dists, nprobe)[:nprobe]

        top5_d = self._top5_dists
        top5_l = self._top5_labels
        top5_d.fill(_INT64_MAX)
        top5_l.fill(0)

        offsets = self.offsets
        for c_idx in best_clusters:
            c = int(c_idx)
            s = int(offsets[c])
            e = int(offsets[c + 1])
            if s < e:
                self._scan_range(s, e, q, q_sq, top5_d, top5_l)

        return int(top5_l.sum())

    def search_adaptive(self, query: np.ndarray, nprobe: int = 2,
                        repair_min: int = 1, repair_max: int = 4,
                        max_repair: int = 4) -> int:
        """Search with adaptive repair for borderline cases."""
        q = self._query_i32
        np.copyto(q, query, casting="unsafe")
        q_sq = q @ q

        qc = self.centroids_i32 @ q
        centroid_dists = q_sq + self.centroids_sq - 2 * qc

        total_probe = min(nprobe + max_repair, self.k)
        if total_probe >= self.k:
            top_sorted = np.argsort(centroid_dists)
        else:
            top_clusters = np.argpartition(centroid_dists, total_probe)[:total_probe]
            top_sorted = top_clusters[np.argsort(centroid_dists[top_clusters])]

        top5_d = self._top5_dists
        top5_l = self._top5_labels
        top5_d.fill(_INT64_MAX)
        top5_l.fill(0)

        offsets = self.offsets

        for c_idx in top_sorted[:nprobe]:
            c = int(c_idx)
            s = int(offsets[c])
            e = int(offsets[c + 1])
            if s < e:
                self._scan_range(s, e, q, q_sq, top5_d, top5_l)

        fraud_count = int(top5_l.sum())
        if fraud_count < repair_min or fraud_count > repair_max:
            return fraud_count

        # Repair phase with bbox pruning
        for c_idx in top_sorted[nprobe:]:
            c = int(c_idx)
            bmin = self.bbox_min_i32[c]
            bmax = self.bbox_max_i32[c]
            below = bmin - q
            above = q - bmax
            d = np.maximum(below, 0) + np.maximum(above, 0)
            if int(np.sum(d * d)) >= top5_d.max():
                break

            s = int(offsets[c])
            e = int(offsets[c + 1])
            if s < e:
                self._scan_range(s, e, q, q_sq, top5_d, top5_l)

        return int(top5_l.sum())

    def _scan_range(self, start, end, query_i32, q_sq, top5_dists, top5_labels):
        """Scan a contiguous range of vectors."""
        dot = self.vectors[start:end] @ query_i32
        dists = self.vector_sq[start:end] + q_sq - 2 * dot.astype(np.int64)

        worst = top5_dists.max()
        mask = dists < worst
        if not np.any(mask):
            return

        cand_dists = dists[mask]
        cand_labels = self.labels[start:end][mask]

        if len(cand_dists) > K_NEIGHBORS:
            idx = np.argpartition(cand_dists, K_NEIGHBORS)[:K_NEIGHBORS]
            cand_dists = cand_dists[idx]
            cand_labels = cand_labels[idx]

        # Merge into pre-allocated buffer (no allocation)
        n_cand = len(cand_dists)
        md = self._merge_dists
        ml = self._merge_labels
        md[:K_NEIGHBORS] = top5_dists
        ml[:K_NEIGHBORS] = top5_labels
        md[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_dists
        ml[K_NEIGHBORS:K_NEIGHBORS + n_cand] = cand_labels
        total = K_NEIGHBORS + n_cand
        idx = np.argpartition(md[:total], K_NEIGHBORS)[:K_NEIGHBORS]
        top5_dists[:] = md[idx]
        top5_labels[:] = ml[idx]

    def close(self):
        self.mm.close()
        os.close(self.fd)
