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

        # Offsets: (k+1) * uint32 → Python list (avoid numpy→int conversion in loop)
        offsets_np = np.frombuffer(
            self.mm, dtype=np.uint32, count=k + 1, offset=offset
        ).copy()
        self.offsets = offsets_np
        self.offsets_list = offsets_np.tolist()
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

        # Pre-allocated repair buffers (avoid allocs in hot path)
        self._bbox_below = np.empty((k, DIMS), dtype=np.int32)
        self._bbox_above = np.empty((k, DIMS), dtype=np.int32)
        self._bbox_d = np.empty((k, DIMS), dtype=np.int32)
        self._bbox_dists = np.empty(k, dtype=np.int64)

    def search(self, query: np.ndarray, nprobe: int = 7) -> int:
        """Find 5 nearest neighbors. Inlined scan, minimal allocations."""
        q = self._query_i32
        np.copyto(q, query, casting="unsafe")
        q_sq = int(q @ q)

        qc = self.centroids_i32 @ q
        centroid_dists = self.centroids_sq + q_sq - 2 * qc
        best_clusters = np.argpartition(centroid_dists, nprobe)[:nprobe]

        top5_d = self._top5_dists
        top5_l = self._top5_labels
        top5_d.fill(_INT64_MAX)
        top5_l.fill(0)
        md = self._merge_dists
        ml = self._merge_labels

        offsets_list = self.offsets_list
        vectors = self.vectors
        vector_sq = self.vector_sq
        labels = self.labels
        worst_val = _INT64_MAX

        for c_idx in best_clusters:
            c = int(c_idx)
            s = offsets_list[c]
            e = offsets_list[c + 1]
            if s >= e:
                continue

            dot = vectors[s:e] @ q
            dists = vector_sq[s:e] + q_sq
            np.subtract(dists, np.multiply(dot, 2, dtype=np.int64), out=dists)

            mask = dists < worst_val
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
            total = K_NEIGHBORS + n_cand
            idx = md[:total].argsort()[:K_NEIGHBORS]
            top5_d[:] = md[idx]
            top5_l[:] = ml[idx]
            worst_val = int(top5_d.max())

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

        best_clusters = np.argpartition(centroid_dists, nprobe)[:nprobe]

        top5_d = self._top5_dists
        top5_l = self._top5_labels
        top5_d.fill(_INT64_MAX)
        top5_l.fill(0)
        md = self._merge_dists
        ml = self._merge_labels

        offsets_list = self.offsets_list
        vectors = self.vectors
        vector_sq = self.vector_sq
        labels = self.labels
        worst_val = _INT64_MAX

        # Initial probe
        for c_idx in best_clusters:
            c = int(c_idx)
            s = offsets_list[c]
            e = offsets_list[c + 1]
            if s >= e:
                continue

            dot = vectors[s:e] @ q
            dists = vector_sq[s:e] + q_sq
            np.subtract(dists, np.multiply(dot, 2, dtype=np.int64), out=dists)

            mask = dists < worst_val
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
            worst_val = int(top5_d.max())

        fraud_count = int(top5_l.sum())
        if fraud_count < repair_min or fraud_count > repair_max:
            return fraud_count

        # Phase 2 (repair): bbox filter ALL clusters using pre-allocated buffers
        bb = self._bbox_below
        ba = self._bbox_above
        bd = self._bbox_d
        bdists = self._bbox_dists

        np.subtract(self.bbox_min_i32, q, out=bb)
        np.subtract(q, self.bbox_max_i32, out=ba)
        np.maximum(bb, 0, out=bb)
        np.maximum(ba, 0, out=ba)
        np.add(bb, ba, out=bd)
        np.multiply(bd, bd, out=bd)
        np.sum(bd, axis=1, out=bdists)

        bdists[best_clusters] = _INT64_MAX
        candidates = np.where(bdists < worst_val)[0]

        for c_idx in candidates:
            c = int(c_idx)
            s = offsets_list[c]
            e = offsets_list[c + 1]
            if s >= e:
                continue

            dot = vectors[s:e] @ q
            dists = vector_sq[s:e] + q_sq
            np.subtract(dists, np.multiply(dot, 2, dtype=np.int64), out=dists)

            mask = dists < worst_val
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
            worst_val = int(top5_d.max())

        return int(top5_l.sum())

    def close(self):
        self.mm.close()
        os.close(self.fd)
