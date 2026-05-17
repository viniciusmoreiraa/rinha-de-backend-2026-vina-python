/*
 * scan.c — C extension for IVF vector scan + top-5 merge.
 * Compile: gcc -O3 -march=native -shared -fPIC -o scan.so scan.c
 * Docker:  gcc -O3 -march=haswell -shared -fPIC -o scan.so scan.c
 *
 * Matches NumPy behavior exactly:
 *   dot = vectors[start:end] @ query_i32  (int16 @ int32 → int64)
 *   dists = vector_sq[start:end] + q_sq - 2 * dot
 */

#include <stdint.h>

#define DIMS 14
#define K 5

/*
 * scan_and_merge: scan vectors in [start, end), compute distances,
 * and merge best candidates into top5.
 *
 * Parameters:
 *   vecs:        pointer to vectors array (int16, N*14 flat)
 *   vsq:         pointer to vector_sq array (int64, N flat)
 *   labels:      pointer to labels array (uint8, N flat)
 *   query:       int32[14] query vector
 *   q_sq:        int64 query squared norm
 *   start, end:  range [start, end) to scan
 *   top5_dists:  int64[5] current top-5 distances (modified in place)
 *   top5_labels: uint8[5] current top-5 labels (modified in place)
 */
void scan_and_merge(
    const int16_t *vecs,
    const int64_t *vsq,
    const uint8_t *labels,
    const int32_t *query,
    int64_t q_sq,
    int32_t start,
    int32_t end,
    int64_t *top5_dists,
    uint8_t *top5_labels
) {
    /* Find current worst in top-5 */
    int worst_idx = 0;
    int64_t worst_val = top5_dists[0];
    for (int i = 1; i < K; i++) {
        if (top5_dists[i] > worst_val) {
            worst_val = top5_dists[i];
            worst_idx = i;
        }
    }

    for (int32_t i = start; i < end; i++) {
        /* Compute dot product in int64 (matching NumPy: int16 @ int32 → int64) */
        const int16_t *v = &vecs[(int64_t)i * DIMS];
        int64_t dot = 0;
        for (int d = 0; d < DIMS; d++) {
            dot += (int64_t)v[d] * (int64_t)query[d];
        }

        /* dist = vsq[i] + q_sq - 2 * dot (all int64, no overflow) */
        int64_t dist = vsq[i] + q_sq - 2 * dot;

        if (dist < worst_val) {
            top5_dists[worst_idx] = dist;
            top5_labels[worst_idx] = labels[i];

            /* Recompute worst */
            worst_val = dist;
            worst_idx = 0;
            for (int j = 0; j < K; j++) {
                if (top5_dists[j] > worst_val) {
                    worst_val = top5_dists[j];
                    worst_idx = j;
                }
            }
        }
    }
}
