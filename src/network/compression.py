import numpy as np

def compress_weights(weights: list[np.ndarray], bits: int) -> list[np.ndarray]:
    """
    Transforms the internal precision definitions corresponding to gradient aggregates shrinking overhead cost.
    Generates accompanying scaling metadata blocks automatically whenever strict octet constraints apply.
    """
    if bits == 32:
        return weights
    if bits == 16:
        return [w.astype(np.float16) for w in weights]
    if bits == 8:
        compressed = []
        for w in weights:
            if w.size == 0:
                compressed.append(w)
                continue
            w_min, w_max = float(w.min()), float(w.max())
            scale = (w_max - w_min) / 255.0 if w_max > w_min else 1.0
            q_w = np.round((w - w_min) / scale).astype(np.uint8)
            compressed.append(q_w)
            # Imbeds the translation metrics permitting downstream expansion.
            compressed.append(np.array([w_min, scale], dtype=np.float32))
        return compressed
    return weights


def decompress_weights(compressed: list[np.ndarray], bits: int) -> list[np.ndarray]:
    """
    Interpolates payload metrics against attached array scalars driving parameter recovery sequences.
    """
    if bits == 32:
        return compressed
    if bits == 16:
        return [w.astype(np.float32) for w in compressed]
    if bits == 8:
        # Analyzes the payload matrix identifying internal metadata offset structural limits.
        if len(compressed) % 2 != 0:
            raise ValueError(f"Corrupted INT8 payload: Expected an even number of arrays, got {len(compressed)}.")
        
        weights = []
        for i in range(0, len(compressed), 2):
            q_w = compressed[i]
            if q_w.size == 0:
                weights.append(q_w.astype(np.float32))
                continue
            meta = compressed[i+1]
            w = (q_w.astype(np.float32) * meta[1]) + meta[0]
            weights.append(w)
        return weights
    return compressed