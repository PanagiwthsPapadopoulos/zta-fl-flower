import numpy as np

def compress_weights(weights: list[np.ndarray], bits: int) -> list[np.ndarray]:
    if bits == 32:
        return weights
    if bits == 16:
        return [w.astype(np.float16) for w in weights]
    if bits == 8:
        compressed = []
        for w in weights:
            if w.size == 0:
                # Cast to uint8 for type consistency and append dummy metadata
                compressed.append(w.astype(np.uint8))
                compressed.append(np.array([0.0, 1.0], dtype=np.float32))
                continue
            
            w_min, w_max = float(w.min()), float(w.max())
            scale = (w_max - w_min) / 255.0 if w_max > w_min else 1.0
            q_w = np.round((w - w_min) / scale).astype(np.uint8)
            compressed.append(q_w)
            compressed.append(np.array([w_min, scale], dtype=np.float32))
        return compressed
    return weights


def decompress_weights(compressed: list[np.ndarray], bits: int) -> list[np.ndarray]:
    if bits == 32:
        return compressed
    if bits == 16:
        return [w.astype(np.float32) for w in compressed]
    if bits == 8:
        # SAFEGUARD: Detect if the payload is actually uncompressed floats.
        # Flower passes raw float32 global parameters during Round 0 evaluation.
        # If there are no uint8 arrays, it's already decompressed.
        is_compressed_payload = any(w.dtype == np.uint8 for w in compressed if w.size > 0)
        
        if not is_compressed_payload and len(compressed) > 0:
            return compressed

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