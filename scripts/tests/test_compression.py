import os
import sys
import numpy as np

# ---------------------------------------------------------
# DOCKER-PROOF PATH INJECTION
# ---------------------------------------------------------
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.network.compression import compress_weights, decompress_weights
from src.utils.config_loader import load_yaml_configs

def test_quantization_fidelity():
    """
    Validates the custom dynamic quantization algorithm.
    Automatically scales the acceptable information loss (MSE) threshold 
    based on the quantization_bits defined in the project configuration.
    """
    print("Executing Dynamic Quantization Fidelity Test...")
    
    # Load configuration
    run_metadata = load_yaml_configs()
    dynamic_bits = int(run_metadata.get("quantization_bits", 8))
    print(f"[CONFIG LOADED] Quantization Bits: {dynamic_bits}")

    # Generate mock weights representing a CNN-LSTM layer
    original_layer = np.random.randn(128, 128).astype(np.float32)
    original_weights = [original_layer]
    
    # Execute dynamic linear uniform quantization
    compressed = compress_weights(original_weights, bits=dynamic_bits)
    reconstructed = decompress_weights(compressed, bits=dynamic_bits)
    
    # Calculate Mean Squared Error (Information Loss)
    mse = np.mean((original_layer - reconstructed[0])**2)
    max_error = np.max(np.abs(original_layer - reconstructed[0]))
    
    print(f"\n{dynamic_bits}-Bit Quantization MSE: {mse:.6f}")
    print(f"Maximum Single-Parameter Error: {max_error:.6f}")
    
    # Dynamically scale the acceptable error threshold based on the bit depth
    if dynamic_bits >= 16:
        acceptable_mse = 0.0001
    elif dynamic_bits >= 8:
        acceptable_mse = 0.005
    elif dynamic_bits >= 4:
        acceptable_mse = 0.05
    else:
        acceptable_mse = 0.5 # 2-bit quantization is effectively binary routing, expect massive loss
        
    assert mse < acceptable_mse, f"Quantization loss too high for {dynamic_bits}-bit. MSE: {mse:.6f} > {acceptable_mse}"
    assert original_layer.shape == reconstructed[0].shape, "Quantization corrupted tensor dimensions."
    
    print(f"\n✅ QUANTIZATION MATH TEST PASSED (Tolerance: < {acceptable_mse})")

if __name__ == "__main__":
    test_quantization_fidelity()