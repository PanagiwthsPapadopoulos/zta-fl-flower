import os
import sys
import logging
import json
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------
# DOCKER-PROOF PATH INJECTION
# ---------------------------------------------------------
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.utils.config_loader import load_yaml_configs
from src.utils.data_loader import DATASET_METADATA

# IMPORTANT: Adjust the import path below if edge_trainer.py lives in a different directory (e.g., src.client.edge_trainer)
from src.federation.edge_trainer import EdgeTrainer

# =====================================================================
# Logger Configuration
# =====================================================================
logger = logging.getLogger("Edge_Trainer_Test")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

def print_separator(title: str):
    print("\n" + "="*80)
    print(f"🚀 TEST: {title}")
    print("="*80)

# =====================================================================
# Deterministic Testing Utilities
# =====================================================================
class DummyModel(nn.Module):
    """A minimal model to test weight extraction, compression, and routing."""
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.fc(x)

def run_edge_trainer_diagnostics():
    """
    Validates the EdgeTrainer's ability to extract live YAML configurations, 
    compress parameters, route adversarial attacks, and integrate TPM attestations.
    """
    print("\n" + "#"*80)
    print("### STARTING EDGE TRAINER INTEGRATION TEST ###")
    print("#"*80)

    # -----------------------------------------------------------------
    # Configuration Loading (Strict Adherence to Live YAML)
    # -----------------------------------------------------------------
    run_metadata = load_yaml_configs()
    
    dataset_name = run_metadata.get("dataset", "edge_iiotset").lower()
    if dataset_name not in DATASET_METADATA:
        logger.error(f"❌ FATAL: Dataset '{dataset_name}' is missing from DATASET_METADATA.")
        sys.exit(1)
        
    # Extract dimensions
    N_CLASSES = DATASET_METADATA[dataset_name]["classes"]
    INPUT_DIM = DATASET_METADATA[dataset_name]["features"]
    
    # --- SPECIFIC ERROR REPORTING LOGIC ---
    # Define exactly which keys MUST exist in the loaded YAMLs for this test to run
    required_keys = [
        "batch_size", 
        "quantization_bits", 
        "learning_rate", 
        "local_epochs", 
        "clip_norm", 
        "benign_adv_ratio"
    ]
    
    # Identify exactly which keys are missing or mapped to None
    missing_keys = [key for key in required_keys if run_metadata.get(key) is None]
    
    if missing_keys:
        logger.error(f"❌ FATAL: The following required configuration variables are missing from your YAML files: {missing_keys}")
        sys.exit(1)
    # --------------------------------------
        
    # Extract operational parameters from YAML safely now that they are validated
    BATCH_SIZE = run_metadata.get("batch_size")
    QUANTIZATION_BITS = run_metadata.get("quantization_bits")
    LEARNING_RATE = run_metadata.get("learning_rate")
    LOCAL_EPOCHS = run_metadata.get("local_epochs")
    CLIP_NORM = run_metadata.get("clip_norm")
    
    # Extract adversarial boundaries from YAML
    BENIGN_ADV_RATIO = run_metadata.get("benign_adv_ratio")
    BENIGN_EPS = run_metadata.get("benign_eps", 0.05)
    BENIGN_ALPHA = run_metadata.get("benign_alpha", 0.2)
    CLIP_MIN = run_metadata.get("clip_min", 0.0)
    CLIP_MAX = run_metadata.get("clip_max", 1.0)

    logger.debug(f"Loaded YAML -> LR: {LEARNING_RATE}, Epochs: {LOCAL_EPOCHS}, Quant: {QUANTIZATION_BITS}-bit, Adv Ratio: {BENIGN_ADV_RATIO}")

    # Create dummy data and loader to simulate the local edge dataset
    X_dummy = torch.randn((100, INPUT_DIM))
    y_dummy = torch.randint(0, N_CLASSES, (100,))
    dummy_loader = DataLoader(TensorDataset(X_dummy, y_dummy), batch_size=BATCH_SIZE)
    model = DummyModel(INPUT_DIM, N_CLASSES)

    # Simulate the exact configuration dictionary the system passes to the trainer
    base_train_config = {
        "quantization_bits": QUANTIZATION_BITS,
        "learning_rate": LEARNING_RATE,
        "local_epochs": LOCAL_EPOCHS,
        "clip_norm": CLIP_NORM,
        "num_classes": N_CLASSES,
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX
    }

    # =====================================================================
    # TEST 1: Parameter Compression & Decompression
    # =====================================================================
    print_separator(f"Compression Integrity ({QUANTIZATION_BITS}-bit Quantization)")
    
    trainer = EdgeTrainer(
        logger=logger, log_prefix="[TEST_EDGE]", model=model, 
        train_loader=dummy_loader, device="cpu", 
        train_config=base_train_config, dataset_metadata={"test": "data"}
    )
    
    # Extract, compress, and immediately decompress back into the model
    try:
        compressed_params = trainer.get_parameters()
        trainer.set_parameters(compressed_params)
        logger.info(f"✅ Parameter compression and decompression succeeded natively at {QUANTIZATION_BITS}-bits.")
    except Exception as e:
        logger.error(f"❌ Compression failure: {e}")
        sys.exit(1)

    # =====================================================================
    # TEST 2: Honest Training & TPM Attestation Injection
    # =====================================================================
    print_separator("Honest Training Route & ZTA Protocol Verification")
    
    honest_config = base_train_config.copy()
    honest_config["role"] = "benign"
    honest_config["adv_ratio"] = 0.0 # Force pure honest mode for this test
    
    trainer.train_config = honest_config
    
    # We patch the mathematical training function and the TPM engine so the test runs instantly.
    # We are testing the IF/ELSE routing logic and metadata attachment, not the math itself.
    with patch('src.federation.edge_trainer.local_train_honest', return_value=0.1234) as mock_honest, \
         patch('src.security.attestation.tpm_core.TPMEngine') as mock_tpm:
        
        # Mock the hardware token output
        mock_tpm_instance = mock_tpm.return_value
        mock_tpm_instance.generate_attestation_token.return_value = {"signature": "valid_test_signature", "pcr": "0000"}
        
        # Execute the main training loop with ZTA strategy
        new_params, dataset_size, metadata = trainer.execute_training(
            parameters=compressed_params, current_round=1, strategy="zta", config={"nonce": "test_nonce"}
        )
        
        # Assertions
        mock_honest.assert_called_with(model=model, loader=dummy_loader, device="cpu", lr=LEARNING_RATE, epochs=1, clip_norm=CLIP_NORM)
        
        if "tpm_token_json" in metadata:
            logger.info("✅ ZTA Strategy successfully triggered TPM Engine and injected hardware token into metadata.")
        else:
            logger.error("❌ ZTA Strategy failed to attach TPM token to metadata.")
            sys.exit(1)
        mock_tpm_instance.generate_attestation_token.assert_called_once_with(
            nonce="test_nonce",
            software_label="[TEST_EDGE]",
        )
            
        logger.info("✅ Honest routing verified perfectly.")

    # =====================================================================
    # TEST 2B: Real Insecure TPM Attestation Smoke Test
    # =====================================================================
    print_separator("Real TPMEngine Insecure-Mode Attestation Handoff")

    trainer.train_config = honest_config

    with patch.dict(os.environ, {"ZTA_INSECURE_MODE": "true"}), \
         patch('src.federation.edge_trainer.local_train_honest', return_value=0.2468):

        _, _, metadata = trainer.execute_training(
            parameters=compressed_params,
            current_round=2,
            strategy="zta",
            config={"nonce": "strict_nonce"},
        )

        try:
            tpm_token = json.loads(metadata["tpm_token_json"])
        except Exception as e:
            logger.error(f"❌ Real TPMEngine smoke test failed to parse attestation metadata: {e}")
            sys.exit(1)

        if tpm_token.get("status") != "insecure_bypass" or tpm_token.get("IDi") != "[TEST_EDGE]":
            logger.error(f"❌ Real TPMEngine smoke test produced unexpected token: {tpm_token}")
            sys.exit(1)

        logger.info("✅ Real TPMEngine insecure-mode handoff verified with nonce and software_label.")

    # =====================================================================
    # TEST 3: Static Adversarial Split Verification (Robustness)
    # =====================================================================
    print_separator("Adversarial Evasion Split (Robustness Augmentation)")
    
    adv_config = base_train_config.copy()
    adv_config["role"] = "benign"
    adv_config["adv_ratio"] = BENIGN_ADV_RATIO
    adv_config["eps"] = BENIGN_EPS
    adv_config["alpha"] = BENIGN_ALPHA
    
    trainer.train_config = adv_config
    
    def dynamic_fgsm_mock(model, x, y, **kwargs):
        return torch.randn_like(x)
    
    with patch('src.security.attacks.adversarial.fgsm_attack', side_effect=dynamic_fgsm_mock) as mock_fgsm, \
         patch('src.federation.edge_trainer.local_train_honest', return_value=0.4321), \
         patch('src.security.attestation.tpm_core.TPMEngine') as mock_tpm:
        
        # Mock the JSON-serializable token output
        mock_tpm.return_value.generate_attestation_token.return_value = {"signature": "test_sig", "pcr": "0000"}
        
        trainer.execute_training(parameters=compressed_params, current_round=2, strategy="zta", config={})
        
        if mock_fgsm.called:
            _, kwargs = mock_fgsm.call_args
            if kwargs["alpha"] == BENIGN_EPS and kwargs["clip_min"] == CLIP_MIN and kwargs["clip_max"] == CLIP_MAX:
                logger.info(f"✅ Adversarial split active. Benign data successfully routed through FGSM with eps={BENIGN_EPS}.")
            else:
                logger.error(f"❌ Adversarial split triggered, but YAML variables were not passed correctly. Got: {kwargs}")
                sys.exit(1)
        else:
            if BENIGN_ADV_RATIO > 0.0:
                logger.error("❌ Adversarial split failed to trigger despite adv_ratio > 0.")
                sys.exit(1)
            else:
                logger.info("✅ Adv_ratio is 0.0. Adversarial split correctly skipped.")
    # =====================================================================
    # TEST 4: Byzantine Threat Routing
    # =====================================================================
    print_separator("Byzantine Threat Routing (Label Flip Example)")
    
    byz_config = base_train_config.copy()
    byz_config["role"] = "label_flip"
    byz_config["alpha"] = run_metadata.get("gradient_alpha", 5.0) # Used as the scale parameter
    
    trainer.train_config = byz_config
    
    with patch('src.federation.edge_trainer.local_train_byzantine', return_value=0.9999) as mock_byz, \
         patch('src.security.attestation.tpm_core.TPMEngine') as mock_tpm:
        
        # Mock the JSON-serializable token output
        mock_tpm.return_value.generate_attestation_token.return_value = {"signature": "test_sig", "pcr": "0000"}
        
        trainer.execute_training(parameters=compressed_params, current_round=3, strategy="zta", config={})
        
        if mock_byz.called:
            _, kwargs = mock_byz.call_args
            if kwargs["attack"] == "label_flip" and kwargs["n_classes"] == N_CLASSES:
                logger.info(f"✅ Threat configuration active. Training loop securely routed to Byzantine logic for 'label_flip'.")
            else:
                logger.error("❌ Byzantine routing failed to pass correct configuration.")
                sys.exit(1)
        else:
            logger.error("❌ Byzantine routing completely failed.")
            sys.exit(1)

    print("\n" + "#"*80)
    print("🎉 EDGE TRAINER TESTS COMPLETE. ALL ACTIVE CONFIGURATIONS & ROUTES VALIDATED. 🎉")
    print("#"*80 + "\n")

if __name__ == "__main__":
    run_edge_trainer_diagnostics()
