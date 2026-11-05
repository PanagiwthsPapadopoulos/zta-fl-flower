from src.shared.utils.config_loader import optional_float
from src.shared.utils.reproducibility import seed_training, derive_seed
import os
import fcntl
import gc
import json
import torch
import copy
import time
from torch.utils.data import DataLoader

from src.shared.security.adversarial_math import (
    local_train_honest,
    label_flip_loader,
    local_train_byzantine,
    adversarial_train_epoch,
    local_train_proximal,
)
from src.shared.security.shap_aware import train_shap_constrained
from src.shared.network.compression import compress_weights
from src.shared.network.model_payload import load_model_arrays
from src.tier_edge.tpm_attestation import TPMAttestation
from src.tier_edge.inference import evaluate_cpu_inference


class EdgeTrainer:
    """Encapsulates the PyTorch machine learning loop for standard edge devices.
    Handles decompression, local epochs, adversarial data splits, and quantization.
    """

    def __init__(self, logger, log_prefix, model, train_loader, device, train_config, dataset_metadata, eval_loader=None, shap_data=None):
        """Initializes the localized Edge Trainer responsible for executing isolated model optimizations."""
        self.logger = logger
        self.log_prefix = log_prefix
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.shap_data = shap_data
        self.device = device
        self.train_config = train_config
        self.dataset_metadata = dataset_metadata
        self.tcti = self.train_config.get("tpm_tcti") or os.getenv("TPM2TOOLS_TCTI", "")
        self.tpm_engine = TPMAttestation(logger=self.logger, tcti=self.tcti)

    def get_parameters(self) -> list:
        """Extracts and compresses weights for network transmission."""
        if self.model is None:
            raise RuntimeError(f"{self.log_prefix} Model is uninitialized.")
        weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        bits = int(self.train_config["quantization_bits"])
        return compress_weights(weights, bits)

    def set_parameters(self, parameters: list):
        """Decompresses inbound payloads and loads them into PyTorch."""
        if self.model is None or not parameters:
            raise ValueError("Training requires a complete server model")
        load_model_arrays(self.model, parameters, int(self.train_config["quantization_bits"]))

    def evaluate(self, parameters=None):
        if parameters is not None:
            self.set_parameters(parameters)
        if self.eval_loader is None:
            raise ValueError("Edge evaluation requires held-out validation data")
        return evaluate_cpu_inference(self.model, self.eval_loader,
                                      int8=bool(self.train_config.get("edge_inference_int8", True)))

    def attestation_metrics(self, current_round: int, config: dict) -> dict:
        nonce = config.get("nonce")
        if not isinstance(nonce, str) or not nonce:
            raise ValueError("Attestation requires a server challenge")
        self.tpm_engine.current_round = current_round
        token = self.tpm_engine.generate_attestation_token(nonce, self.log_prefix, current_round)
        return {**self.dataset_metadata, "node_name": self.log_prefix, "log_prefix": self.log_prefix,
                "role": self.train_config["role"], "nonce": nonce, "tpm_tcti": self.tcti,
                "tpm_token_json": json.dumps(token)}

    def execute_training(self, parameters: list, current_round: int, config: dict):
        """Executes the core training loop, managing epochs and adversarial logic with a multi-slot GPU lock."""
        if config.get("attestation_only", False):
            return [], 0, self.attestation_metrics(current_round, config)
        self.set_parameters(parameters)

        def _run_core_training():
            seed_training(int(self.train_config["random_seed"]), int(self.train_config["client_index"]), current_round)
            global_model = copy.deepcopy(self.model)
            global_model.eval()
            global_model.to(self.device)
            self.model.to(self.device)
            lr = self.train_config["learning_rate"]
            role = self.train_config["role"]
            epochs = self.train_config["local_epochs"]
            active_loader = self.train_loader
            if role == "label_flip":
                # Freeze one poisoned label set across the local epochs of a round.
                active_loader = label_flip_loader(active_loader, self.train_config["num_classes"],
                                                  active_loader.batch_size, self.train_config["p_flip"])
            self.logger.info(f"{self.log_prefix} Starting Training with learning rate: {lr}, epochs: {epochs}, role: {role}", extra={"round": current_round})
            loss = 0.0
            attack_metrics = {}
            optimization_start = time.time()
            if role == "shap_aware":
                loss, final_params, attack_metrics = train_shap_constrained(
                    self.model, global_model, active_loader, self.shap_data,
                    attack=self.train_config["shap_aware_base_attack"],
                    n_classes=int(self.train_config["num_classes"]),
                    constrained=self.train_config.get("shap_aware_constrained", True),
                    tau=optional_float(self.train_config.get("shap_aware_tau")),
                    epochs=int(epochs), lr=float(lr), clip_norm=float(self.train_config["clip_norm"]),
                    n_explain=int(self.train_config["shap_explain_count"]),
                    seed=derive_seed(int(self.train_config["random_seed"]), "shap",
                                     int(self.train_config["fog_num"]), current_round),
                    quantization_bits=int(self.train_config["quantization_bits"]),
                    p_flip=float(self.train_config["p_flip"]),
                    max_backtracks=int(self.train_config.get("shap_aware_max_backtracks", 10)),
                    reference_arrays=parameters,
                )
                self.logger.info(f"{self.log_prefix} SHAP attack: {attack_metrics}", extra={"round": current_round})
            else:
                # Preserve Adam momentum across ordinary local epochs.
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                for epoch in range(epochs):
                    loss = self._train_based_on_role(
                        role, lr, current_round, active_loader, global_model, optimizer,
                        apply_post_attack=(epoch == epochs - 1),
                    )
                    self.logger.info(f"{self.log_prefix} Epoch {epoch + 1}/{epochs} complete. Loss: {loss:.4f}", extra={"round": current_round})
                final_params = self.get_parameters()
            optimization_end = time.time()
            inference_metrics = {}
            if self.train_config.get("edge_inference_int8", False):
                _, _, inference_metrics = self.evaluate()
            # Quote after inference so its runtime does not consume freshness.
            metadata = self.attestation_metrics(current_round, config)
            metadata.update(inference_metrics)
            metadata.update(attack_metrics)
            metadata["loss"] = loss
            metadata["local_adversarial_training_start_unix"] = optimization_start
            metadata["local_adversarial_training_end_unix"] = optimization_end
            metadata["latency_adv_training_sec"] = optimization_end - optimization_start
            metadata["update_array_bytes"] = sum(array.nbytes for array in final_params)
            from flwr.common import ndarrays_to_parameters
            metadata["update_tensor_bytes"] = sum(len(value) for value in ndarrays_to_parameters(final_params).tensors)
            if self.device != "cpu":
                self.logger.info(f"{self.log_prefix} Local epochs complete. Purging GPU VRAM.", extra={"round": current_round})
                global_model.to("cpu")
                del global_model
                self.model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()
            return final_params, len(self.train_loader.dataset), metadata

        if self.device != "cpu" and self.train_config.get("use_gpu_lock", False):
            max_slots = int(self.train_config.get("max_concurrent_gpus", 1))
            lock_dir = "/app/runtime/locks"
            os.makedirs(lock_dir, exist_ok=True)
            acquired_slot = None
            lock_file_handle = None
            self.logger.info(f"{self.log_prefix} Queued for GPU. Waiting for 1 of {max_slots} slots...", extra={"round": current_round})
            while acquired_slot is None:
                for slot in range(max_slots):
                    lock_path = os.path.join(lock_dir, f"gpu_slot_{slot}.lock")
                    try:
                        f = open(lock_path, "w")
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired_slot = slot
                        lock_file_handle = f
                        break
                    except BlockingIOError:
                        f.close()
                        continue
                if acquired_slot is None:
                    time.sleep(2)
            self.logger.info(f"{self.log_prefix} Acquired GPU Slot {acquired_slot}! Initializing CUDA context.", extra={"round": current_round})
            try:
                return _run_core_training()
            finally:
                fcntl.flock(lock_file_handle.fileno(), fcntl.LOCK_UN)
                lock_file_handle.close()
                self.logger.info(f"{self.log_prefix} Released GPU Slot {acquired_slot}.", extra={"round": current_round})
        else:
            self.logger.info(f"{self.log_prefix} Bypassing lock for CPU execution.", extra={"round": current_round})
            return _run_core_training()

    def _adversarial_options(self):
        """One configured attack and training budget, independent of label poisoning."""
        attack = str(self.train_config["robustness_eval_attack"]).lower()
        if attack not in ("fgsm", "pgd"):
            raise ValueError("robustness_eval_attack must be fgsm or pgd")
        return dict(
            adv_ratio=float(self.train_config["adv_ratio"]),
            eps=float(self.train_config["eps"]),
            alpha=float(self.train_config["alpha"]),
            n_iter=int(self.train_config["n_iter"]),
            use_pgd=attack == "pgd",
            clip_min=optional_float(self.train_config["clip_min"]),
            clip_max=optional_float(self.train_config["clip_max"]),
        )

    def _train_based_on_role(
        self,
        role: str,
        lr: float,
        current_round: int,
        active_loader: DataLoader,
        global_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        apply_post_attack: bool = True,
    ):
        """Executes the specific PyTorch optimization loop variant strictly dictated by the node's assigned profile role."""
        clip_norm = float(self.train_config["clip_norm"])
        if role in ["label_flip", "gradient_manip"]:
            alpha_scale = float(self.train_config.get("gradient_alpha", 10.0))
            num_classes = self.train_config["num_classes"]
            p_flip = float(self.train_config.get("p_flip", 0.1))
            self.logger.info(f"{self.log_prefix} Performing {role} Attack! Number of classes: {num_classes}, Alpha scale: {alpha_scale}, p_flip: {p_flip}, Learning Rate: {lr}, Clip norm: {clip_norm}", extra={"round": current_round})
            return local_train_byzantine(
                model=self.model,
                loader=active_loader,
                attack=role,
                n_classes=num_classes,
                scale=alpha_scale,
                p_flip=p_flip,
                device=self.device,
                lr=lr,
                clip_norm=clip_norm,
                optimizer=optimizer,
                global_model=global_model,
                apply_post_attack=apply_post_attack,
                labels_prepared=role == "label_flip",
                **self._adversarial_options(),
            )

        elif role in ["backdoor"]:
            self.logger.info(f"{self.log_prefix} Backdoor attack detected! Dataset already poisoned, performing honest training!", extra={"round": current_round})
            return local_train_honest(model=self.model, loader=active_loader, device=self.device, lr=lr, clip_norm=clip_norm, optimizer=optimizer)

        elif role in ["benign"]:
            options = self._adversarial_options()
            if options["adv_ratio"] > 0:
                self.logger.info(
                    f"{self.log_prefix} Performing Adversarial Training on Benign Node! "
                    f"Adversary Ratio: {options['adv_ratio']}, Epsilon: {options['eps']}, "
                    f"Alpha: {options['alpha']}, Number of Iter: {options['n_iter']}, "
                    f"Use pgd (if false, use fgsm): {options['use_pgd']}, "
                    f"clip_min: {options['clip_min']}, clip_max: {options['clip_max']}, Clip norm: {clip_norm}",
                    extra={"round": current_round},
                )
                return adversarial_train_epoch(
                    model=self.model,
                    loader=active_loader,
                    optimizer=optimizer,
                    device=self.device,
                    clip_norm=clip_norm,
                    **options,
                )
            else:
                self.logger.info(f"{self.log_prefix} Could not perform adversarial training! Adversary ratio is set to 0! Executing honest training with learning rate: {lr} and clip_norm: {clip_norm}.", extra={"round": current_round})
                return local_train_honest(model=self.model, loader=active_loader, device=self.device, lr=lr, clip_norm=clip_norm, optimizer=optimizer)
        else:
            self.logger.info(f"{self.log_prefix} No valid role detected! Detected role: {role}. Executing honest training with learning rate: {lr} and clip_norm: {clip_norm}.", extra={"round": current_round})
            return local_train_honest(model=self.model, loader=active_loader, device=self.device, lr=lr, clip_norm=clip_norm, optimizer=optimizer)
