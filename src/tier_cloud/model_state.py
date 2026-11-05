"""Exact wire-model state shared by the simulation and deployment strategies."""
import copy
import math

import torch

from src.shared.models.factory import get_seeded_model
from src.shared.network.model_payload import decode_model, encode_model
from src.shared.utils.metrics import federated_averaging
from src.shared.utils.hardware import get_device


class GlobalModelState:
    def initialize_global_state(self):
        self.quantization_bits = int(self.run_metadata.get("quantization_bits", 32))
        self.global_model = get_seeded_model(
            self.model_architecture, self.n_features, self.num_classes,
            int(self.run_metadata.get("random_seed", 42)),
        )
        parameters = self.initial_parameters
        if parameters is None:
            parameters = encode_model(self.global_model, self.quantization_bits)
        self.sync_global_state(parameters)
        # FedAvg must never request its initial model from an arbitrary client.
        self.initial_parameters = parameters
        self.cached_global_state = copy.deepcopy(self.global_model.state_dict())
        self.previous_val_acc = self.validation_accuracy(self.global_model)

    def sync_global_state(self, parameters):
        self.global_model = decode_model(self.global_model, parameters, self.quantization_bits)
        self._broadcast_parameters = parameters

    def validation_accuracy(self, model):
        if self.val_data is None:
            return None
        X, y = self.val_data
        if len(y) == 0:
            raise ValueError("Rollback validation set is empty")
        original_device = next(model.parameters()).device
        original_mode = model.training
        device = get_device()
        correct = 0
        try:
            model.to(device).eval()
            with torch.no_grad():
                for start in range(0, len(y), 512):
                    logits = model(X[start:start + 512].to(device))
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite rollback predictions")
                    correct += (logits.argmax(-1) == y[start:start + 512].to(device)).sum().item()
            return correct / len(y)
        finally:
            model.to(original_device).train(original_mode)

    def aggregate_regions(self, models, weights, server_round):
        """Keep the exact previous broadcast when there is no positive regional trust."""
        positive = [(m, w) for m, w in zip(models, weights) if math.isfinite(w) and w > 0]
        if not positive:
            return self._broadcast_parameters, {"aggregation_status": "no_positive_trust"}
        try:
            candidate = federated_averaging([m for m, _ in positive], [w for _, w in positive])
            parameters = encode_model(candidate, self.quantization_bits)
            # Validate and roll back the model clients actually receive, including quantization.
            candidate = decode_model(self.global_model, parameters, self.quantization_bits)
            accuracy = self.validation_accuracy(candidate)
            # The previous accepted model is the fixed reference, not a configuration choice.
            baseline = self.previous_val_acc
            if (accuracy is not None and baseline is not None and
                    accuracy <= baseline * float(self.run_metadata["rollback_threshold"])):
                self.logger.warning("Global candidate rolled back", extra={"round": server_round})
                return self._broadcast_parameters, {"aggregation_status": "rolled_back"}
            self.global_model = candidate
            self._broadcast_parameters = parameters
            self.cached_global_state = copy.deepcopy(candidate.state_dict())
            self.previous_val_acc = accuracy
            return parameters, {"aggregation_status": "aggregated"}
        except Exception:
            self.logger.exception("Global aggregation failed; retaining broadcast", extra={"round": server_round})
            return self._broadcast_parameters, {"aggregation_status": "error"}
