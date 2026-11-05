"""Regional verification and telemetry, used in both execution modes."""
import copy
import json
import math
from pathlib import Path
import time
from datetime import datetime

from src.shared.network.model_payload import decode_model, encode_model, validate_sample_count
from src.shared.utils.reproducibility import derive_seed
from src.tier_fog.shap_verifier import shap_weighted_aggregate


def _finite_metric(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def persist_round(history, record, metadata, fog_num, round_num, simulated):
    history["metadata"].update({"telemetry_mode": "simulated" if simulated else "deployment",
                                "fog_cloud_transport": "in_process_wire_roundtrip" if simulated else "flower_grpc_via_tcp_ipc"})
    history["rounds"][round_num] = record
    common = ["status", "error", "timestamp"]
    pillars = {
        "trustdb": common + ["attestation_accepted", "attestation_rejected", "attestation_outcomes",
                              "admission_outcomes", "attestation_bypassed", "update_outcomes",
                              "update_rejections", "trust_db", "node_ground_truths"],
        "shap_scores": common + ["raw_shap_scores", "shap_filter", "saboteurs", "rewarded_nodes", "total_regional_trust"],
        "performance_metrics": common + [key for key in record if key.startswith(("latency_", "node_", "token_"))
                                          or key.endswith("_unix")],
    }
    for directory, fields in pillars.items():
        target = Path("results") / metadata.get("experiment_name", "default_run") / directory / f"fog_{fog_num}_state.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": history["metadata"], "rounds": {
            r: {key: data.get(key) for key in fields} for r, data in history["rounds"].items()
        }}
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=4, allow_nan=False))
        temporary.replace(target)


def process_region(*, gatekeeper, reference, val_data, results, nonces, expected_clients,
                   metadata, fog_num, round_num, history, n_explain, num_classes, simulated=False):
    record = {"status": "no_results", "error": None, "round_start_unix": time.time(),
              "timestamp": datetime.now().isoformat(), "attestation_accepted": [],
              "attestation_rejected": [], "attestation_outcomes": {}, "update_rejections": {},
              "admission_outcomes": copy.deepcopy(gatekeeper.admission_outcomes),
              "latency_admission_ms": gatekeeper.admission_latency_ms,
              "attestation_bypassed": gatekeeper.tpm_engine.insecure_mode,
              "update_outcomes": {}, "shap_filter": {},
              "raw_shap_scores": {}, "saboteurs": [], "rewarded_nodes": [], "total_regional_trust": 0.0}
    for key in ("node_training_latencies", "node_payload_sizes", "token_sizes_bytes", "node_ground_truths",
                "node_local_training_start_unix", "node_local_training_end_unix", "node_inference_metrics",
                "node_update_array_bytes", "node_update_tensor_bytes", "node_shap_attack_metrics"):
        record[key] = {}
    for key in ("latency_attestation_ms", "latency_shap_computation_sec", "attestation_validation_start_unix",
                "attestation_validation_end_unix", "shap_calculation_start_unix", "shap_calculation_end_unix",
                "aggregation_start_unix", "aggregation_end_unix"):
        record[key] = None
    try:
        record["attestation_validation_start_unix"] = time.time()
        gatekeeper.filter_node_updates("fog", round_num, results, nonces, expected_clients)
        record["attestation_validation_end_unix"] = time.time()
        record["latency_update_attestation_ms"] = 1000 * (record["attestation_validation_end_unix"] - record["attestation_validation_start_unix"])
        record["latency_attestation_ms"] = record["latency_admission_ms"] + record["latency_update_attestation_ms"]
        outcomes = gatekeeper.last_outcomes
        record["attestation_outcomes"] = copy.deepcopy(outcomes)
        for cid, outcome in outcomes.items():
            if outcome.get("attempted") is not False:
                record["attestation_accepted" if outcome["authenticated"] else "attestation_rejected"].append(outcome["tpm_id"])
            record["node_ground_truths"][outcome["tpm_id"]] = gatekeeper.ground_truth(cid).get("role", "unknown")
        models, sizes, identities = [], [], []
        seen = set()
        bits = int(metadata.get("quantization_bits", 32))
        for proxy, fit_res in results:
            if proxy.cid in seen:
                continue
            seen.add(proxy.cid)
            outcome = outcomes[proxy.cid]
            identity = outcome["tpm_id"]
            token = fit_res.metrics.get("tpm_token_json", "")
            record["token_sizes_bytes"][identity] = len(token.encode()) if isinstance(token, str) else 0
            if not outcome["authenticated"]:
                continue
            for target, source in (("node_training_latencies", "latency_adv_training_sec"),
                                   ("node_payload_sizes", "payload_size_mb"),
                                   ("node_local_training_start_unix", "local_adversarial_training_start_unix"),
                                   ("node_local_training_end_unix", "local_adversarial_training_end_unix")):
                record[target][identity] = _finite_metric(fit_res.metrics.get(source))
            record["node_inference_metrics"][identity] = {
                key: _finite_metric(value) for key, value in fit_res.metrics.items()
                if key.startswith(("edge_fp32_", "edge_int8_", "edge_inference_"))
            }
            # Attacker-reported diagnostics, not cryptographically proven attack
            # efficacy or a substitute for the Fog's own filtering decision.
            record["node_shap_attack_metrics"][identity] = {
                key: (value if isinstance(value, (str, bool)) else _finite_metric(value))
                for key, value in fit_res.metrics.items() if key.startswith("shap_attack_")
            }
            for key in ("array", "tensor"):
                record[f"node_update_{key}_bytes"][identity] = _finite_metric(fit_res.metrics.get(f"update_{key}_bytes"))
            if proxy.cid in gatekeeper.attestation_only_clients:
                if fit_res.num_examples != 0 or fit_res.parameters.tensors:
                    record["update_rejections"][identity] = "Model supplied during attestation-only round"
                    gatekeeper.trust_db.apply_behavioral_penalty(identity, identity, round_num)
                continue
            if not outcome["accepted"]:
                continue  # A quarantined identity did not submit an eligible model update.
            update_outcome = {"role": gatekeeper.ground_truth(proxy.cid).get("role", "unknown"),
                              "accepted": None, "stage": "payload"}
            record["update_outcomes"][identity] = update_outcome
            try:
                validate_sample_count(fit_res.num_examples)
                model = decode_model(reference, fit_res.parameters, bits)
            except (ValueError, TypeError, OverflowError, IndexError, KeyError, EOFError) as exc:
                record["update_rejections"][identity] = str(exc)
                update_outcome["accepted"] = False
                gatekeeper.trust_db.apply_behavioral_penalty(identity, identity, round_num)
                continue
            if outcome["accepted"]:
                models.append(model)
                sizes.append(fit_res.num_examples)
                identities.append(identity)
        if not models:
            # A client crash is a reported execution failure, not evidence that
            # the defense neutralized a poisoning attack.
            if any(outcome.get("reason") == "client_training_crash" for outcome in outcomes.values()):
                record["status"] = "error"
                record["error"] = "No eligible updates; at least one client reported a training crash"
            else:
                record["status"] = "all_rejected" if results else "no_results"
            return None, 0, {"total_regional_trust": 0.0, "aggregation_status": record["status"]}
        if val_data is None:
            raise ValueError("SHAP verification requires validation data")
        record["shap_calculation_start_unix"] = time.time()
        aggregate, trust, passed, rewarded, scores = shap_weighted_aggregate(
            models, reference, *val_data, sizes, num_classes, n_explain=n_explain,
            seed=derive_seed(int(metadata.get("random_seed", 42)), "shap", fog_num, round_num),
            diagnostics=record["shap_filter"],
        )
        record["shap_calculation_end_unix"] = time.time()
        record["latency_shap_computation_sec"] = record["shap_calculation_end_unix"] - record["shap_calculation_start_unix"]
        record["raw_shap_scores"] = dict(zip(identities, scores))
        for identity, accepted, reward in zip(identities, passed, rewarded):
            record["update_outcomes"][identity].update(accepted=accepted, stage="shap")
            if not accepted:
                record["saboteurs"].append((identity, identity))
                record["update_rejections"][identity] = "SHAP or validation rejection"
                gatekeeper.trust_db.apply_behavioral_penalty(identity, identity, round_num)
            elif reward:
                record["rewarded_nodes"].append((identity, identity))
                gatekeeper.trust_db.apply_behavioral_reward(identity, identity, round_num)
        if not math.isfinite(trust) or trust <= 0:
            record["status"] = "no_positive_trust"
            return None, 0, {"total_regional_trust": 0.0, "aggregation_status": record["status"]}
        record["aggregation_start_unix"] = time.time()
        parameters = encode_model(aggregate, bits)
        record["aggregation_end_unix"] = time.time()
        record["total_regional_trust"] = trust
        record["status"] = "aggregated"
        return parameters, sum(n for n, ok in zip(sizes, passed) if ok), {
            "total_regional_trust": trust, "aggregation_status": "aggregated"}
    except Exception as exc:
        record["status"] = "error"
        record["error"] = str(exc)
        gatekeeper.logger.exception("Regional verification failed", extra={"round": round_num})
        return None, 0, {"total_regional_trust": 0.0, "aggregation_status": "error"}
    finally:
        record["trust_db"] = copy.deepcopy(gatekeeper.trust_db._db)
        record["round_end_unix"] = time.time()
        persist_round(history, record, metadata, fog_num, round_num, simulated)
        if record["status"] == "aggregated":
            gatekeeper.logger.info(
                f"{gatekeeper.log_prefix} Saved 3-pillar Fog state. Fog aggregation completed.",
                extra={"round": round_num},
            )
