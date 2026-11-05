"""Separate attestation validity, model-update decisions and quarantine state."""
import json
from pathlib import Path


def _percentage(numerator, denominator):
    return round(100.0 * numerator / denominator, 4) if denominator else None


def calculate_security_rates(run_dir):
    attestation = dict(valid_attempts=0, invalid_attempts=0, false_rejections_count=0,
                       false_acceptances_count=0, unknown_attempts=0, missing_responses=0,
                       bypassed_attempts=0, reported_client_crashes=0)
    updates = dict(honest_updates=0, poisoned_updates=0, honest_rejections=0,
                   poisoned_acceptances=0, unknown_updates=0)
    quarantine = dict(node_rounds=0, quarantined_node_rounds=0, readmissions=0)
    legacy_rounds = 0
    for path in sorted((Path(run_dir) / "trustdb").glob("*.json")):
        data = json.loads(path.read_text())
        previous = {}
        for _, record in sorted(data.get("rounds", {}).items(), key=lambda item: int(item[0])):
            events = list((record.get("admission_outcomes") or {}).values())
            events += list((record.get("attestation_outcomes") or {}).values())
            if "admission_outcomes" not in record:
                legacy_rounds += 1
            for outcome in events:
                if record.get("attestation_bypassed", False):
                    attestation["bypassed_attempts"] += 1
                    continue
                if outcome.get("attempted") is False:
                    attestation["missing_responses"] += 1
                    attestation["reported_client_crashes"] += int(outcome.get("reason") == "client_training_crash")
                    continue
                expected = outcome.get("expected_valid")
                authenticated = outcome.get("authenticated")
                if type(expected) is not bool or type(authenticated) is not bool:
                    attestation["unknown_attempts"] += 1
                elif expected:
                    attestation["valid_attempts"] += 1
                    attestation["false_rejections_count"] += int(not authenticated)
                else:
                    attestation["invalid_attempts"] += 1
                    attestation["false_acceptances_count"] += int(authenticated)
            for outcome in (record.get("update_outcomes") or {}).values():
                role, accepted = outcome.get("role"), outcome.get("accepted")
                if type(accepted) is not bool or role not in ("benign", "label_flip", "gradient_manip", "backdoor", "shap_aware"):
                    updates["unknown_updates"] += 1
                elif role == "benign":
                    updates["honest_updates"] += 1
                    updates["honest_rejections"] += int(not accepted)
                else:
                    updates["poisoned_updates"] += 1
                    updates["poisoned_acceptances"] += int(accepted)
            for identity, state in (record.get("trust_db") or {}).items():
                quarantined = state["is_quarantined"]
                quarantine["node_rounds"] += 1
                quarantine["quarantined_node_rounds"] += int(quarantined)
                quarantine["readmissions"] += int(previous.get(identity, False) and not quarantined)
                previous[identity] = quarantined
    attestation["FRR_percentage"] = _percentage(attestation["false_rejections_count"], attestation["valid_attempts"])
    attestation["FAR_percentage"] = _percentage(attestation["false_acceptances_count"], attestation["invalid_attempts"])
    attestation["legacy_rounds_without_admission_records"] = legacy_rounds
    updates["honest_rejection_percentage"] = _percentage(updates["honest_rejections"], updates["honest_updates"])
    updates["poisoned_acceptance_percentage"] = _percentage(updates["poisoned_acceptances"], updates["poisoned_updates"])
    return {"attestation_gatekeeper": attestation, "update_verification": updates, "quarantine": quarantine}
