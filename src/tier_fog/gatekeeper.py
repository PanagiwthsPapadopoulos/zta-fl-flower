import json
import logging
import os
import re
from typing import Tuple

from src.tier_fog.tpm_verifier import TPMVerifier
from src.tier_fog.trust_db import TrustDatabase


class ZeroTrustGatekeeper:
    """Authenticate identities before binding them to Flower clients for this run."""

    def __init__(self, logger: logging.Logger, log_prefix: str, run_metadata: dict, fog_num=None):
        self.logger = logger
        self.log_prefix = log_prefix
        self.run_metadata = run_metadata
        self.fog_num = fog_num
        self.tpm_state_root = "/app/runtime/tpm_state"
        insecure = run_metadata.get("insecure", False) or os.getenv("ZTA_INSECURE_MODE", "false").lower() == "true"
        self.tpm_engine = TPMVerifier(logger=logger, insecure_mode=insecure)
        self.trust_db = TrustDatabase(logger=logger)
        self.client_identities = {}  # cid -> (authenticated TPM identity, canonical label)
        self.identity_clients = {}  # one Flower client per TPM per run
        self.attestation_only_clients = set()
        self.admission_outcomes = {}
        self.admission_latency_ms = 0.0
        self.last_outcomes = {}
        self._processed = set()
        self._round = None

    def get_live_ledger(self, server_round: int) -> dict:
        try:
            with open(os.path.join(self.tpm_state_root, "pcr_ledger.json")) as handle:
                ledger = json.load(handle)
            if not isinstance(ledger, dict):
                raise ValueError("PCR ledger must be an object")
            return ledger
        except (OSError, ValueError) as exc:
            self.logger.error(f"{self.log_prefix} PCR ledger unavailable: {exc}", extra={"round": server_round})
            return {}

    def _get_folder_path(self, label: str) -> str:
        match = re.fullmatch(r"\[EDGE ([1-9]\d*)_([1-9]\d*)\]", label)
        if match is None or (self.fog_num is not None and int(match[1]) != self.fog_num):
            raise ValueError("Invalid edge label or Fog assignment")
        return os.path.join(self.tpm_state_root, f"edge_{match[1]}_{match[2]}")

    def _begin_round(self, server_round):
        if self._round != server_round:
            self._round = server_round
            self._processed = set()
            self.last_outcomes = {}
            self.admission_outcomes = {}

    def ground_truth(self, cid):
        """Resolve experiment labels only after identity binding, never from role metrics."""
        bound = self.client_identities.get(cid)
        return self.run_metadata.get("experiment_ground_truth", {}).get(bound[1], {}) if bound else {}

    def _reject(self, cid, server_round, reason, attempted=True):
        bound = self.client_identities.get(cid)
        identity, label = bound or (f"CID-{cid}", f"CID-{cid}")
        if bound:
            if attempted:
                self.trust_db.process_attestation(identity, label, False, server_round)
            else:
                self.trust_db.record_missing_attestation(identity, server_round)
        self.last_outcomes[cid] = {"tpm_id": identity, "accepted": False,
                                   "authenticated": False, "reason": reason,
                                   "attempted": attempted,
                                   "expected_valid": self.ground_truth(cid).get("attestation_valid")}
        return False, identity, label

    def filter_node_updates(self, tier: str, server_round: int, results: list, active_nonces: dict,
                            expected_clients=None) -> list:
        self._begin_round(server_round)
        ledger = self.get_live_ledger(server_round) if not self.tpm_engine.insecure_mode else {}
        trusted = []
        returned = set()
        for proxy, fit_res in results:
            returned.add(proxy.cid)
            # Duplicates are ignored without changing the first authenticated result.
            if proxy.cid in self._processed:
                self.logger.warning(f"Duplicate result from {proxy.cid} ignored.", extra={"round": server_round})
                continue
            accepted, identity, label = self._verify_single_node(proxy, fit_res, server_round, active_nonces, ledger)
            # verify_pipeline consumes these post-training result messages.
            # Admission also calls _verify_single_node, so log here rather than
            # marking the update-token phase complete during admission.
            outcome = self.last_outcomes[proxy.cid]
            if outcome["authenticated"] and not self.tpm_engine.insecure_mode:
                self.logger.info(
                    f"{self.log_prefix} Ingesting Cryptographic Result for {label}",
                    extra={"round": server_round},
                )
            elif not outcome["authenticated"] and outcome.get("attempted"):
                display_label = self.client_identities.get(proxy.cid, (None, None))[1]
                if display_label is None:
                    claimed_label = fit_res.metrics.get("log_prefix", "")
                    try:
                        self._get_folder_path(claimed_label)
                        display_label = claimed_label  # Display only; not an identity binding.
                    except (ValueError, TypeError):
                        display_label = f"CID-{proxy.cid}"
                self.logger.error(
                    f"{self.log_prefix} REJECTED: Attestation/PCR mismatch for {display_label}: {outcome['reason']}",
                    extra={"round": server_round},
                )
            fit_res.metrics["tpm_id"] = identity
            fit_res.metrics["display_identity"] = label
            if accepted:
                trusted.append((proxy, fit_res))
        expected = set(active_nonces) if expected_clients is None else set(expected_clients)
        for cid in expected - returned:
            self._reject(cid, server_round, "missing_response", attempted=False)
        return trusted

    def _verify_single_node(self, client_proxy, fit_res, server_round, active_nonces, current_ledger) -> Tuple[bool, str, str]:
        self._begin_round(server_round)
        cid = client_proxy.cid
        if cid in self._processed:
            identity, label = self.client_identities.get(cid, (f"CID-{cid}", str(cid)))
            return False, identity, label
        self._processed.add(cid)
        # A missing quote is not a cryptographic false rejection. A client-provided
        # crash status is only an unverified failure report, never authentication.
        # If a token is present, verify it even when the client claims a crash.
        if not fit_res.metrics.get("tpm_token_json"):
            reason = "client_training_crash" if fit_res.metrics.get("status") == "crashed" else "missing_attestation"
            return self._reject(cid, server_round, reason, attempted=False)
        try:
            label = fit_res.metrics.get("log_prefix", "")
            folder = self._get_folder_path(label)
            token = json.loads(fit_res.metrics.get("tpm_token_json", ""))
            if not isinstance(token, dict) or not isinstance(token.get("IDi"), str) or not token["IDi"]:
                raise ValueError("Missing or invalid TPM identity")
            identity = token["IDi"]
            bound = self.client_identities.get(cid)
            if bound and bound != (identity, label):
                raise ValueError("Authenticated client identity changed")
            owner = self.identity_clients.get(identity)
            if owner is not None and owner != cid:
                raise ValueError("TPM identity already bound to another client")
            nonce = active_nonces.get(cid)
            if not nonce:
                raise ValueError("No outstanding server challenge")
            if not self.tpm_engine.insecure_mode:
                with open(os.path.join(folder, "tpm_id.txt")) as handle:
                    enrolled_id = handle.read().strip()
                if identity != enrolled_id:
                    raise ValueError("Token does not match enrolled TPM identity")
                expected_pcr = current_ledger.get(identity)
                if not expected_pcr or not self.tpm_engine.verify_attestation_token(
                    token=token, expected_nonce=nonce, public_key_path=os.path.join(folder, "ak.pub"),
                    expected_pcr=expected_pcr, max_age_seconds=int(self.run_metadata["tpm_freshness_window"]),
                    round_num=server_round,
                ):
                    raise ValueError("Attestation verification failed")
            self.client_identities[cid] = (identity, label)
            self.identity_clients[identity] = cid
            self.trust_db.process_attestation(identity, label, True, server_round)
            accepted = not self.trust_db.is_quarantined(identity)
            self.last_outcomes[cid] = {"tpm_id": identity, "authenticated": True,
                                      "accepted": accepted, "reason": "accepted" if accepted else "quarantined",
                                      "attempted": True,
                                      "expected_valid": self.ground_truth(cid).get("attestation_valid")}
            return accepted, identity, f"{identity} ({label})"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.logger.warning(f"{self.log_prefix} Rejected {cid}: {exc}", extra={"round": server_round})
            return self._reject(cid, server_round, str(exc))
