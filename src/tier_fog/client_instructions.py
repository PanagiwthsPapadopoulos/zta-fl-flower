"""Authenticate every edge before releasing a round's global parameters."""
import re
import secrets
import time
from types import SimpleNamespace

from flwr.common import Code, FitIns, GetPropertiesIns, ndarrays_to_parameters


def configure_edge_fit(instructions, gatekeepers, round_num, nonces, timeout=60.0):
    if not gatekeepers:
        raise ValueError("Edge admission requires a Fog gatekeeper")
    owners = {cid: gatekeeper for gatekeeper in gatekeepers
              for cid in gatekeeper.client_identities}
    fogs = {gatekeeper.fog_num: gatekeeper for gatekeeper in gatekeepers}
    for gatekeeper in gatekeepers:
        gatekeeper._begin_round(round_num)
        gatekeeper.attestation_only_clients = set()
        gatekeeper.admission_latency_ms = 0.0
    configured = []
    for proxy, fit_ins in instructions:
        # Admission and returned updates use separate challenges. An admission
        # quote cannot be replayed as the post-training quote.
        admission_nonce = secrets.token_hex(16)
        gatekeeper = owners.get(proxy.cid, gatekeepers[0])
        accepted = False
        started = time.perf_counter()
        try:
            response = proxy.get_properties(
                GetPropertiesIns({"server_round": round_num, "nonce": admission_nonce}),
                timeout=timeout, group_id=round_num,
            )
            if response.status.code != Code.OK:
                raise ValueError("Admission quote request failed")
            properties = dict(response.properties)
            if proxy.cid not in owners:
                label = properties.get("log_prefix", "")
                match = re.fullmatch(r"\[EDGE ([1-9]\d*)_([1-9]\d*)\]", label) if isinstance(label, str) else None
                if match:
                    # The claim selects a verifier, but is not trusted until
                    # that Fog verifies the enrolled key, identity and PCRs.
                    gatekeeper = fogs.get(int(match[1]), gatekeeper)
            ledger = {} if gatekeeper.tpm_engine.insecure_mode else gatekeeper.get_live_ledger(round_num)
            accepted, _, _ = gatekeeper._verify_single_node(
                proxy, SimpleNamespace(metrics=properties), round_num,
                {proxy.cid: admission_nonce}, ledger,
            )
        except Exception as exc:
            gatekeeper._reject(proxy.cid, round_num, f"admission_failed: {exc}", attempted=False)
        finally:
            gatekeeper.admission_latency_ms += 1000 * (time.perf_counter() - started)
            gatekeeper.admission_outcomes[proxy.cid] = dict(gatekeeper.last_outcomes[proxy.cid])
            # TrustDB counts recovery and penalties once per round even though
            # admission and returned updates have separate verification exchanges.
            gatekeeper._processed.discard(proxy.cid)

        nonce = secrets.token_hex(16)
        nonces[proxy.cid] = nonce
        config = {**fit_ins.config, "server_round": round_num, "nonce": nonce,
                  "attestation_only": not accepted}
        if not accepted:
            gatekeeper.attestation_only_clients.add(proxy.cid)
        parameters = fit_ins.parameters if accepted else ndarrays_to_parameters([])
        configured.append((proxy, FitIns(parameters, config)))
    return configured
