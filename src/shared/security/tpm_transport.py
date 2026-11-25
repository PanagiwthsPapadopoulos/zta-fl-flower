"""Optional native-to-Docker TPM transport using the existing TPM implementations."""
import argparse
from contextlib import redirect_stdout
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys


def container_transport_enabled():
    return bool(os.getenv("ZTA_TPM_CONTAINER_PREFIX"))


def edge_identity(value):
    match = (re.fullmatch(r"\[EDGE ([1-9]\d*)_([1-9]\d*)\]", value)
             or re.fullmatch(r"edge_([1-9]\d*)_([1-9]\d*)", value))
    if not match:
        raise ValueError("Invalid TPM edge identity")
    return int(match[1]), int(match[2])


def request_tpm(action, identity, **arguments):
    prefix = os.environ["ZTA_TPM_CONTAINER_PREFIX"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", prefix):
        raise ValueError("Invalid TPM container prefix")
    fog, edge = edge_identity(identity)
    command = ["docker", "exec", "-i", "-e", "ZTA_TPM_CONTAINER_PREFIX=",
               "-e", "ZTA_INSECURE_MODE=false", f"{prefix}{fog}-{edge}",
               "python3", "-m", "src.shared.security.tpm_transport", str(fog), str(edge)]
    result = subprocess.run(command, input=json.dumps({"action": action, **arguments}, allow_nan=False),
                            text=True, capture_output=True, check=True, timeout=60)
    response = json.loads(result.stdout)
    if not isinstance(response, dict) or "result" not in response:
        raise ValueError("Invalid TPM container response")
    return response["result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fog", type=int)
    parser.add_argument("edge", type=int)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    if min(args.fog, args.edge) < 1:
        parser.error("fog and edge must be positive")
    # Container-side calls must never recurse into Docker or bypass attestation.
    os.environ.pop("ZTA_TPM_CONTAINER_PREFIX", None)
    os.environ["ZTA_INSECURE_MODE"] = "false"
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logger = logging.getLogger("zta.tpm.container")
    label = f"[EDGE {args.fog}_{args.edge}]"
    request = ({"action": "quote", "nonce": "FACTORY_BOOT_NONCE", "round_num": 0}
               if args.bootstrap else json.load(sys.stdin))
    with redirect_stdout(sys.stderr):
        if request["action"] == "quote":
            from src.tier_edge.tpm_attestation import TPMAttestation
            engine = TPMAttestation(logger, tcti="swtpm:host=127.0.0.1,port=2321")
            result = engine.generate_attestation_token(request["nonce"], label, request["round_num"])
            if result.get("status") != "attested":
                raise RuntimeError("TPM failed to generate an attested quote")
        elif request["action"] == "verify":
            from src.tier_fog.tpm_verifier import TPMVerifier
            key = Path(f"/app/runtime/tpm_state/edge_{args.fog}_{args.edge}/ak.pub")
            result = TPMVerifier(logger).verify_attestation_token(
                request["token"], request["expected_nonce"], str(key), request["round_num"],
                request["expected_pcr"], request["max_age_seconds"])
        else:
            raise ValueError("Unknown TPM action")
    print(json.dumps({"result": result}, allow_nan=False))


if __name__ == "__main__":
    main()
