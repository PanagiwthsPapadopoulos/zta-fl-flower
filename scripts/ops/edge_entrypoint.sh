#!/bin/bash
set -e

# Helper function for polling a port
wait_for_port() {
    local host=$1
    local port=$2
    for i in {1..30}; do
        if nc -z "$host" "$port" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    echo "Timeout waiting for $host:$port"
    exit 1
}

# Validate required TPM and Python binaries
for cmd in swtpm tpm2_startup python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "FATAL: Required binary '$cmd' is not installed inside the Edge container."
        exit 1
    fi
done

echo "🛡️ Starting Hardware Root of Trust (TPM 2.0)..."

TPM_DIR=$(find /app/runtime/tpm_state -mindepth 1 -maxdepth 1 -type d -name "edge_*" | head -n 1)

if [ -z "$TPM_DIR" ]; then
    echo "🛑 FATAL: Could not find the mounted edge directory in /app/runtime/tpm_state/"
    exit 1
fi

# Unlock the volume so Python can write the .bin file
chmod 700 "$TPM_DIR"

echo "🔌 Booting swtpm daemon inside $TPM_DIR..."

swtpm socket --tpmstate dir="$TPM_DIR" \
             --tpm2 \
             --server type=tcp,port=2321 \
             --ctrl type=tcp,port=2322 \
             --flags startup-clear \
             --daemon

wait_for_port 127.0.0.1 2321

export TPM2TOOLS_TCTI="swtpm:port=2321"
echo "✅ TPM TCTI configured."

# Initialize TPM hardware state natively
tpm2_startup -c || true

# =====================================================================
# NATIVE PROVISIONING: Call the existing TPMAttestation automatically
# =====================================================================
echo "⚙️ Provisioning TPM Identity and PCR Baseline natively..."
EDGE_NAME=$(basename "$TPM_DIR")
FOG_NUM=$(echo $EDGE_NAME | cut -d'_' -f2)
EDGE_NUM=$(echo $EDGE_NAME | cut -d'_' -f3)

python3 -c "
import logging
from src.tier_edge.tpm_attestation import TPMAttestation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Bootstrapper')

# This automatically runs _provision_ak() 
tpm = TPMAttestation(logger=logger)

# Automatically export the ID and PCR to the volume
tpm.generate_attestation_token(nonce='FACTORY_BOOT_NONCE', software_label='[EDGE ${FOG_NUM}_${EDGE_NUM}]', round_num=0)
"
echo "✅ TPM Bootstrap complete. Keys and Baseline secured in volume."
# =====================================================================

echo "🌸 Booting Flower Client..."
exec flower-superexec "$@"