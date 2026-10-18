#!/bin/bash
# =========================================================
#   boot_network_sim_apptainer.sh
#   Orchestrates the Apptainer Flower simulation environment.
#   Boots background swtpm daemons and provisions Zero-Trust
#   TPM states for simulated Edge actors.
# =========================================================

# =========================================================
# PRE-FLIGHT CHECKS
# =========================================================
for cmd in apptainer python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "FATAL: Required binary '$cmd' is not installed or not in PATH."
        exit 1
    fi
done

# =========================================================
# PATH & ENVIRONMENT CONFIGURATION
# =========================================================
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export LOG_DIR="$PROJECT_ROOT/logs"
export IMAGE_DIR="$PROJECT_ROOT/images"
export LOCKS_DIR="$PROJECT_ROOT/runtime/locks"

# Check for --new-tpm flag
for arg in "$@"; do
    if [ "$arg" == "--new-tpm" ]; then
        echo "  --new-tpm flag detected. Clearing existing TPM state..."
        rm -rf "$PROJECT_ROOT/runtime/tpm_state"
        break
    fi
done

mkdir -p "$PROJECT_ROOT/runtime/infra" \
         "$LOG_DIR/system" \
         "$LOG_DIR/nodes/cloud" \
         "$PROJECT_ROOT/data" \
         "$IMAGE_DIR" \
         "$LOCKS_DIR" \
         "$PROJECT_ROOT/runtime/tpm_state"
chmod 777 "$LOCKS_DIR"

PIDS=()

cleanup() {
    echo -e "\n  Caught Shutdown Signal (Ctrl+C / Ctrl+Z)! Shutting down the Apptainer Simulation Engine..."
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    pkill -9 -f 'flwr run|flwr-simulation|flower-superlink|flower-supernode|flower-superexec' 2>/dev/null || true
    rm -f "$PROJECT_ROOT/runtime/infra/.network_ready"
    echo "  Teardown complete. Network is offline."
    exit 0
}
trap cleanup SIGINT SIGTERM SIGTSTP

if [ -L /etc/localtime ]; then
    HOST_TZ=$(readlink /etc/localtime | sed 's#^.*zoneinfo/##')
else
    HOST_TZ="UTC"
fi
export TZ="$HOST_TZ"
export APPTAINERENV_PATH="/python/venv/bin:$PATH"

# =========================================================
# 1. DYNAMIC TOPOLOGY EXTRACTION
# =========================================================
echo "================================================="
echo "   READING TOPOLOGY FROM network.yaml           "
echo "================================================="

CONFIG_VARS=$(python3 "$PROJECT_ROOT/scripts/setup/parse_topology.py")
if [ $? -ne 0 ]; then
    echo "$CONFIG_VARS"
    exit 1
fi
eval "$CONFIG_VARS"

# Provision the TPM Volumes using the setup script
chmod +x "$PROJECT_ROOT/scripts/setup/setup_tpm.sh"
"$PROJECT_ROOT/scripts/setup/setup_tpm.sh" "$NUM_FOGS" "${EDGES_PER_FOG_ARRAY[*]}"

# =========================================================
# 2. IMAGE RETRIEVAL
# =========================================================
echo "================================================="
echo " 2. IMAGE RETRIEVAL"
echo "================================================="

cd "$IMAGE_DIR" || exit 1
if [ ! -f "cloud_node.sif" ]; then
    echo "  Pulling Pre-Built Cloud Execution Image..."
    apptainer pull cloud_node.sif docker://panagiotispapadopoulos/zta-cloud-node:latest
else
    echo "    Cloud image found locally. Skipping pull."
fi

if [ ! -f "edge_node.sif" ]; then
    echo "  Pulling Pre-Built Edge Execution Image (TPM Enabled)..."
    apptainer pull edge_node.sif docker://panagiotispapadopoulos/zta-edge-node:latest
else
    echo "    Edge image found locally. Skipping pull."
fi
cd "$PROJECT_ROOT" || exit 1

# =========================================================
# 3. BOOTING APPTAINER TPM EMULATORS
# =========================================================
echo "================================================="
echo " 3. BOOTING APPTAINER TPM EMULATORS             "
echo "================================================="

for i in $(seq 1 "$NUM_FOGS"); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_TPM_DIR="$PROJECT_ROOT/runtime/tpm_state/edge_${i}_${j}"
            CONTAINER_TPM_DIR="/app/runtime/tpm_state/edge_${i}_${j}"
            SOCK_PATH="${CONTAINER_TPM_DIR}/swtpm.sock"
            CTRL_PATH="${CONTAINER_TPM_DIR}/swtpm.sock.ctrl"

            mkdir -p "$EDGE_TPM_DIR"
            rm -f "$EDGE_TPM_DIR/swtpm.sock" "$EDGE_TPM_DIR/swtpm.sock.ctrl"

            echo "--> [Edge ${i}_${j}] Starting swtpm via UNIX socket..."
            apptainer exec \
                --bind "$PROJECT_ROOT/runtime/tpm_state:/app/runtime/tpm_state:rw" \
                "$IMAGE_DIR/edge_node.sif" swtpm socket \
                    --tpmstate dir="$CONTAINER_TPM_DIR" \
                    --tpm2 \
                    --server type=unixio,path="$SOCK_PATH" \
                    --ctrl type=unixio,path="$CTRL_PATH" \
                    --flags startup-clear \
                    > "$LOG_DIR/system/edge_${i}_${j}_swtpm.log" 2>&1 &
            PIDS+=($!)

            while [ ! -S "$EDGE_TPM_DIR/swtpm.sock" ]; do
                sleep 0.5
            done
            sleep 1

            apptainer exec --pwd /app \
                --env TZ="${HOST_TZ}" \
                --env TPM2TOOLS_TCTI="swtpm:path=${SOCK_PATH}" \
                --bind "$PROJECT_ROOT/runtime/tpm_state:/app/runtime/tpm_state:rw" \
                --bind "$PROJECT_ROOT:/app" \
                "$IMAGE_DIR/edge_node.sif" bash -c "
                    tpm2_startup -c || true
                    PYTHONPATH=/app python3 -c \"
import logging
from src.tier_edge.tpm_attestation import TPMAttestation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Bootstrapper')
tpm = TPMAttestation(logger=logger)
tpm.generate_attestation_token(nonce='FACTORY_BOOT_NONCE', software_label='[EDGE ${i}_${j}]', round_num=0)
\"
                    echo 'TPM ready'
                " > "$LOG_DIR/system/edge_${i}_${j}_init.log" 2>&1 &
        done
    fi
done

# =========================================================
# 4. OFFLINE ZERO-TRUST NETWORK PROVISIONING (COLLECTOR)
# =========================================================
echo "================================================="
echo " 4. OFFLINE ZERO-TRUST NETWORK PROVISIONING (COLLECTOR)"
echo "================================================="

TOTAL_EDGES=0
for nodes in "${EDGES_PER_FOG_ARRAY[@]}"; do
    TOTAL_EDGES=$((TOTAL_EDGES + nodes))
done
export TOTAL_EDGES="$TOTAL_EDGES"

echo "  Polling for $TOTAL_EDGES Edge container TPM boot sequences to complete..."
python3 "$PROJECT_ROOT/scripts/setup/collect_ledgers.py"

echo ""
echo "  ENGINE IS LIVE. RUN DEPLOY_CODE_SIM_APPTAINER.SH"
touch "$PROJECT_ROOT/runtime/infra/.network_ready"
wait