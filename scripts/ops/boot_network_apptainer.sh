#!/bin/bash
# =========================================================
#   boot_network_apptainer.sh
# 
#   Orchestrates the entire Apptainer Flower federation natively.
#   Uses unified root bindings and standard Flower orchestrators.
# =========================================================

INSECURE_MODE=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --insecure) INSECURE_MODE=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure swtpm is NOT in this list
for cmd in apptainer python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "FATAL: Required binary '$cmd' is not installed or not in PATH."
        exit 1
    fi
done

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export LOG_DIR="$PROJECT_ROOT/logs"
export CERTS_DIR="$PROJECT_ROOT/runtime/certs"
export NGINX_CONF="$PROJECT_ROOT/runtime/infra/nginx.conf"
export IMAGE_DIR="$PROJECT_ROOT/images"
export LOCKS_DIR="$PROJECT_ROOT/runtime/locks"

mkdir -p "$PROJECT_ROOT/runtime/infra" "$LOG_DIR/system" "$LOG_DIR/nodes" "$PROJECT_ROOT/data" "$IMAGE_DIR" "$LOCKS_DIR"
chmod 777 "$LOCKS_DIR"

mkdir -p "$LOG_DIR/nodes/cloud"
PIDS=()

cleanup() {
    echo -e "\n🛑 Caught Shutdown Signal! Shutting down the Apptainer Engine..."
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    rm -f "$PROJECT_ROOT/runtime/infra/.network_ready"
    echo "✅ Teardown complete. Network is offline."
    exit 0
}

trap cleanup SIGINT SIGTERM SIGTSTP

if [ -L /etc/localtime ]; then
    HOST_TZ=$(readlink /etc/localtime | sed 's#^.*zoneinfo/##')
else
    HOST_TZ="UTC"
fi
export TZ="$HOST_TZ"

# Inject the virtual environment path so SuperNodes can find and spawn ClientApps
export APPTAINERENV_PATH="/python/venv/bin:$PATH"

echo "="
echo " 🔍 READING TOPOLOGY FROM network.yaml           "
echo "="

CONFIG_VARS=$(python3 "$PROJECT_ROOT/scripts/setup/parse_topology.py")
if [ $? -ne 0 ]; then
    echo "$CONFIG_VARS"
    exit 1
fi
eval "$CONFIG_VARS"

# =========================================================
# HOST GPU CAPABILITY DETECTION
# =========================================================
# This is the only GPU-related decision the boot script makes.
# It answers a hardware question, not a policy question.
# The YAML key `gpu_enabled_tiers` is deliberately NOT read here;
# containers read it themselves at runtime.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_ARGS="--nv"
    ZTA_GPU_AVAILABLE="true"
    echo "🟢 Host GPU detected. Every container will be launched with --nv."
else
    GPU_ARGS=""
    ZTA_GPU_AVAILABLE="false"
    echo "⚠️  No usable host GPU. Containers launched without --nv."
    echo "⚠️  Even if 'gpu_enabled_tiers' is non-empty, every tier will run on CPU."

fi

if [ "$INSECURE_MODE" = false ]; then
    echo "  SECURITY ENABLED: Generating certificates and NGINX config..."

    rm -rf "$CERTS_DIR"
    mkdir -p "$CERTS_DIR"

    # --- Cert generation (container-native script, hardcoded /app paths) ---
    apptainer exec \
        --env OPENSSL_CONF= \
        --env LOG_DIR=/app/logs/system \
        --bind "$PROJECT_ROOT:/app" --pwd /app \
        "$IMAGE_DIR/cloud_node.sif" \
        bash /app/scripts/setup/setup_security.sh \
            "$NUM_FOGS" "${EDGES_PER_FOG_ARRAY[*]}" "127.0.0.1"

    # --- Hard fail if certs missing (setup_security.sh always exits 0) ---
    if [ ! -f "$CERTS_DIR/cloud_server/certificates.pem" ] \
       || [ ! -f "$CERTS_DIR/cloud_ca/ca.crt" ]; then
        echo "FATAL: Certificate generation failed."
        echo "---- security_setup.log ----"
        cat "$LOG_DIR/system/security_setup.log" 2>/dev/null || true
        exit 1
    fi
    echo "  ✓ Certs generated:"
    ls -la "$CERTS_DIR/cloud_server"

    # --- NGINX config (unchanged; only consumes certs from disk) ---
    apptainer exec \
        --env OPENSSL_CONF= \
        --env PROJECT_ROOT=/app \
        --bind "$PROJECT_ROOT:/app" --pwd /app \
        "$IMAGE_DIR/cloud_node.sif" \
        /app/scripts/setup/setup_nginx.sh \
            "$NUM_FOGS" "${EDGES_PER_FOG_ARRAY[*]}" \
            "127.0.0.1" "$FOG_FL_BASE" "false"
fi

chmod +x "$PROJECT_ROOT/scripts/setup/setup_tpm.sh"
"$PROJECT_ROOT/scripts/setup/setup_tpm.sh" "$NUM_FOGS" "${EDGES_PER_FOG_ARRAY[*]}"

# =========================================================
# 2. IMAGE RETRIEVAL
# =========================================================
cd "$IMAGE_DIR"
if [ ! -f "cloud_node.sif" ]; then
    echo "⏳ Pulling Pre-Built Cloud Execution Image..."
    apptainer pull cloud_node.sif docker://panagiotispapadopoulos/zta-cloud-node:latest
fi
if [ ! -f "edge_node.sif" ]; then
    echo "⏳ Pulling Pre-Built Edge Execution Image (TPM Enabled)..."
    apptainer pull edge_node.sif docker://panagiotispapadopoulos/zta-edge-node:latest
fi
[ ! -f "nginx.sif" ] && echo "⏳ Pulling NGINX..." && apptainer pull nginx.sif docker://nginx:alpine
cd "$PROJECT_ROOT"

# =========================================================
# 3. BOOTING APPTAINER FEDERATION
# =========================================================
echo "  Starting Apptainer network processes..."

INTERNAL_CLOUD_FL=$([ "$INSECURE_MODE" = false ] && echo "1${CLOUD_FL}" || echo "${CLOUD_FL}")

apptainer exec $GPU_ARGS --pwd /app --env TZ="${HOST_TZ}" --env ZTA_INSECURE_MODE="${INSECURE_MODE}" \
    --env ZTA_TIER=cloud --env ZTA_GPU_AVAILABLE="${ZTA_GPU_AVAILABLE}" \
    --bind "$PROJECT_ROOT:/app" "$IMAGE_DIR/cloud_node.sif" /python/venv/bin/flower-superlink \
    --isolation process --insecure --serverappio-api-address "127.0.0.1:${CLOUD_SA}" \
    --fleet-api-address "127.0.0.1:${INTERNAL_CLOUD_FL}" --control-api-address "127.0.0.1:${CLOUD_CTRL}" \
    > "$LOG_DIR/system/cloud.log" 2>&1 &
PIDS+=($!)

# Cloud ServerApp
apptainer exec $GPU_ARGS --pwd /app --env TZ="${HOST_TZ}" --env ZTA_INSECURE_MODE="${INSECURE_MODE}" \
    --env ZTA_TIER=cloud --env ZTA_GPU_AVAILABLE="${ZTA_GPU_AVAILABLE}" \
    --bind "$LOG_DIR/nodes/cloud:/app/logs,$PROJECT_ROOT/data:/app/data:ro,$PROJECT_ROOT/results:/app/results,$PROJECT_ROOT/config:/app/config:ro" \
    "$IMAGE_DIR/cloud_node.sif" \
    flower-superexec --insecure --plugin-type serverapp --appio-api-address "127.0.0.1:${CLOUD_SA}" \
    > "$LOG_DIR/system/cloud_serverapp.log" 2>&1 &
PIDS+=($!)

if [ "$INSECURE_MODE" = false ]; then
    apptainer exec --writable-tmpfs \
        --bind "$PROJECT_ROOT:/app" \
        "$IMAGE_DIR/nginx.sif" nginx -c /app/runtime/infra/nginx.conf -g "daemon off;" \
        > "$LOG_DIR/system/nginx.log" 2>&1 &
    PIDS+=($!)
fi

for i in $(seq 1 "$NUM_FOGS"); do
    FOG_SA=$((FOG_SA_BASE + i)); FOG_FL=$((FOG_FL_BASE + i)); FOG_CTRL=$((FOG_CTRL_BASE + i)); FOG_CLIENT_IO=$((FOG_CIO_BASE + i))
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    FOG_LOG_MOUNT="$LOG_DIR/nodes/fog_${i}"
    mkdir -p "$FOG_LOG_MOUNT"
    
    CLOUD_UPLINK=$([ "$INSECURE_MODE" = false ] && echo "127.0.0.1:$((CLOUD_FL + 20000 + i))" || echo "127.0.0.1:${CLOUD_FL}")
    FOG_INTERNAL_FL=$([ "$INSECURE_MODE" = false ] && echo "$((FOG_FL_BASE + 10000 + i))" || echo "$FOG_FL")

    apptainer exec $GPU_ARGS --pwd /app --env TZ="${HOST_TZ}" --env FOG_SERVER_HOST="127.0.0.1" --env IPC_PORT="${FOG_CLIENT_IO}" \
        --env ZTA_TIER=fog --env ZTA_GPU_AVAILABLE="${ZTA_GPU_AVAILABLE}" \
        --bind "$PROJECT_ROOT:/app" "$IMAGE_DIR/cloud_node.sif" /python/venv/bin/flower-supernode \
        --isolation process --insecure --superlink "${CLOUD_UPLINK}" \
        --clientappio-api-address "127.0.0.1:${FOG_CLIENT_IO}" --node-config "fog_id=${i}" \
        > "$LOG_DIR/system/fog_${i}_supernode.log" 2>&1 &
    PIDS+=($!)
    
    # Fog ClientApp
    apptainer exec $GPU_ARGS --pwd /app --env TZ="${HOST_TZ}" --env FOG_SERVER_HOST="127.0.0.1" --env IPC_PORT="${FOG_CLIENT_IO}" \
    --env ZTA_TIER=fog --env ZTA_GPU_AVAILABLE="${ZTA_GPU_AVAILABLE}" \
    --bind "$LOG_DIR/nodes/fog_${i}:/app/logs,$PROJECT_ROOT/data:/app/data:ro,$PROJECT_ROOT/config:/app/config:ro" \
    "$IMAGE_DIR/cloud_node.sif" \
    flower-superexec --insecure --plugin-type clientapp --appio-api-address "127.0.0.1:${FOG_CLIENT_IO}" \
    > "$LOG_DIR/system/fog_${i}_clientapp.log" 2>&1 &
PIDS+=($!)

    apptainer exec $GPU_ARGS --pwd /app --env TZ="${HOST_TZ}" --env IPC_PORT="${FOG_SA}" --env ZTA_INSECURE_MODE="${INSECURE_MODE}" \
        --env ZTA_TIER=fog --env ZTA_GPU_AVAILABLE="${ZTA_GPU_AVAILABLE}" \
        --bind "$PROJECT_ROOT:/app" "$IMAGE_DIR/cloud_node.sif" /python/venv/bin/flower-superlink \
        --isolation process --insecure --serverappio-api-address "127.0.0.1:${FOG_SA}" \
        --fleet-api-address "127.0.0.1:${FOG_INTERNAL_FL}" --control-api-address "127.0.0.1:${FOG_CTRL}" \
        > "$LOG_DIR/system/fog_${i}_superlink.log" 2>&1 &
    PIDS+=($!)
    
    # Fog ServerApp
    apptainer exec $GPU_ARGS --pwd /app --env TZ="${HOST_TZ}" --env IPC_PORT="${FOG_SA}" --env ZTA_INSECURE_MODE="${INSECURE_MODE}" \
    --env ZTA_TIER=fog --env ZTA_GPU_AVAILABLE="${ZTA_GPU_AVAILABLE}" \
    --bind "$LOG_DIR/nodes/fog_${i}:/app/logs,$PROJECT_ROOT/data:/app/data:ro,$PROJECT_ROOT/config:/app/config:ro,$PROJECT_ROOT/runtime/tpm_state:/app/runtime/tpm_state:rw,$PROJECT_ROOT/results:/app/results" \
    "$IMAGE_DIR/cloud_node.sif" \
    flower-superexec --insecure --plugin-type serverapp --appio-api-address "127.0.0.1:${FOG_SA}" \
    > "$LOG_DIR/system/fog_${i}_serverapp.log" 2>&1 &
PIDS+=($!)

    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_CLIENT_IO=$((EDGE_CIO_BASE + (i * 100) + j))
            EDGE_LOG_MOUNT="$LOG_DIR/nodes/edge_${i}_${j}"
            mkdir -p "$EDGE_LOG_MOUNT"
            
            EDGE_UPLINK=$([ "$INSECURE_MODE" = false ] && echo "127.0.0.1:$((FOG_FL_BASE + 20000 + (i * 100) + j))" || echo "127.0.0.1:${FOG_INTERNAL_FL}")
            
            EDGE_TPM_DIR="$PROJECT_ROOT/runtime/tpm_state/edge_${i}_${j}"
            CONTAINER_TPM_DIR="/app/runtime/tpm_state/edge_${i}_${j}"
            SOCK_PATH="${CONTAINER_TPM_DIR}/swtpm.sock"
            CTRL_PATH="${CONTAINER_TPM_DIR}/swtpm.sock.ctrl"

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
                --env TPM2TOOLS_TCTI="swtpm:path=${SOCK_PATH}" \
                --bind "$PROJECT_ROOT/runtime/tpm_state:/app/runtime/tpm_state:rw" \
                --bind "$PROJECT_ROOT:/app" \
                "$IMAGE_DIR/edge_node.sif" bash -c "
                    tpm2_startup -c || true
                    python3 -c \"
import logging
from src.tier_edge.tpm_attestation import TPMAttestation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Bootstrapper')
tpm = TPMAttestation(logger=logger)
tpm.generate_attestation_token(nonce='FACTORY_BOOT_NONCE', software_label='[EDGE ${i}_${j}]', round_num=0)
\"
" > "$LOG_DIR/system/edge_${i}_${j}_init.log" 2>&1 &

            echo "--> [Edge ${i}_${j}] Starting SuperNode..."
            apptainer exec $GPU_ARGS --pwd /app \
                --env TZ="${HOST_TZ}" \
                --env TPM2TOOLS_TCTI="swtpm:path=${SOCK_PATH}" \
                --env ZTA_INSECURE_MODE="${INSECURE_MODE}" \
                --env ZTA_TIER=edge --env ZTA_GPU_AVAILABLE="${ZTA_GPU_AVAILABLE}" \
                --bind "$PROJECT_ROOT/runtime/tpm_state:/app/runtime/tpm_state:rw" \
                --bind "$PROJECT_ROOT:/app" \
                "$IMAGE_DIR/edge_node.sif" /python/venv/bin/flower-supernode \
                    --isolation process \
                    --insecure \
                    --superlink "${EDGE_UPLINK}" \
                    --clientappio-api-address "127.0.0.1:${EDGE_CLIENT_IO}" \
                    --node-config "fog_num=${i} partition-id=${j}" \
                    > "$LOG_DIR/system/edge_${i}_${j}.log" 2>&1 &
            PIDS+=($!)
            
            # Edge ClientApp
	    apptainer exec $GPU_ARGS --pwd /app \
        --env TZ="${HOST_TZ}" \
        --env TPM2TOOLS_TCTI="swtpm:path=${SOCK_PATH}" --env ZTA_INSECURE_MODE="${INSECURE_MODE}" \
        --env ZTA_TIER=edge --env ZTA_GPU_AVAILABLE="${ZTA_GPU_AVAILABLE}" \
        --bind "$LOG_DIR/nodes/edge_${i}_${j}:/app/logs,$PROJECT_ROOT/data:/app/data:ro,$PROJECT_ROOT/config:/app/config:ro,$PROJECT_ROOT/runtime/tpm_state/edge_${i}_${j}:/app/runtime/tpm_state/edge_${i}_${j}:rw,$LOCKS_DIR:/app/runtime/locks:rw" \
        "$IMAGE_DIR/edge_node.sif" \
        flower-superexec --insecure --plugin-type clientapp --appio-api-address "127.0.0.1:${EDGE_CLIENT_IO}" \
        > "$LOG_DIR/system/edge_${i}_${j}_clientapp.log" 2>&1 &
	    PIDS+=($!)
        done
    fi
done

# =========================================================
# 5. OFFLINE ZERO-TRUST NETWORK PROVISIONING (COLLECTOR)
# =========================================================
echo "="
echo " 🛡️  FACTORY PROVISIONING (COLLECTING STATES)     "
echo "="

TOTAL_EDGES=0
for nodes in "${EDGES_PER_FOG_ARRAY[@]}"; do TOTAL_EDGES=$((TOTAL_EDGES + nodes)); done
export TOTAL_EDGES="$TOTAL_EDGES"

echo "  Polling for $TOTAL_EDGES Edge container TPM boot sequences to complete..."
python3 "$PROJECT_ROOT/scripts/setup/collect_ledgers.py"

FLWR_GLOBAL_DIR="$HOME/.flwr"
mkdir -p "$FLWR_GLOBAL_DIR"

cat << EOF > "$FLWR_GLOBAL_DIR/config.toml"
[superlink.cloud]
address = "127.0.0.1:$CLOUD_CTRL"
insecure = true

[federation.cloud]
address = "127.0.0.1:$CLOUD_CTRL"
insecure = true
EOF

for i in $(seq 1 "$NUM_FOGS"); do
    FOG_CTRL=$((FOG_CTRL_BASE + i))
    cat << EOF >> "$FLWR_GLOBAL_DIR/config.toml"

[superlink.fog${i}]
address = "127.0.0.1:${FOG_CTRL}"
insecure = true

[federation.fog${i}]
address = "127.0.0.1:${FOG_CTRL}"
insecure = true
EOF
done

echo ""
echo "✅ ENGINE IS LIVE. RUN DEPLOY_CODE_APPTAINER.SH"
touch "$PROJECT_ROOT/runtime/infra/.network_ready"
wait
