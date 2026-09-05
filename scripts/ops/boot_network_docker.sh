#!/bin/bash

# =========================================================
#   boot_network_docker.sh
# 
#   Orchestrates the entire Dockerized Flower federation. 
#   Generates the docker-compose topology, provisions TLS 
#   certificates via mTLS, and boots the infrastructure layer.
#
# ARGUMENTS:
#   --insecure : Optional. Disables mTLS and runs all 
#                services in plain-text mode.
# =========================================================

# Default to secure mode (TLS/mTLS)
INSECURE_MODE=false

# =========================================================
# PRE-FLIGHT CHECKS
# =========================================================
# Validate Dependencies
for cmd in docker python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "FATAL: Required binary '$cmd' is not installed or not in PATH."
        exit 1
    fi
done

# Validate Docker Daemon
if ! docker info >/dev/null 2>&1; then
    echo "FATAL: Docker daemon is not running or accessible."
    exit 1
fi

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --insecure) INSECURE_MODE=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# =========================================================
# PATH & ENVIRONMENT CONFIGURATION
# =========================================================
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"
export LOG_DIR="$PROJECT_ROOT/logs"
export CERTS_DIR="$PROJECT_ROOT/runtime/certs"
export NGINX_CONF="$PROJECT_ROOT/runtime/nginx.conf"

# Dynamically determine a safe Compose project name based on the root directory
PROJECT_DIR_NAME=$(basename "$PROJECT_ROOT" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')
COMPOSE_PROJECT_NAME="${PROJECT_DIR_NAME:-flwr-federation}"

PIDS=()

# =========================================================
# UTILITIES: CLEANUP & TIMEZONE DETECTION
# =========================================================
cleanup() {
    echo -e "\n🛑 Caught Shutdown Signal (Ctrl+C / Ctrl+Z)! Shutting down the Docker Engine..."
    docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" down --remove-orphans 2>/dev/null
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

# =========================================================
# DYNAMIC TOPOLOGY EXTRACTION
# =========================================================
echo "================================================="
echo " 🔍 READING TOPOLOGY FROM network.yaml           "
echo "================================================="

# Execute the standalone Python parser
CONFIG_VARS=$(python3 "$PROJECT_ROOT/scripts/setup/parse_topology.py")

# Ensure the script didn't fail before evaluating
if [ $? -ne 0 ]; then
    echo "$CONFIG_VARS"
    exit 1
fi

# Inject variables into the bash environment
eval "$CONFIG_VARS"

# =========================================================
# 1. SECURITY SETUP & NGINX PROVISIONING
# =========================================================
if [ "$INSECURE_MODE" = false ]; then
    echo "🔐 SECURITY ENABLED: Generating certificates and NGINX config..."
    chmod +x "$PROJECT_ROOT/scripts/setup/setup_security.sh"
    chmod +x "$PROJECT_ROOT/scripts/setup/setup_nginx.sh"
    "$PROJECT_ROOT/scripts/setup/setup_security.sh" "$NUM_FOGS" "${EDGES_PER_FOG_ARRAY[*]}" "127.0.0.1"
    "$PROJECT_ROOT/scripts/setup/setup_nginx.sh" "$NUM_FOGS" "${EDGES_PER_FOG_ARRAY[*]}" "127.0.0.1" "$FOG_FL_BASE" "true"
fi

# Provision the TPM Volumes for the Edge Nodes
chmod +x "$PROJECT_ROOT/scripts/setup/setup_tpm.sh"
"$PROJECT_ROOT/scripts/setup/setup_tpm.sh" "$NUM_FOGS" "${EDGES_PER_FOG_ARRAY[*]}"

# =========================================================
# 2. IMAGE RETRIEVAL
# =========================================================
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" down --remove-orphans 2>/dev/null
mkdir -p "$LOG_DIR/system" "$LOG_DIR/nodes" "$PROJECT_ROOT/data" 

if ! docker image inspect panagiotispapadopoulos/zta-cloud-node:latest >/dev/null 2>&1; then
    echo "⏳ Pulling Pre-Built Cloud Execution Image..."
    docker pull panagiotispapadopoulos/zta-cloud-node:latest
fi

if ! docker image inspect panagiotispapadopoulos/zta-edge-node:latest >/dev/null 2>&1; then
    echo "⏳ Pulling Pre-Built Edge Execution Image (TPM Enabled)..."
    docker pull panagiotispapadopoulos/zta-edge-node:latest
fi

# =========================================================
# 3. DOCKER COMPOSE TOPOLOGY GENERATION
# =========================================================

echo "  Generating dynamic docker-compose.yml..."
chmod +x "$PROJECT_ROOT/scripts/setup/generate_compose.sh"
"$PROJECT_ROOT/scripts/setup/generate_compose.sh" "$INSECURE_MODE" "$HOST_TZ" "${EDGES_PER_FOG_ARRAY[*]}"

# =================================================
#  4. BOOTING DOCKER FEDERATION                    
# =================================================
cd "$PROJECT_ROOT" || exit 1

echo "  Starting Docker Compose network and waiting for health checks..."

if ! docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" up -d --wait; then
    echo "❌ FATAL: Network failed to boot or health checks timed out."
    docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" down
    exit 1
fi

docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" logs -f > "$LOG_DIR/system/docker_mesh.log" 2>&1 &
PIDS+=($!)

# =========================================================
# 5 OFFLINE ZERO-TRUST NETWORK PROVISIONING (COLLECTOR)
# =========================================================
echo "================================================="
echo " 🛡️  FACTORY PROVISIONING (COLLECTING STATES)     "
echo "================================================="

# Calculate total number of edges
TOTAL_EDGES=0
for nodes in "${EDGES_PER_FOG_ARRAY[@]}"; do
    TOTAL_EDGES=$((TOTAL_EDGES + nodes))
done

# Export variables so the Python script knows what to look for and how many to wait for
export TOTAL_EDGES="$TOTAL_EDGES"

echo "  Polling for $TOTAL_EDGES Edge container TPM boot sequences to complete..."
# Execute the decoupled script (which now handles its own deterministic polling)
python3 "$PROJECT_ROOT/scripts/setup/collect_ledgers.py"

# =================================================
# 6. INJECTING GLOBAL CONFIGURATION (~/.flwr)     
# =================================================
FLWR_GLOBAL_DIR="$HOME/.flwr"
mkdir -p "$FLWR_GLOBAL_DIR"

cat <<EOF > "$FLWR_GLOBAL_DIR/config.toml"
[superlink.cloud]
address = "127.0.0.1:$CLOUD_CTRL"
insecure = ${INSECURE_MODE}
EOF

if [ "$INSECURE_MODE" = false ]; then
    echo "root-certificates = \"$CERTS_DIR/cloud_ca/ca.crt\"" >> "$FLWR_GLOBAL_DIR/config.toml"
fi

cat <<EOF >> "$FLWR_GLOBAL_DIR/config.toml"
[federation.cloud]
address = "127.0.0.1:$CLOUD_CTRL"
insecure = ${INSECURE_MODE}
EOF

if [ "$INSECURE_MODE" = false ]; then
    echo "root-certificates = \"$CERTS_DIR/cloud_ca/ca.crt\"" >> "$FLWR_GLOBAL_DIR/config.toml"
fi

for i in $(seq 1 $NUM_FOGS); do
    FOG_CTRL=$((FOG_CTRL_BASE + i))
    cat <<EOF >> "$FLWR_GLOBAL_DIR/config.toml"

[superlink.fog${i}]
address = "127.0.0.1:${FOG_CTRL}"
insecure = true

[federation.fog${i}]
address = "127.0.0.1:${FOG_CTRL}"
insecure = true
EOF
done

echo ""
echo "✅ ENGINE IS LIVE. RUN DEPLOY_CODE_DOCKER.SH"

# Create the state flag indicating the network is ready
mkdir -p "$PROJECT_ROOT/runtime/infra"
touch "$PROJECT_ROOT/runtime/infra/.network_ready"

wait