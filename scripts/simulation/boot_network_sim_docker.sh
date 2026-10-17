#!/bin/bash

# =========================================================
#   boot_network_sim_docker.sh
# 
#   Orchestrates the entire Dockerized Flower federation. 
#   Generates the docker-compose topology.
# =========================================================

# =========================================================
# PRE-FLIGHT CHECKS
# =========================================================
for cmd in docker python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "FATAL: Required binary '$cmd' is not installed or not in PATH."
        exit 1
    fi
done

if ! docker info >/dev/null 2>&1; then
    echo "FATAL: Docker daemon is not running or accessible."
    exit 1
fi

# =========================================================
# PATH & ENVIRONMENT CONFIGURATION
# =========================================================
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"
export LOG_DIR="$PROJECT_ROOT/logs"

# Check for --new-tpm flag
for arg in "$@"; do
    if [ "$arg" == "--new-tpm" ]; then
        echo "🧹 --new-tpm flag detected. Clearing existing TPM state..."
        rm -rf "$PROJECT_ROOT/runtime/tpm_state"
        break
    fi
done

mkdir -p "$PROJECT_ROOT/runtime/infra"

PROJECT_DIR_NAME=$(basename "$PROJECT_ROOT" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')
COMPOSE_PROJECT_NAME="${PROJECT_DIR_NAME:-flwr-federation}"

PIDS=()

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

CONFIG_VARS=$(python3 "$PROJECT_ROOT/scripts/setup/parse_topology.py")

if [ $? -ne 0 ]; then
    echo "$CONFIG_VARS"
    exit 1
fi

eval "$CONFIG_VARS"

# Provision the TPM Volumes using the unmodifiable setup script
chmod +x "$PROJECT_ROOT/scripts/setup/setup_tpm.sh"
"$PROJECT_ROOT/scripts/setup/setup_tpm.sh" "$NUM_FOGS" "${EDGES_PER_FOG_ARRAY[*]}"

# =========================================================
# 2. IMAGE RETRIEVAL
# =========================================================
echo "================================================="
echo " 2. IMAGE RETRIEVAL"
echo "================================================="
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" down --remove-orphans 2>/dev/null
mkdir -p "$LOG_DIR/system" "$LOG_DIR/nodes" "$PROJECT_ROOT/data" 

if ! docker image inspect panagiotispapadopoulos/zta-cloud-node:latest >/dev/null 2>&1; then
    echo "⏳ Pulling Pre-Built Cloud Execution Image..."
    docker pull panagiotispapadopoulos/zta-cloud-node:latest
else
    echo "  ✅ Cloud image found locally. Skipping pull."
fi

if ! docker image inspect panagiotispapadopoulos/zta-edge-node:latest >/dev/null 2>&1; then
    echo "⏳ Pulling Pre-Built Edge Execution Image (TPM Enabled)..."
    docker pull panagiotispapadopoulos/zta-edge-node:latest
else
    echo "  ✅ Edge image found locally. Skipping pull."
fi

# =========================================================
# 3. DOCKER COMPOSE TOPOLOGY GENERATION
# =========================================================
echo "================================================="
echo " 3. DOCKER COMPOSE TOPOLOGY GENERATION"
echo "================================================="
echo "  Generating dynamic docker-compose.yml..."
chmod +x "$PROJECT_ROOT/scripts/simulation/generate_compose_sim.sh"
"$PROJECT_ROOT/scripts/simulation/generate_compose_sim.sh" "$HOST_TZ" "${EDGES_PER_FOG_ARRAY[*]}"

# =================================================
#  4. BOOTING DOCKER FEDERATION                    
# =================================================
echo "================================================="
echo " 4. BOOTING DOCKER FEDERATION                    "
echo "================================================="
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
# 5. OFFLINE ZERO-TRUST NETWORK PROVISIONING (COLLECTOR)
# =========================================================
echo "================================================="
echo " 5. OFFLINE ZERO-TRUST NETWORK PROVISIONING (COLLECTOR)"
echo "================================================="

TOTAL_EDGES=0
for nodes in "${EDGES_PER_FOG_ARRAY[@]}"; do
    TOTAL_EDGES=$((TOTAL_EDGES + nodes))
done

export TOTAL_EDGES="$TOTAL_EDGES"

echo "  Polling for $TOTAL_EDGES Edge container TPM boot sequences to complete..."
python3 "$PROJECT_ROOT/scripts/setup/collect_ledgers.py"

echo ""
echo "✅ ENGINE IS LIVE. RUN DEPLOY_CODE_SIM_DOCKER.SH"
touch "$PROJECT_ROOT/runtime/infra/.network_ready"

wait