#!/bin/bash
# deploy_code_sim_docker.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_FLWR="/python/venv/bin/flwr"
FEDERATION=local

# =========================================================
# PRE-FLIGHT DEPENDENCY & HEALTH CHECK
# =========================================================
if [ ! -f "$PROJECT_ROOT/runtime/infra/.network_ready" ]; then
    echo "❌ FATAL: The simulation network is not running or failed to boot."
    echo "   Please run 'boot_network_sim_docker.sh' first and wait for the success message."
    exit 1
fi

COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"
RUNNING_CONTAINERS=$(docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" ps --status running -q 2>/dev/null)

if [ -z "$RUNNING_CONTAINERS" ]; then
    echo "❌ FATAL: State mismatch detected. The lock file exists, but the Docker network is offline."
    echo "   Cleaning up stale lock file..."
    rm -f "$PROJECT_ROOT/runtime/infra/.network_ready"
    echo "   Please run 'boot_network_sim_docker.sh' to boot the network."
    exit 1
fi

# =========================================================
# 1. LOG ROTATION & TOPOLOGY ANALYSIS
# =========================================================
echo "================================================="
echo " 1. LOG ROTATION & TOPOLOGY ANALYSIS"
echo "================================================="
echo "🧹 Clearing stale Flower processes..."
docker exec sim-cloud pkill -9 -f 'flower-superlink|flower-supernode|flower-superexec|flwr-serverapp|flwr-clientapp|flwr run|flwr-simulation' 2>/dev/null || true

echo "🧹 Clearing old node logs safely..."
mkdir -p "$PROJECT_ROOT/logs/nodes/cloud"
mkdir -p "$PROJECT_ROOT/logs/system"
find "$PROJECT_ROOT/logs/nodes" -type f -delete 2>/dev/null || true
docker exec sim-cloud rm -rf /app/.flwr 2>/dev/null || true

TOTAL_EDGES=$(docker ps --format '{{.Names}}' | grep -c '^sim-tpm-' || echo 0)

if [ "$TOTAL_EDGES" -eq 0 ]; then
  echo "❌ No sim-tpm-* containers found. Ensure the network is booted."
  exit 1
fi

echo "✅ Execution environment refreshed."
echo "📖 Detected ${TOTAL_EDGES} Edge Nodes in the local network."

CONFIG_VARS=$(python3 "$PROJECT_ROOT/scripts/setup/parse_topology.py")
eval "$CONFIG_VARS"

echo "================================================="
echo " SYSTEM ARCHITECTURE & TOPOLOGY SUMMARY       "
echo "================================================="
echo ""
echo "🐳 Docker Host Network"
echo "│"
echo "├── 📦 Container: sim-cloud (The Simulation Engine)"
echo "│   │"
echo "│   ├── ☁️  [CLOUD] Global Aggregator (Tier 1)"

for i in $(seq 1 $NUM_FOGS); do
    echo "│   │"
    if [ "$i" -eq "$NUM_FOGS" ]; then
        FOG_PREFIX="│   └──"
        SPACER="│       "
    else
        FOG_PREFIX="│   ├──"
        SPACER="│   │   "
    fi
    
    echo "${FOG_PREFIX} 🌫️  [FOG $i] Virtual Aggregator (Tier 2)"
    
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            if [ "$j" -eq "$CURRENT_EDGES" ]; then
                EDGE_PREFIX="└──"
            else
                EDGE_PREFIX="├──"
            fi
            echo "${SPACER} ${EDGE_PREFIX} 📱 [EDGE ${i}_${j}] Ephemeral Ray Actor  ----(TCP)----> 📦 Container: sim-tpm-${i}-${j}:2321"
        done
    else
        echo "${SPACER} └── (No Edge Nodes Assigned)"
    fi
done

echo "│"
echo "└── 📦 External Hardware Emulators (${TOTAL_EDGES} sim-tpm-* Containers)"

# =========================================================
# 2. ARTIFACT BUILDER (THE "BIG CRUNCH")
# =========================================================
echo "================================================="
echo " 🧱 COMPILING DATASET ARTIFACTS (IF MISSING)      "
echo "================================================="
# Spin up a temporary container. If it fails (!), abort everything.
if ! docker run --rm \
    --entrypoint python3 \
    -v "$PROJECT_ROOT/data:/app/data" \
    -v "$PROJECT_ROOT/config:/app/config:ro" \
    -v "$PROJECT_ROOT/src:/app/src:ro" \
    -v "$PROJECT_ROOT/scripts:/app/scripts:ro" \
    panagiotispapadopoulos/zta-cloud-node:latest \
    /app/scripts/setup/build_artifacts.py; then
    
    echo "🛑 FATAL: Artifact compilation failed! Aborting network boot."
    exit 1
fi

echo "✅ Dataset Artifacts Verified!"

# =========================================================
# 3. SIMULATION DEPLOYMENT DISPATCH
# =========================================================
echo "================================================="
echo " 3. SIMULATION DEPLOYMENT DISPATCH"
echo "================================================="

TEMPLATE="$PROJECT_ROOT/pyproject.toml"
echo "📝 Rewriting pyproject for sim-cloud (num-supernodes=${TOTAL_EDGES})"
sed "s/^options.num-supernodes = .*/options.num-supernodes = ${TOTAL_EDGES}/" "$TEMPLATE" > /tmp/pyproject.cloud.toml
docker cp /tmp/pyproject.cloud.toml sim-cloud:/app/pyproject.toml

touch config/training.yaml

echo "🚀 [cloud] sim: 1 Central Server + ${TOTAL_EDGES} Edge Actors"

# Deployment runs natively in the background (-d), completely muting noisy Flower warnings
docker exec -d -w /app \
  -e RUN_CFG="mode=\"sim\" tier=\"cloud\" min_clients=$TOTAL_EDGES" \
  -e N_SIM="$TOTAL_EDGES" \
  sim-cloud bash -c "$BIN_FLWR run . $FEDERATION --run-config \"\$RUN_CFG\" > /app/logs/system/run_cloud.log 2>&1"

echo ""
echo "✅ Global synchronization dispatched!"
echo "All output is safely redirected. Your terminal is now clean."
echo ""
echo "To monitor the background training, run:"
echo "cat logs/nodes/cloud/cloud_server.jsonl"