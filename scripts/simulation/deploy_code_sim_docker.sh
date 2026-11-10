#!/bin/bash
# deploy_code_sim_docker.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_FLWR="/python/venv/bin/flwr"
FEDERATION=local
cd "$PROJECT_ROOT"
BUILDER_ARGS=()
if [ "${ZTA_BATCH_MANAGED:-0}" = "1" ]; then
    BUILDER_ARGS=(--name "zta-builder-${ZTA_BATCH_ATTEMPT}" --label "zta.batch_attempt=${ZTA_BATCH_ATTEMPT}")
    printf 'preparing\n' > "$ZTA_BATCH_PHASE_FILE"
fi

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

# A batch owns one cloud container and one coordinator for its whole queue.
if [ -n "${ZTA_BATCH_SESSION:-}" ]; then
    exec docker exec -w /app \
      -e ZTA_BATCH_ATTEMPT="$ZTA_BATCH_ATTEMPT" \
      -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
      -e TORCH_NUM_THREADS=1 \
      sim-cloud /python/venv/bin/python3 /app/scripts/ops/batch_worker.py "$ZTA_BATCH_SESSION"
fi

# =========================================================
# 1. LOG ROTATION & TOPOLOGY ANALYSIS
# =========================================================
echo "================================================="
echo " 1. LOG ROTATION & TOPOLOGY ANALYSIS"
echo "================================================="
echo "🧹 Clearing stale Flower processes..."
if [ "${ZTA_BATCH_MANAGED:-0}" != "1" ]; then
docker exec sim-cloud pkill -9 -f 'flower-superlink|flower-supernode|flower-superexec|flwr-serverapp|flwr-clientapp|flwr run|flwr-simulation' 2>/dev/null || true

echo "🧹 Clearing old node logs safely..."
mkdir -p "$PROJECT_ROOT/logs/nodes/cloud"
mkdir -p "$PROJECT_ROOT/logs/system"
find "$PROJECT_ROOT/logs/nodes" -type f -delete 2>/dev/null || true
docker exec sim-cloud rm -rf /app/.flwr 2>/dev/null || true
fi

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
# 2. DATASET ARTIFACT PREPARATION
# =========================================================
echo "================================================="
echo " 🧱 COMPILING DATASET ARTIFACTS (IF MISSING)      "
echo "================================================="
# Spin up a temporary container. If it fails (!), abort everything.
if ! docker run --rm \
    ${BUILDER_ARGS[@]+"${BUILDER_ARGS[@]}"} \
    --entrypoint python3 \
    --workdir /app --env PYTHONPATH=/app \
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

echo "📝 Using the mounted application manifest; simulation node count is passed explicitly to Flower."

touch config/training.yaml

echo "🚀 [cloud] sim: 1 Central Server + ${TOTAL_EDGES} Edge Actors"

# Dispatch in the background, streaming Flower progress into the run log.
EXEC_ARGS=(-d)
if [ "${ZTA_BATCH_MANAGED:-0}" = "1" ]; then
    EXEC_ARGS=()
    printf 'training\n' > "$ZTA_BATCH_PHASE_FILE"
fi
docker exec ${EXEC_ARGS[@]+"${EXEC_ARGS[@]}"} -w /app \
  -e RUN_CFG="mode=\"sim\" tier=\"cloud\" min_clients=$TOTAL_EDGES" \
  -e N_SIM="$TOTAL_EDGES" \
  -e ZTA_BATCH_ATTEMPT="${ZTA_BATCH_ATTEMPT:-}" \
  sim-cloud bash -e -c "
    exec > /app/logs/system/run_cloud.log 2>&1
    # Ray hides CUDA from actors unless they request GPU resources.
    EDGE_GPUS=\$(python3 -c 'from src.shared.utils.hardware import get_device; print(1.0 if get_device(tier=\"edge\") == \"cuda\" else 0.0)')
    $BIN_FLWR run . $FEDERATION --stream --federation-config \"num-supernodes=\$N_SIM client-resources-num-gpus=\$EDGE_GPUS\" --run-config \"\$RUN_CFG\"
  "

echo ""
echo "✅ Global synchronization dispatched!"
echo "All output is safely redirected. Your terminal is now clean."
echo ""
echo "To monitor the background training, run:"
echo "cat logs/nodes/cloud/cloud_server.jsonl"
