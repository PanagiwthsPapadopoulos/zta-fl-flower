#!/bin/bash

# =========================================================
#   deploy_code_docker.sh
# 
#   Acts as the "Fuel" for the infrastructure. Deploys code
#   bundles (FABs) to the SuperLink network dynamically.
#   Cleans stale socket files to prevent runtime deadlocks.
# =========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/logs"

# =========================================================
# PRE-FLIGHT DEPENDENCY & HEALTH CHECK
# =========================================================
# Check for the state flag
if [ ! -f "$PROJECT_ROOT/runtime/infra/.network_ready" ]; then
    echo "❌ FATAL: The SuperLink network is not running or failed to boot."
    echo "   Please run 'boot_network_docker.sh' first and wait for the success message."
    exit 1
fi

# Verify Docker containers are actively running for this project
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"

# Get a list of running container IDs for this specific compose file
RUNNING_CONTAINERS=$(docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" ps --status running -q 2>/dev/null)

if [ -z "$RUNNING_CONTAINERS" ]; then
    echo "❌ FATAL: State mismatch detected. The lock file exists, but the Docker network is offline."
    echo "   Cleaning up stale lock file..."
    rm -f "$PROJECT_ROOT/runtime/.network_ready"
    echo "   Please run 'boot_network_docker.sh' to boot the network."
    exit 1
fi

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


# =========================================================
# 1. LOG ROTATION & TOPOLOGY ANALYSIS
# =========================================================
mkdir -p "$LOG_DIR/system"
find "$LOG_DIR/nodes/" -type f -delete 2>/dev/null
rm -f "$LOG_DIR"/system/run_*.log 2>/dev/null
echo "✅ Execution environment refreshed."

cd "$PROJECT_ROOT" || exit 1

# We update the file's "modified time" without touching its contents.
# Git ignores this, but the Docker containers will read this frozen 
# timestamp to perfectly synchronize their directory names.
touch config/training.yaml

# Execute the standalone Python parser
CONFIG_VARS=$(python3 "$PROJECT_ROOT/scripts/setup/parse_topology.py")
eval "$CONFIG_VARS"

echo "================================================="
echo " SYSTEM ARCHITECTURE & TOPOLOGY SUMMARY       "
echo "================================================="
echo ""
echo "☁️  [CLOUD] Docker SuperLink (Tier 1)"
echo "    ├─ Internal Fleet: cloud-superlink:9092"
echo "    └─ Control API:    localhost:$CLOUD_CTRL <-- (flwr run . cloud)"
echo "    │"

for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    FOG_CTRL=$((FOG_CTRL_BASE + i))

    if [ "$i" -eq 1 ]; then
        PREFIX="├──"
        SPACER="│   "
    elif [ "$i" -eq "$NUM_FOGS" ]; then
        PREFIX="└──"
        SPACER="    "
    else
        PREFIX="├──"
        SPACER="│   "
    fi

    echo "    $PREFIX 🌫️  [FOG $i] Docker SuperNode & SuperLink (Tier 2)"
    echo "    $SPACER │"
    echo "    $SPACER ├─ Connects Up To: cloud-superlink:9092"
    echo "    $SPACER ├─ Internal Fleet: fog-${i}-superlink:9092"
    echo "    $SPACER └─ Control API:    localhost:$FOG_CTRL <-- (flwr run . fog${i})"
    
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            if [ "$j" -eq "$CURRENT_EDGES" ]; then
                EDGE_PREFIX="└──"
            else
                EDGE_PREFIX="├──"
            fi
            echo "    $SPACER          $EDGE_PREFIX 📱 [EDGE ${i}_${j}] Docker SuperNode -> Connects to fog-${i}-superlink:9092"
        done
    else
        echo "    $SPACER          └── (No Edge Nodes Assigned)"
    fi
    
    if [ "$i" -ne "$NUM_FOGS" ]; then
        echo "    │"
    fi
done

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
# 3. FAB DEPLOYMENT DISPATCH
# =========================================================
pkill -f "flwr run" 2>/dev/null

for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    SAFE_MIN_CLIENTS=$(( CURRENT_EDGES > 0 ? CURRENT_EDGES : 1 ))
    
    FOG_CTRL=$((FOG_CTRL_BASE + i))
    
    echo "Shipping FAB to Fog $i (Expecting $CURRENT_EDGES edges)..."
    flwr run . fog${i} --run-config "tier=\"fog\" min-clients=${SAFE_MIN_CLIENTS} fog_id=\"fog_${i}\"" --stream > "$LOG_DIR/system/run_fog${i}.log" 2>&1 &

    echo "  ⏳ Cooling down Fog $i stack..."
    wait_for_port 127.0.0.1 $FOG_CTRL
done

sleep 2

echo "Shipping FAB to Cloud ..."
flwr run . cloud --run-config "tier=\"cloud\" min-clients=${NUM_FOGS}" --stream > "$LOG_DIR/system/run_cloud.log" 2>&1 &

echo ""
echo "✅ Global synchronization dispatched!"
echo "All output is safely redirected. Your terminal is now clean."
echo ""
echo "To monitor the background training, run:"
echo "cat logs/nodes/cloud/cloud_server.jsonl"