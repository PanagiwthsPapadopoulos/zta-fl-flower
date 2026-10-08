#!/bin/bash
# =========================================================
#   deploy_code_apptainer.sh
# 
#   Acts as the "Fuel" for the infrastructure. Deploys code
#   bundles (FABs) to the Apptainer SuperLink network.
# =========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/logs"

# =========================================================
# PRE-FLIGHT DEPENDENCY & HEALTH CHECK
# =========================================================
if [ ! -f "$PROJECT_ROOT/runtime/infra/.network_ready" ]; then
    echo "❌ FATAL: The SuperLink network is not running or failed to boot."
    echo "   Please run 'boot_network_apptainer.sh' first and wait for the success message."
    exit 1
fi

if ! pgrep -f "flower-superlink" > /dev/null; then
    echo "❌ FATAL: State mismatch detected. The lock file exists, but the Apptainer network is offline."
    echo "   Cleaning up stale lock file..."
    rm -f "$PROJECT_ROOT/runtime/infra/.network_ready"
    echo "   Please run 'boot_network_apptainer.sh' to boot the network."
    exit 1
fi

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
touch config/training.yaml

CONFIG_VARS=$(python3 "$PROJECT_ROOT/scripts/setup/parse_topology.py")
eval "$CONFIG_VARS"

echo "================================================="
echo " SYSTEM ARCHITECTURE & TOPOLOGY SUMMARY       "
echo "================================================="
echo ""
echo "☁️  [CLOUD] Apptainer SuperLink (Tier 1)"
echo "    ├─ Internal Fleet: 127.0.0.1:19002 (or 9002 insecure)"
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

    echo "    $PREFIX 🌫️  [FOG $i] Apptainer SuperNode & SuperLink (Tier 2)"
    echo "    $SPACER │"
    echo "    $SPACER ├─ Connects Up To: 127.0.0.1"
    echo "    $SPACER ├─ Internal Fleet: 127.0.0.1"
    echo "    $SPACER └─ Control API:    localhost:$FOG_CTRL <-- (flwr run . fog${i})"
    
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            if [ "$j" -eq "$CURRENT_EDGES" ]; then
                EDGE_PREFIX="└──"
            else
                EDGE_PREFIX="├──"
            fi
            echo "    $SPACER          $EDGE_PREFIX 📱 [EDGE ${i}_${j}] Apptainer SuperNode -> Connects to Fog ${i}"
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
if ! apptainer exec \
    --bind "$PROJECT_ROOT/data:/app/data" \
    --bind "$PROJECT_ROOT/config:/app/config:ro" \
    --bind "$PROJECT_ROOT/src:/app/src:ro" \
    --bind "$PROJECT_ROOT/scripts:/app/scripts:ro" \
    "$PROJECT_ROOT/images/cloud_node.sif" \
    python3 /app/scripts/setup/build_artifacts.py; then
    
    echo "🛑 FATAL: Artifact compilation failed! Aborting network boot."
    exit 1
fi

echo "✅ Dataset Artifacts Verified!"

# =========================================================
# 3. FAB DEPLOYMENT DISPATCH
# =========================================================
# 1. Kill the deployment dispatchers (including Apptainer wrappers)
pkill -9 -f "flwr run" 2>/dev/null || true

# 2. Kill the ephemeral server-side FAB sessions
pkill -9 -f "flwr-serverapp" 2>/dev/null || true

# 3. Kill the ephemeral client-side FAB sessions
pkill -9 -f "flwr-clientapp" 2>/dev/null || true

for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    SAFE_MIN_CLIENTS=$(( CURRENT_EDGES > 0 ? CURRENT_EDGES : 1 ))
    FOG_CTRL=$((FOG_CTRL_BASE + i))
    
    echo "Shipping FAB to Fog $i (Expecting $CURRENT_EDGES edges)..."
    apptainer exec --pwd /app --bind "$PROJECT_ROOT:/app" "$PROJECT_ROOT/images/cloud_node.sif" \
        /python/venv/bin/flwr run /app fog${i} --run-config "tier=\"fog\" min-clients=${SAFE_MIN_CLIENTS} fog_id=\"fog_${i}\"" --stream > "$LOG_DIR/system/run_fog${i}.log" 2>&1 &

    echo "  ⏳ Cooling down Fog $i stack..."
    wait_for_port 127.0.0.1 $FOG_CTRL
done

sleep 2

echo "Shipping FAB to Cloud ..."
apptainer exec --pwd /app --bind "$PROJECT_ROOT:/app" "$PROJECT_ROOT/images/cloud_node.sif" \
        /python/venv/bin/flwr run /app cloud --run-config "tier=\"cloud\" min-clients=${NUM_FOGS}" --stream > "$LOG_DIR/system/run_cloud.log" 2>&1 &

echo ""
echo "✅ Global synchronization dispatched!"
echo "All output is safely redirected. Your terminal is now clean."
echo ""
echo "To monitor the background training, run:"
echo "cat logs/nodes/cloud/cloud_server.jsonl"
