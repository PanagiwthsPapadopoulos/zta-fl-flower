#!/bin/bash

# =========================================================
#   deply_code_docker.sh
# 
#   Acts as the "Fuel" for the infrastructure. Deploys code
#   bundles (FABs) to the SuperLink network dynamically.
#   Cleans stale socket files to prevent runtime deadlocks.
# =========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/logs"

# =========================================================
# 1. ORPHAN PROCESS CLEANUP
# =========================================================
CONTAINERS=$(docker ps --format "{{.Names}}")
for container in $CONTAINERS; do
    if [[ "$container" == *"serverapp"* ]] || [[ "$container" == *"clientapp"* ]]; then
        echo "  Cleaning $container..."
        docker exec "$container" pkill -9 -f python 2>/dev/null
    fi
done

# =========================================================
# 2. LOG ROTATION & TOPOLOGY ANALYSIS
# =========================================================
mkdir -p "$LOG_DIR/system"
rm -f "$LOG_DIR"/nodes/* 2>/dev/null
rm -f "$LOG_DIR"/system/run_*.log 2>/dev/null
echo "✅ Execution environment refreshed."

cd "$PROJECT_ROOT" || exit 1

CONFIG_VARS=$(python3 - <<EOF
import re, ast
try:
    with open('pyproject.toml', 'r') as f: content = f.read()
    
    def get_val(key, default):
        m = re.search(fr'{key}\s*=\s*(["0-9\.]+)', content)
        return m.group(1).replace('"', '') if m else default

    print(f"CLOUD_CTRL={get_val('cloud_ctrl_port', '9093')}")
    print(f"FOG_CTRL_BASE={get_val('fog_ctrl_base', '9390')}")

    m_fogs = re.search(r'num_fogs\s*=\s*(\d+)', content)
    num_fogs = int(m_fogs.group(1)) if m_fogs else 2
    
    m_uni = re.search(r'uniform_edges_per_fog\s*=\s*(\d+)', content)
    uniform = int(m_uni.group(1)) if m_uni else 2
    
    c_match = re.search(r'custom_fog_topology\s*=\s*"(\[.*?\])"', content)
    custom_top = ast.literal_eval(c_match.group(1)) if c_match else []
    
    edges_arr = custom_top[:num_fogs] if custom_top and len(custom_top) >= num_fogs else [uniform] * num_fogs
    print(f"NUM_FOGS={num_fogs}")
    print(f"EDGES_PER_FOG_ARRAY=({' '.join(map(str, edges_arr))})")
except:
    pass
EOF
)
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
# 3. FAB DEPLOYMENT DISPATCH
# =========================================================
pkill -f "flwr run" 2>/dev/null

for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    SAFE_MIN_CLIENTS=$(( CURRENT_EDGES > 0 ? CURRENT_EDGES : 1 ))
    
    echo "Shipping FAB to Fog $i (Expecting $CURRENT_EDGES edges)..."
    flwr run . fog${i} --run-config "tier=\"fog\" min-clients=${SAFE_MIN_CLIENTS} fog_id=\"fog_${i}\"" --stream > "$LOG_DIR/system/run_fog${i}.log" 2>&1 &

    echo "  ⏳ Cooling down Fog $i stack..."
    sleep 3
done

sleep 2

echo "Shipping FAB to Cloud ..."
flwr run . cloud --run-config "tier=\"cloud\" min-clients=${NUM_FOGS}" --stream > "$LOG_DIR/system/run_cloud.log" 2>&1 &

echo ""
echo "✅ Global synchronization dispatched!"
echo "All output is safely redirected. Your terminal is now clean."
echo ""
echo "To monitor the background training, run:"
echo "cat logs/system/run_cloud.log"