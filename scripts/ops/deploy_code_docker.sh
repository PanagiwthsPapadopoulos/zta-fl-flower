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
find "$LOG_DIR/nodes/" -type f -delete 2>/dev/null
rm -f "$LOG_DIR"/system/run_*.log 2>/dev/null
echo "✅ Execution environment refreshed."

cd "$PROJECT_ROOT" || exit 1

# We update the file's "modified time" without touching its contents.
# Git ignores this, but the Docker containers will read this frozen 
# timestamp to perfectly synchronize their directory names.
touch config/training.yaml

CONFIG_VARS=$(python3 - <<EOF
import re, ast

def get_yaml_val(filepath, key, default):
    """Safely extracts YAML values using regex to avoid external host OS dependencies."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        m = re.search(fr'^{key}:\s*(.+)$', content, re.MULTILINE)
        if m:
            return m.group(1).split('#')[0].strip().strip('"').strip("'")
    except:
        pass
    return default

net_conf = 'config/network.yaml'

try:
    print(f"CLOUD_CTRL={get_yaml_val(net_conf, 'cloud_ctrl_port', '9003')}")
    print(f"FOG_CTRL_BASE={get_yaml_val(net_conf, 'fog_ctrl_base', '9300')}")

    num_fogs = int(get_yaml_val(net_conf, 'num_fogs', '2'))
    uniform = int(get_yaml_val(net_conf, 'uniform_edges_per_fog', '2'))
    
    custom_top_str = get_yaml_val(net_conf, 'custom_fog_topology', '[]')
    custom_top = ast.literal_eval(custom_top_str) if custom_top_str else []
    
    edges_arr = custom_top[:num_fogs] if custom_top and len(custom_top) >= num_fogs else [uniform] * num_fogs
    
    print(f"NUM_FOGS={num_fogs}")
    print(f"EDGES_PER_FOG_ARRAY=({' '.join(map(str, edges_arr))})")
except Exception as e:
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

# Injecting Zero_trust PKI into GRPC
if [ -f "$PROJECT_ROOT/runtime/certs/cloud_ca/ca.crt" ]; then
    COMBINED_CA="$PROJECT_ROOT/runtime/certs/combined_ca.crt"
    cat "$PROJECT_ROOT/runtime/certs/cloud_ca/ca.crt" "$PROJECT_ROOT/runtime/certs/edge_ca/ca.crt" > "$COMBINED_CA" 2>/dev/null
    export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH="$COMBINED_CA"
    echo "🔒 Custom Root CAs injected into the gRPC environment."
fi

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
echo "cat logs/nodes/cloud/cloud_server.jsonl"