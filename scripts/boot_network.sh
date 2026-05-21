#!/bin/bash
export GRPC_ENABLE_FORK_SUPPORT=1

export PATH="$(dirname "$(which python)"):$PATH"

# =========================================================
# PRECISE PATH RESOLUTION
# =========================================================
# 1. Get the directory where the script is currently running (.../scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2. Project root is exactly one level up from the scripts folder
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. Map the target directories relative to the project root
PYTHON_DIR="$PROJECT_ROOT/src/federation" 
LOG_DIR="$PROJECT_ROOT/logs"              

# =========================================================
# DYNAMIC TOPOLOGY EXTRACTION (Reads pyproject.toml)
# =========================================================
echo "================================================="
echo " 🔍 READING TOPOLOGY FROM pyproject.toml         "
echo "================================================="

# Use Heredoc to prevent shell quoting collisions with Python
CONFIG_VARS=$(python - <<EOF
import re, ast

try:
    with open('$PROJECT_ROOT/pyproject.toml', 'r') as f:
        content = f.read()
    
    def get_val(key, default):
        # Extract quoted strings or numeric/dot values
        m = re.search(fr'{key}\s*=\s*(["0-9\.]+)', content)
        return m.group(1).replace('"', '') if m else default

    # Network Variables
    print(f"BROKER_IP={get_val('broker_ip', '127.0.0.1')}")
    print(f"CLOUD_SA={get_val('cloud_sa_port', '9091')}")
    print(f"CLOUD_FL={get_val('cloud_fl_port', '9092')}")
    print(f"CLOUD_CTRL={get_val('cloud_ctrl_port', '9093')}")
    print(f"FOG_SA_BASE={get_val('fog_sa_base', '9190')}")
    print(f"FOG_FL_BASE={get_val('fog_fl_base', '9290')}")
    print(f"FOG_CTRL_BASE={get_val('fog_ctrl_base', '9390')}")
    print(f"FOG_CIO_BASE={get_val('fog_client_io_base', '9490')}")
    print(f"EDGE_CIO_BASE={get_val('edge_client_io_base', '9500')}")

    # Topology Variables
    num_fogs_match = re.search(r'num_fogs\s*=\s*(\d+)', content)
    uniform_match = re.search(r'uniform_edges_per_fog\s*=\s*(\d+)', content)
    custom_match = re.search(r'custom_fog_topology\s*=\s*"(\[.*?\])"', content)
    
    num_fogs = int(num_fogs_match.group(1)) if num_fogs_match else 2
    uniform = int(uniform_match.group(1)) if uniform_match else 2
    
    custom_top = []
    if custom_match:
        try:
            custom_top = ast.literal_eval(custom_match.group(1))
        except:
            pass
            
    if custom_top and len(custom_top) >= num_fogs:
        edges_array = custom_top[:num_fogs]
    else:
        edges_array = [uniform] * num_fogs
        
    print(f"NUM_FOGS={num_fogs}")
    print(f"EDGES_PER_FOG_ARRAY=({' '.join(map(str, edges_array))})")
    
except Exception as e:
    print(f'echo "Error parsing pyproject.toml: {e}"')
    print('exit 1')
EOF
)

# Execute the exported variables into the bash environment
eval "$CONFIG_VARS"

echo "Discovered Fog Nodes: $NUM_FOGS"
echo "Edge Distribution per Fog: (${EDGES_PER_FOG_ARRAY[*]})"
echo ""

echo "================================================="
echo " ⚠️ WARNING: GLOBAL CONFIGURATION OVERWRITE"
echo "================================================="
echo "This script will overwrite your global ~/.flwr/config.toml."
read -p "Do you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 1
fi

echo "================================================="
echo " 1. SYSTEM PURGE: Clearing previous network...   "
echo "================================================="
pkill -9 -f flower-superlink 2>/dev/null
pkill -9 -f flower-supernode 2>/dev/null
pkill -9 -f "flwr run" 2>/dev/null
pkill -9 -f flwr-serverapp 2>/dev/null
pkill -9 -f flwr-clientapp 2>/dev/null
sleep 5 

rm -rf "$LOG_DIR"
# Create segregated folders
mkdir -p "$LOG_DIR/system"
mkdir -p "$LOG_DIR/nodes"

# =========================================================
# GLOBAL FLOWER CONFIGURATION INJECTION (~/.flwr)
# =========================================================
# Pointing to the user's global Flower directory
FLWR_GLOBAL_DIR="$HOME/.flwr"
mkdir -p "$FLWR_GLOBAL_DIR"

# Initialize the global config file with the Cloud SuperLink
cat <<EOF > "$FLWR_GLOBAL_DIR/config.toml"
[superlink.cloud]
address = "$BROKER_IP:$CLOUD_CTRL"
insecure = true
EOF

for i in $(seq 1 $NUM_FOGS); do
    FOG_CTRL=$((FOG_CTRL_BASE + i))
    cat <<EOF >> "$FLWR_GLOBAL_DIR/config.toml"

[superlink.fog${i}]
address = "$BROKER_IP:${FOG_CTRL}"
insecure = true
EOF
done
# =========================================================

cd "$PROJECT_ROOT" || { echo "Directory $PROJECT_ROOT not found!"; exit 1; }

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

PIDS=()
cleanup() {
    echo -e "\nShutting down 3-Tier Architecture..."
    for pid in "${PIDS[@]}"; do kill -9 $pid 2>/dev/null; done
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "================================================="
echo " 2. BOOTING SECURE, ISOLATED ARCHITECTURE        "
echo "================================================="

# ---------------------------------------------------------
# TIER 1: CLOUD INFRASTRUCTURE
# ---------------------------------------------------------
flower-superlink --insecure \
    --serverappio-api-address $BROKER_IP:$CLOUD_SA \
    --fleet-api-address $BROKER_IP:$CLOUD_FL \
    --control-api-address $BROKER_IP:$CLOUD_CTRL > "$LOG_DIR/system/cloud_superlink.log" 2>&1 &
PIDS+=($!)

sleep 2

# ---------------------------------------------------------
# TIER 2 & 3: FOG AND EDGE INFRASTRUCTURE
# ---------------------------------------------------------
for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]}
    
    # Unique ports for this specific Fog node
    FOG_SA=$((FOG_SA_BASE + i))
    FOG_FL=$((FOG_FL_BASE + i))
    FOG_CTRL=$((FOG_CTRL_BASE + i))
    FOG_CLIENT_IO=$((FOG_CIO_BASE + i))

    # 1. Fog as a Client: Connects UP to the Cloud's Fleet API
    flower-supernode --insecure \
        --superlink $BROKER_IP:$CLOUD_FL \
        --clientappio-api-address $BROKER_IP:$FOG_CLIENT_IO \
        --node-config "fog_id=${i}" > "$LOG_DIR/system/fog${i}_supernode.log" 2>&1 &
    PIDS+=($!)    

    # 2. Fog as a Server: Creates its own SuperLink for its Edges
    flower-superlink --insecure \
        --serverappio-api-address $BROKER_IP:$FOG_SA \
        --fleet-api-address $BROKER_IP:$FOG_FL \
        --control-api-address $BROKER_IP:$FOG_CTRL > "$LOG_DIR/system/fog${i}_superlink.log" 2>&1 &
    PIDS+=($!)

    
    # ---------------------------------------------------------
    # TIER 3: EDGE INFRASTRUCTURE (Dynamically scaled per Fog)
    # ---------------------------------------------------------
    for j in $(seq 1 $CURRENT_EDGES); do
        EDGE_CLIENT_IO=$((EDGE_CIO_BASE + (i * 100) + j))
        
        # FIXED: Pass j directly so the ID matches 1, 2, 3
        flower-supernode --insecure \
            --superlink $BROKER_IP:$FOG_FL \
            --clientappio-api-address $BROKER_IP:$EDGE_CLIENT_IO \
            --node-config "fog_num=${i} partition-id=${j}" > "$LOG_DIR/system/edge${i}_${j}_supernode.log" 2>&1 &
        PIDS+=($!)
    done
done

sleep 3

echo "================================================="
echo " 3. SYSTEM ARCHITECTURE & PORT SUMMARY           "
echo "================================================="
echo ""
echo "☁️  [CLOUD] SuperLink"
echo "    ├─ Fleet API:   $BROKER_IP:$CLOUD_FL  <-- (Fogs connect here)"
echo "    ├─ Control API: $BROKER_IP:$CLOUD_CTRL  <-- (flwr run . cloud)"
echo "    └─ ServerAppIO: $BROKER_IP:$CLOUD_SA"
echo "    │"

for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]}
    
    # FIXED: Using the new base variables instead of hardcoded math
    FOG_SA=$((FOG_SA_BASE + i))
    FOG_FL=$((FOG_FL_BASE + i))
    FOG_CTRL=$((FOG_CTRL_BASE + i))
    FOG_CLIENT_IO=$((FOG_CIO_BASE + i))

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

    echo "    $PREFIX 🌫️  [FOG $i] SuperNode (ClientAppIO: $FOG_CLIENT_IO) --> Connects to Cloud Fleet $CLOUD_FL"
    echo "    $SPACER │"
    echo "    $SPACER └─ [FOG $i] SuperLink"
    echo "    $SPACER     ├─ Fleet API:   $BROKER_IP:$FOG_FL  <-- (Edges connect here)"
    echo "    $SPACER     ├─ Control API: $BROKER_IP:$FOG_CTRL  <-- (flwr run . fog$i)"
    echo "    $SPACER     └─ ServerAppIO: $BROKER_IP:$FOG_SA"
    
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 $CURRENT_EDGES); do
            EDGE_CLIENT_IO=$((EDGE_CIO_BASE + (i * 100) + j))
            if [ "$j" -eq "$CURRENT_EDGES" ]; then
                EDGE_PREFIX="└──"
            else
                EDGE_PREFIX="├──"
            fi
            # FIXED: Display the dynamic IP and Ports perfectly
            echo "    $SPACER          $EDGE_PREFIX 📱 [EDGE ${i}_${j}] SuperNode (ClientAppIO: $EDGE_CLIENT_IO) --> Connects to Fog Fleet $FOG_FL"
        done
    else
        echo "    $SPACER          └── (No Edge Nodes Assigned)"
    fi
    
    if [ "$i" -ne "$NUM_FOGS" ]; then
        echo "    │"
    fi
done

echo ""

echo ""
echo "================================================="
echo " 4. DEPLOYING APPLICATIONS TO NODES              "
echo "================================================="

for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]}
    
    # If a Fog has 0 edges, we pass 1 to min-clients to avoid crashing, but the Fog strategy handles empty rounds
    SAFE_MIN_CLIENTS=$(( CURRENT_EDGES > 0 ? CURRENT_EDGES : 1 ))
    
    echo "Shipping FAB to Fog $i (Expecting $CURRENT_EDGES edges)..."
    flwr run . fog${i} --run-config "tier=\"fog\" min-clients=${SAFE_MIN_CLIENTS} fog_id=\"fog_${i}\"" --stream > "$LOG_DIR/system/run_fog${i}.log" 2>&1 &
    PIDS+=($!)

    echo "  ⏳ Cooling down Fog $i stack..."
    sleep 3
done

sleep 2

# Submit run for the Cloud node and stream directly to terminal
echo "Shipping FAB to Cloud ..."
flwr run . cloud --run-config "tier=\"cloud\" min-clients=${NUM_FOGS}" --stream > "$LOG_DIR/system/run_cloud.log" 2>&1 &
PIDS+=($!)

echo ""
echo "✅ Global synchronization started in the background."
echo "Check the logs to watch progress: logs/nodes/ or logs/system"
echo "Press Ctrl+C to stop the network."
wait