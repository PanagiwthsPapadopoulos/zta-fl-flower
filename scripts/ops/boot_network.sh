#!/bin/bash

# =========================================================
# !!! OBSOLETE !!!
# This script has been superseded by `boot_network_docker.sh`.
# Do NOT use this script for orchestration, as it lacks Docker 
# container encapsulation and proper service isolation.
# Refer to the Docker-based workflow for actual deployments.
# =========================================================

# =========================================================
# LOCAL SIMULATION OVERRIDES
# NOTE: The following two exports are required ONLY for macOS/Apple Silicon 
# environments to prevent known gRPC and multiprocessing deadlocks. 
# If deploying to Linux/Windows architectures, these can be safely removed.
# =========================================================
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export GRPC_ENABLE_FORK_SUPPORT=1

# Prepend local Python environment to PATH
export PATH="$(dirname "$(which python)"):$PATH"

# =========================================================
# PRECISE PATH RESOLUTION
# =========================================================
# Resolve absolute paths to ensure deterministic execution regardless of invocation directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON_DIR="$PROJECT_ROOT/src/federation" 
LOG_DIR="$PROJECT_ROOT/logs"     
CERTS_DIR="$PROJECT_ROOT/src/network/certs"
NGINX_CONF="$PROJECT_ROOT/src/network/nginx.conf"

# =========================================================
# DYNAMIC TOPOLOGY EXTRACTION (Reads pyproject.toml)
# =========================================================
echo "================================================="
echo " 🔍 READING TOPOLOGY FROM pyproject.toml         "
echo "================================================="

# Parse pyproject.toml via an embedded Python script to extract network architecture constraints
CONFIG_VARS=$(python - <<EOF
import re, ast

try:
    with open('$PROJECT_ROOT/pyproject.toml', 'r') as f:
        content = f.read()
    
    # Helper to parse key-value pairs using regex
    def get_val(key, default):
        m = re.search(fr'{key}\s*=\s*(["0-9\.]+)', content)
        return m.group(1).replace('"', '') if m else default

    # Output bash-compatible variable assignments
    print(f"BROKER_IP={get_val('broker_ip', '127.0.0.1')}")
    print(f"CLOUD_SA={get_val('cloud_sa_port', '9091')}")
    print(f"CLOUD_FL={get_val('cloud_fl_port', '9092')}")
    print(f"CLOUD_CTRL={get_val('cloud_ctrl_port', '9093')}")
    print(f"FOG_SA_BASE={get_val('fog_sa_base', '9190')}")
    print(f"FOG_FL_BASE={get_val('fog_fl_base', '9290')}")
    print(f"FOG_CTRL_BASE={get_val('fog_ctrl_base', '9390')}")
    print(f"FOG_CIO_BASE={get_val('fog_client_io_base', '9490')}")
    print(f"EDGE_CIO_BASE={get_val('edge_client_io_base', '9500')}")

    # Parse dynamic network topology (number of Fogs and Edges)
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
            
    # Resolve Edge node distribution across the Fog layer
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

# Inject parsed variables into the current bash environment
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
# Aggressively terminate any lingering application and sidecar processes
pkill -9 -f flower-superlink 2>/dev/null
pkill -9 -f flower-supernode 2>/dev/null
pkill -9 -f "flwr run" 2>/dev/null
pkill -9 -f flwr-serverapp 2>/dev/null
pkill -9 -f flwr-clientapp 2>/dev/null
pkill -9 -f nginx 2>/dev/null

# Wipe Flower's global caching layer to prevent stale App Bundle deployments
rm -rf ~/.flwr/fab 2>/dev/null
rm -rf ~/.flwr/node 2>/dev/null

# Reinitialize local logging and state directories
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR/system"
mkdir -p "$LOG_DIR/nodes"

# Provide buffer time for the OS kernel to release TIME_WAIT TCP sockets
sleep 5 

echo "================================================="
echo " 2. GENERATING ZERO-TRUST TLS CERTIFICATES       "
echo "================================================="

SETUP_SCRIPT="$PROJECT_ROOT/scripts/setup/setup_security.sh"
NGINX_SCRIPT="$PROJECT_ROOT/scripts/setup/setup_nginx.sh"

if [ ! -f "$SETUP_SCRIPT" ]; then
    echo "❌ Error: Could not find $SETUP_SCRIPT"
    exit 1
fi

# Execute Public Key Infrastructure (PKI) setup
chmod +x "$SETUP_SCRIPT"
SETUP_OUTPUT=$("$SETUP_SCRIPT" "$NUM_FOGS" "${EDGES_PER_FOG_ARRAY[*]}" "$BROKER_IP")
SETUP_EXIT_CODE=$?

if [ $SETUP_EXIT_CODE -ne 0 ]; then
    echo "❌ Security setup was aborted or failed."
    exit 1
fi

if echo "$SETUP_OUTPUT" | grep -q "STATUS:KEPT"; then
    echo "✅ Preserving existing static cryptographic identities."
elif echo "$SETUP_OUTPUT" | grep -q "STATUS:GENERATED"; then
    echo "✅ Successfully minted new cryptographic identities."
fi

# Execute NGINX sidecar routing configuration
chmod +x "$NGINX_SCRIPT"
"$NGINX_SCRIPT" "$NUM_FOGS" "${EDGES_PER_FOG_ARRAY[*]}" "$BROKER_IP" "$FOG_FL_BASE"

# =========================================================
# GLOBAL SECURE FLOWER CONFIGURATION INJECTION (~/.flwr)
# =========================================================
FLWR_GLOBAL_DIR="$HOME/.flwr"
mkdir -p "$FLWR_GLOBAL_DIR"

# Initialize global configuration for the Cloud SuperLink
cat <<EOF > "$FLWR_GLOBAL_DIR/config.toml"
[superlink.cloud]
address = "$BROKER_IP:$CLOUD_CTRL"
insecure = false
root-certificates = "$CERTS_DIR/cloud_ca/ca.crt"
EOF

# Inject control plane routing for all Fog intermediate nodes
for i in $(seq 1 $NUM_FOGS); do
    FOG_CTRL=$((FOG_CTRL_BASE + i))
    cat <<EOF >> "$FLWR_GLOBAL_DIR/config.toml"

[superlink.fog${i}]
address = "$BROKER_IP:${FOG_CTRL}"
insecure = true
EOF
done

cd "$PROJECT_ROOT" || { echo "Directory $PROJECT_ROOT not found!"; exit 1; }
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Setup process tracking array and graceful teardown handler
PIDS=()
cleanup() {
    echo -e "\n🛑 Forcefully shutting down entire network..."
    exec 2>/dev/null
    
    # Kill the parent processes spawned by this script
    for pid in "${PIDS[@]}"; do kill -9 $pid 2>/dev/null; done
    
    # Kill any orphaned background workers that escaped
    pkill -9 -f flower 2>/dev/null
    pkill -9 -f flwr 2>/dev/null
    pkill -9 -f nginx 2>/dev/null
    
    rm -rf ~/.flwr/fab 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# =========================================================
# HELPER: WAIT FOR PORT (DETERMINISTIC BOOT)
# =========================================================
# Blocks execution until a specific TCP port is actively listening
wait_for_port() {
    local port=$1
    echo "  ⏳ Waiting for port $port to open..."
    for k in {1..30}; do
        if nc -z 127.0.0.1 $port 2>/dev/null; then
            echo "✅ Port $port is open."
            return 0
        fi
        sleep 1
    done
    echo "❌ Timeout waiting for port $port"
    cleanup
    return 1
}

echo "================================================="
echo " 3. BOOTING SECURE, ISOLATED ARCHITECTURE        "
echo "================================================="

# ---------------------------------------------------------
# TIER 1: CLOUD INFRASTRUCTURE (TLS ENABLED)
# ---------------------------------------------------------
echo "☁️  Booting Cloud SuperLink..."
mkdir -p "$LOG_DIR/state/cloud"
FLWR_HOME="$LOG_DIR/state/cloud" flower-superlink \
    --serverappio-api-address $BROKER_IP:$CLOUD_SA \
    --fleet-api-address $BROKER_IP:$CLOUD_FL \
    --control-api-address $BROKER_IP:$CLOUD_CTRL \
    --ssl-certfile "$CERTS_DIR/cloud_server/certificates.pem" \
    --ssl-keyfile "$CERTS_DIR/cloud_server/private-key.pem" \
    --ssl-ca-certfile "$CERTS_DIR/cloud_ca/ca.crt" > "$LOG_DIR/system/cloud_superlink.log" 2>&1 &
PIDS+=($!)

wait_for_port $CLOUD_FL

# ---------------------------------------------------------
# TIER 2: FOG INFRASTRUCTURE & NGINX
# ---------------------------------------------------------
echo "🛡️  Booting NGINX mTLS Bouncer & Sidecars..."
nginx -g 'daemon off;' -c "$NGINX_CONF" > "$LOG_DIR/system/nginx_daemon.log" 2>&1 &
PIDS+=($!)

wait_for_port $((FOG_FL_BASE + 1))

# Initialize intermediate Fog aggregation nodes
for i in $(seq 1 $NUM_FOGS); do
    FOG_SA=$((FOG_SA_BASE + i))
    FOG_FL=$((FOG_FL_BASE + i))
    FOG_INTERNAL_FL=$((FOG_FL_BASE + 10000 + i))
    FOG_CTRL=$((FOG_CTRL_BASE + i))
    FOG_CLIENT_IO=$((FOG_CIO_BASE + i))

    echo "🌫️  Booting Fog $i Nodes..."
    
    # Boot Fog SuperNode (Acts as a client to the Cloud)
    mkdir -p "$LOG_DIR/state/fog_${i}_node"
    FLWR_HOME="$LOG_DIR/state/fog_${i}_node" flower-supernode \
        --superlink $BROKER_IP:$CLOUD_FL \
        --clientappio-api-address $BROKER_IP:$FOG_CLIENT_IO \
        --root-certificates "$CERTS_DIR/cloud_ca/ca.crt" \
        --node-config "fog_id=${i}" > "$LOG_DIR/system/fog${i}_supernode.log" 2>&1 &
    PIDS+=($!)    

    # Boot Fog SuperLink (Acts as a server for the Edges)
    mkdir -p "$LOG_DIR/state/fog_${i}_link"
    FLWR_HOME="$LOG_DIR/state/fog_${i}_link" flower-superlink \
        --insecure \
        --serverappio-api-address $BROKER_IP:$FOG_SA \
        --fleet-api-address $BROKER_IP:$FOG_INTERNAL_FL \
        --control-api-address $BROKER_IP:$FOG_CTRL > "$LOG_DIR/system/fog${i}_superlink.log" 2>&1 &
    PIDS+=($!)
    
    wait_for_port $FOG_INTERNAL_FL
done

# ---------------------------------------------------------
# TIER 3: EDGE INFRASTRUCTURE
# ---------------------------------------------------------
echo "📱 Booting Edge Nodes..."
for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        echo "    📱 Starting $CURRENT_EDGES Edge Agents for Fog $i..."
        # Initialize Edge nodes and bind them to their respective NGINX proxy ports
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_CLIENT_IO=$((EDGE_CIO_BASE + (i * 100) + j))
            EDGE_PROXY_PORT=$((FOG_FL_BASE + 20000 + (i * 100) + j))
            
            mkdir -p "$LOG_DIR/state/edge_${i}_${j}"
            FLWR_HOME="$LOG_DIR/state/edge_${i}_${j}" flower-supernode \
                --insecure \
                --superlink 127.0.0.1:$EDGE_PROXY_PORT \
                --clientappio-api-address $BROKER_IP:$EDGE_CLIENT_IO \
                --node-config "fog_num=${i} partition-id=${j}" > "$LOG_DIR/system/edge${i}_${j}_supernode.log" 2>&1 &
            PIDS+=($!)
        done
    else
        echo "    ⚠️ No Edge SuperNodes assigned to Fog $i."
    fi
done

echo "================================================="
echo " 4. SYSTEM ARCHITECTURE & PORT SUMMARY           "
echo "================================================="
echo ""
echo "☁️  [CLOUD] SuperLink (TLS Active)"
echo "    ├─ Fleet API:   $BROKER_IP:$CLOUD_FL  <-- (Fogs connect here)"
echo "    ├─ Control API: $BROKER_IP:$CLOUD_CTRL  <-- (flwr run . cloud)"
echo "    └─ ServerAppIO: $BROKER_IP:$CLOUD_SA"
echo "    │"

# Output visual representation of the dynamically generated topology
for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    
    FOG_SA=$((FOG_SA_BASE + i))
    FOG_FL=$((FOG_FL_BASE + i))
    FOG_INTERNAL_FL=$((FOG_FL_BASE + 10000 + i))
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
    echo "    $SPACER └─ [FOG $i] NGINX mTLS Proxy (Listening on $FOG_FL) --> Proxies to $FOG_INTERNAL_FL"
    echo "    $SPACER     ├─ Hidden Fleet API: $BROKER_IP:$FOG_INTERNAL_FL  <-- (Proxy Destination)"
    echo "    $SPACER     ├─ Fleet API:   $BROKER_IP:$FOG_FL  <-- (Edges connect here via Sidecar)"
    echo "    $SPACER     ├─ Control API: $BROKER_IP:$FOG_CTRL  <-- (flwr run . fog$i)"
    echo "    $SPACER     └─ ServerAppIO: $BROKER_IP:$FOG_SA"
    
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_CLIENT_IO=$((EDGE_CIO_BASE + (i * 100) + j))
            EDGE_PROXY_PORT=$((FOG_FL_BASE + 20000 + (i * 100) + j))
            if [ "$j" -eq "$CURRENT_EDGES" ]; then
                EDGE_PREFIX="└──"
            else
                EDGE_PREFIX="├──"
            fi
            echo "    $SPACER          $EDGE_PREFIX 📱 [EDGE ${i}_${j}] SuperNode connects to Sidecar on $EDGE_PROXY_PORT"
        done
    else
        echo "    $SPACER          └── (No Edge Nodes Assigned)"
    fi
    
    if [ "$i" -ne "$NUM_FOGS" ]; then
        echo "    │"
    fi
done

echo ""
echo "================================================="
echo " 5. DEPLOYING APPLICATIONS TO NODES              "
echo "================================================="

# Dispatch the ServerApp logic to each intermediate Fog layer
for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    SAFE_MIN_CLIENTS=$(( CURRENT_EDGES > 0 ? CURRENT_EDGES : 1 ))
    
    echo "Shipping FAB to Fog $i (Expecting $CURRENT_EDGES edges)..."
    flwr run . fog${i} --run-config "tier=\"fog\" min-clients=${SAFE_MIN_CLIENTS} fog_id=\"fog_${i}\"" --stream > "$LOG_DIR/system/run_fog${i}.log" 2>&1 &
    PIDS+=($!)

    # Stagger deployments to avoid CPU race conditions
    echo "  ⏳ Cooling down Fog $i stack..."
    sleep 3
done

sleep 2

# Dispatch the overarching aggregation logic to the central Cloud layer
echo "Shipping FAB to Cloud ..."
flwr run . cloud --run-config "tier=\"cloud\" min-clients=${NUM_FOGS}" --stream > "$LOG_DIR/system/run_cloud.log" 2>&1 &
PIDS+=($!)

echo ""
echo "✅ Global synchronization started in the background."
echo "Check the logs to watch progress: logs/nodes/ or logs/system"
echo "Press Ctrl+C to stop the network."
wait 2>/dev/null