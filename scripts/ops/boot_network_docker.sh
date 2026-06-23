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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"
LOG_DIR="$PROJECT_ROOT/logs"
CERTS_DIR="$PROJECT_ROOT/runtime/certs"
NGINX_CONF="$PROJECT_ROOT/runtime/nginx.conf"

# Dynamically determine a safe Compose project name based on the root directory
PROJECT_DIR_NAME=$(basename "$PROJECT_ROOT" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')
COMPOSE_PROJECT_NAME="${PROJECT_DIR_NAME:-flwr-federation}"

PIDS=()

# =========================================================
# UTILITIES: CLEANUP & TIMEZONE DETECTION
# =========================================================
cleanup() {
    echo -e "\n🛑 Caught Ctrl+C! Shutting down the Docker Engine..."
    docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" down --remove-orphans 2>/dev/null
    echo "✅ Teardown complete. Network is offline."
    exit 0
}
trap cleanup SIGINT SIGTERM

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
CONFIG_VARS=$(python3 - <<EOF
import re, ast

def get_yaml_val(filepath, key, default):
    """Safely extracts YAML values using regex to avoid external host OS dependencies."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        # Matches key: value, stripping out inline comments and quotes
        m = re.search(fr'^{key}:\s*(.+)$', content, re.MULTILINE)
        if m:
            return m.group(1).split('#')[0].strip().strip('"').strip("'")
    except:
        pass
    return default

net_conf = '$PROJECT_ROOT/config/network.yaml'

try:
    print(f"CLOUD_SA={get_yaml_val(net_conf, 'cloud_sa_port', '9001')}")
    print(f"CLOUD_FL={get_yaml_val(net_conf, 'cloud_fl_port', '9002')}")
    print(f"CLOUD_CTRL={get_yaml_val(net_conf, 'cloud_ctrl_port', '9003')}")
    print(f"FOG_SA_BASE={get_yaml_val(net_conf, 'fog_sa_base', '9100')}")
    print(f"FOG_FL_BASE={get_yaml_val(net_conf, 'fog_fl_base', '9200')}")
    print(f"FOG_CTRL_BASE={get_yaml_val(net_conf, 'fog_ctrl_base', '9300')}")
    print(f"FOG_CIO_BASE={get_yaml_val(net_conf, 'fog_client_io_base', '9400')}")
    print(f"EDGE_CIO_BASE={get_yaml_val(net_conf, 'edge_client_io_base', '10000')}")

    num_fogs = int(get_yaml_val(net_conf, 'num_fogs', '2'))
    uniform = int(get_yaml_val(net_conf, 'uniform_edges_per_fog', '2'))
    
    custom_top_str = get_yaml_val(net_conf, 'custom_fog_topology', '[]')
    custom_top = ast.literal_eval(custom_top_str) if custom_top_str else []
    
    edges_array = custom_top[:num_fogs] if custom_top and len(custom_top) >= num_fogs else [uniform] * num_fogs
    
    print(f"NUM_FOGS={num_fogs}")
    print(f"EDGES_PER_FOG_ARRAY=({' '.join(map(str, edges_array))})")
except Exception as e:
    print(f'echo "Error parsing topology: {e}"; exit 1')
EOF
)
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
# 2. IMAGE GENERATION
# =========================================================
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" down --remove-orphans 2>/dev/null
mkdir -p "$LOG_DIR/system" "$LOG_DIR/nodes" "$PROJECT_ROOT/data" "$PROJECT_ROOT/.pip-cache"

if ! docker image inspect zta-cloud-node:latest >/dev/null 2>&1; then
    echo "⏳ Compiling Cloud Execution Image (Optimized Base)..."
    docker build -t zta-cloud-node:latest -f docker/cloud.Dockerfile .
fi

if ! docker image inspect zta-edge-node:latest >/dev/null 2>&1; then
    echo "⏳ Compiling Edge Execution Image (TPM Enabled / Optimized Base)..."
    docker build -t zta-edge-node:latest -f docker/edge.Dockerfile .
fi

# =========================================================
# 3. DOCKER COMPOSE TOPOLOGY GENERATION
# =========================================================

cat <<EOF > "$COMPOSE_FILE"
name: ${COMPOSE_PROJECT_NAME}

networks:
  flwr-network:
    driver: bridge

services:
  # ---------------------------------------------------------
  # TIER 1: CLOUD INFRASTRUCTURE
  # ---------------------------------------------------------
  cloud-superlink:
    image: flwr/superlink:1.30.0
    command:
      - "--isolation"
      - "process"
      - "--serverappio-api-address"
      - "0.0.0.0:${CLOUD_SA}"
      - "--fleet-api-address"
      - "0.0.0.0:${CLOUD_FL}"
      - "--control-api-address"
      - "0.0.0.0:${CLOUD_CTRL}"
EOF

if [ "$INSECURE_MODE" = false ]; then
cat <<EOF >> "$COMPOSE_FILE"
      - "--ssl-certfile=/app/certs/cloud_server/certificates.pem"
      - "--ssl-keyfile=/app/certs/cloud_server/private-key.pem"
      - "--ssl-ca-certfile=/app/certs/cloud_ca/ca.crt"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - "$CERTS_DIR:/app/certs:ro"
EOF
else
cat <<EOF >> "$COMPOSE_FILE"
      - "--insecure"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
EOF
fi

cat <<EOF >> "$COMPOSE_FILE"
    networks: [flwr-network]
    ports: 
      - "${CLOUD_CTRL}:${CLOUD_CTRL}"

  cloud-serverapp:
    image: zta-cloud-node:latest
    environment: 
      - TZ=${HOST_TZ}
    command:
      - "--insecure" # Internal Docker ServerAppIo traffic is ALWAYS plaintext
      - "--plugin-type"
      - "serverapp"
      - "--appio-api-address"
      - "cloud-superlink:${CLOUD_SA}"
    networks: [flwr-network]
    depends_on: [cloud-superlink]
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./results:/app/results
      - ./config:/app/config:ro
EOF

if [ "$INSECURE_MODE" = false ]; then
    cat <<EOF >> "$COMPOSE_FILE"

  nginx-proxy:
    image: nginx:alpine
    volumes: 
      - "$NGINX_CONF:/etc/nginx/nginx.conf:ro"
      - "$CERTS_DIR:/etc/nginx/certs:ro"
    networks: [flwr-network]
    ports: ["9200-9300:9200-9300"]
EOF
fi

for i in $(seq 1 $NUM_FOGS); do
    FOG_SA=$((FOG_SA_BASE + i))
    FOG_FL=$((FOG_FL_BASE + i))
    FOG_CTRL=$((FOG_CTRL_BASE + i))
    FOG_CLIENT_IO=$((FOG_CIO_BASE + i))
    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    
    if [ "$INSECURE_MODE" = false ]; then
        FOG_INTERNAL_FL=$((FOG_FL_BASE + 10000 + i))
    else
        FOG_INTERNAL_FL=$FOG_FL
    fi

    cat <<EOF >> "$COMPOSE_FILE"

  # ---------------------------------------------------------
  # TIER 2: FOG ${i} INFRASTRUCTURE
  # ---------------------------------------------------------
  fog-${i}-supernode:
    image: flwr/supernode:1.30.0
    environment:
      - TZ=${HOST_TZ}
    command:
      - "--isolation"
      - "process"
      - "--superlink"
      - "cloud-superlink:${CLOUD_FL}"
      - "--clientappio-api-address"
      - "0.0.0.0:${FOG_CLIENT_IO}"
      - "--node-config"
      - "fog_id=${i}"
EOF
    if [ "$INSECURE_MODE" = false ]; then
    cat <<EOF >> "$COMPOSE_FILE"
      - "--root-certificates=/app/certs/cloud_ca/ca.crt" # Secure Uplink to Cloud
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - "$CERTS_DIR:/app/certs:ro"
EOF
    else
    cat <<EOF >> "$COMPOSE_FILE"
      - "--insecure"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
EOF
    fi
    cat <<EOF >> "$COMPOSE_FILE"
    networks: [flwr-network]
    depends_on: [cloud-superlink]

  fog-${i}-clientapp:
    image: zta-cloud-node:latest
    environment: [TZ=${HOST_TZ}, FOG_SERVER_HOST=fog-${i}-serverapp, IPC_PORT=${FOG_CLIENT_IO}]
    command:
      - "--insecure" # Internal Docker ClientAppIo traffic is ALWAYS plaintext
      - "--plugin-type"
      - "clientapp"
      - "--appio-api-address"
      - "fog-${i}-supernode:${FOG_CLIENT_IO}"
    networks: [flwr-network]
    depends_on: [fog-${i}-supernode]
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./config:/app/config:ro

  fog-${i}-superlink:
    image: flwr/superlink:1.30.0
    environment:
      - TZ=${HOST_TZ}
    command:
      - "--isolation"
      - "process"
      - "--insecure" # ALWAYS insecure because NGINX handles external TLS wrapper
      - "--serverappio-api-address"
      - "0.0.0.0:${FOG_SA}"
      - "--fleet-api-address"
      - "0.0.0.0:${FOG_INTERNAL_FL}"
      - "--control-api-address"
      - "0.0.0.0:${FOG_CTRL}"
    networks: [flwr-network]
    ports: 
      - "${FOG_CTRL}:${FOG_CTRL}"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data

  fog-${i}-serverapp:
    image: zta-cloud-node:latest
    environment: [TZ=${HOST_TZ}, IPC_PORT=${FOG_SA}]
    command:
      - "--insecure" # Internal Docker ServerAppIo traffic is ALWAYS plaintext
      - "--plugin-type"
      - "serverapp"
      - "--appio-api-address"
      - "fog-${i}-superlink:${FOG_SA}"
    networks: [flwr-network]
    depends_on: [fog-${i}-superlink]
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./config:/app/config:ro
      - ./runtime/tpm_state:/app/runtime/tpm_state:rw
      - ./runtime/gatekeeper:/app/runtime/gatekeeper
EOF

    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_CLIENT_IO=$((EDGE_CIO_BASE + (i * 100) + j))
            EDGE_PROXY_PORT=$((FOG_FL_BASE + 20000 + (i * 100) + j))
            
            if [ "$INSECURE_MODE" = false ]; then
                EDGE_UPLINK="nginx-proxy:${EDGE_PROXY_PORT}"
            else
                EDGE_UPLINK="fog-${i}-superlink:${FOG_INTERNAL_FL}"
            fi

            cat <<EOF >> "$COMPOSE_FILE"

  # ---------------------------------------------------------
  # TIER 3: EDGE ${j} FOR FOG ${i} (TPM ENABLED)
  # ---------------------------------------------------------
  edge-${i}-${j}-supernode:
    image: flwr/supernode:1.30.0
    environment:
      - TZ=${HOST_TZ}
    command:
      - "--isolation"
      - "process"
      - "--insecure" # ALWAYS insecure because it talks to the plaintext port of the local NGINX sidecar
      - "--superlink"
      - "${EDGE_UPLINK}"
      - "--clientappio-api-address"
      - "0.0.0.0:${EDGE_CLIENT_IO}"
      - "--node-config"
      - "fog_num=${i} partition-id=${j}"
    networks: [flwr-network]
    depends_on: [fog-${i}-superlink]
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data

  edge-${i}-${j}-clientapp:
    image: zta-edge-node:latest
    environment: 
      - TZ=${HOST_TZ}
      - TPM2TOOLS_TCTI=swtpm:port=2321
    command:
      - "--insecure" # Internal Docker ClientAppIo traffic is ALWAYS plaintext
      - "--plugin-type"
      - "clientapp"
      - "--appio-api-address"
      - "edge-${i}-${j}-supernode:${EDGE_CLIENT_IO}"
    networks: [flwr-network]
    depends_on: [edge-${i}-${j}-supernode]
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./runtime/tpm_state/edge_${i}_${j}:/app/runtime/tpm_state/edge_${i}_${j}
      - ./config:/app/config:ro
EOF
        done
    fi
done

# =================================================
#  4. BOOTING DOCKER FEDERATION                    
# =================================================
cd "$PROJECT_ROOT" || exit 1
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" up -d
docker compose -f "$COMPOSE_FILE" --project-directory "$PROJECT_ROOT" logs -f > "$LOG_DIR/system/docker_mesh.log" 2>&1 &
PIDS+=($!)

# =========================================================
# 4.5 OFFLINE ZERO-TRUST NETWORK PROVISIONING (COLLECTOR)
# =========================================================
echo "================================================="
echo " 🛡️  FACTORY PROVISIONING (COLLECTING STATES)     "
echo "================================================="
echo "⏳ Waiting 5 seconds for container TPM boot sequences to complete..."
sleep 5

# Export the project root so the Python script knows where to look
export PROJECT_ROOT="$PROJECT_ROOT"

# Execute the decoupled script
python3 "$PROJECT_ROOT/scripts/setup/collect_ledgers.py"

# =================================================
#  5. INJECTING GLOBAL CONFIGURATION (~/.flwr)     
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
wait