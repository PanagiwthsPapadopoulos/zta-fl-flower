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
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
DOCKERFILE="$PROJECT_ROOT/superexec.Dockerfile"
LOG_DIR="$PROJECT_ROOT/logs"
CERTS_DIR="$PROJECT_ROOT/src/network/certs"
NGINX_CONF="$PROJECT_ROOT/src/network/nginx.conf"
NGINX_DOCKER_CONF="$PROJECT_ROOT/src/network/nginx_docker.conf"

PIDS=()

# =========================================================
# UTILITIES: CLEANUP & TIMEZONE DETECTION
# =========================================================
cleanup() {
    echo -e "\n🛑 Caught Ctrl+C! Shutting down the Docker Engine..."
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null
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
echo " 🔍 READING TOPOLOGY FROM pyproject.toml         "
echo "================================================="
CONFIG_VARS=$(python3 - <<EOF
import re, ast
try:
    with open('$PROJECT_ROOT/pyproject.toml', 'r') as f: content = f.read()
    def get_val(key, default):
        m = re.search(fr'{key}\s*=\s*(["0-9\.]+)', content)
        return m.group(1).replace('"', '') if m else default

    print(f"CLOUD_SA={get_val('cloud_sa_port', '9091')}")
    print(f"CLOUD_FL={get_val('cloud_fl_port', '9092')}")
    print(f"CLOUD_CTRL={get_val('cloud_ctrl_port', '9093')}")
    print(f"FOG_SA_BASE={get_val('fog_sa_base', '9190')}")
    print(f"FOG_FL_BASE={get_val('fog_fl_base', '9290')}")
    print(f"FOG_CTRL_BASE={get_val('fog_ctrl_base', '9390')}")
    print(f"FOG_CIO_BASE={get_val('fog_client_io_base', '9490')}")
    print(f"EDGE_CIO_BASE={get_val('edge_client_io_base', '9500')}")

    num_fogs = int(re.search(r'num_fogs\s*=\s*(\d+)', content).group(1))
    uniform = int(re.search(r'uniform_edges_per_fog\s*=\s*(\d+)', content).group(1))
    custom_match = re.search(r'custom_fog_topology\s*=\s*"(\[.*?\])"', content)
    custom_top = ast.literal_eval(custom_match.group(1)) if custom_match else []
    edges_array = custom_top[:num_fogs] if custom_top and len(custom_top) >= num_fogs else [uniform] * num_fogs
    
    print(f"NUM_FOGS={num_fogs}")
    print(f"EDGES_PER_FOG_ARRAY=({' '.join(map(str, edges_array))})")
except Exception as e:
    print(f'echo "Error: {e}"; exit 1')
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

# =========================================================
# 2. IMAGE GENERATION (DOCKERFILE)
# =========================================================
docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null
mkdir -p "$LOG_DIR/system" "$LOG_DIR/nodes" "$PROJECT_ROOT/data" "$PROJECT_ROOT/.pip-cache"

cat <<EOF > "$DOCKERFILE"
FROM flwr/superexec:1.30.0
USER root
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml .
RUN /python/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml && /python/venv/bin/pip install -U .
ENTRYPOINT ["flower-superexec"]
EOF

if ! docker image inspect local-flower-node:latest >/dev/null 2>&1; then
    echo "⏳ Compiling Master Execution Image..."
    docker build -t local-flower-node:latest -f "$DOCKERFILE" .
fi

# =========================================================
# 3. DOCKER COMPOSE TOPOLOGY GENERATION
# =========================================================

cat <<EOF > "$COMPOSE_FILE"
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
    image: local-flower-node:latest
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
    image: local-flower-node:latest
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
    image: local-flower-node:latest
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
  # TIER 3: EDGE ${j} FOR FOG ${i}
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
    image: local-flower-node:latest
    environment: [TZ=${HOST_TZ}]
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
EOF
        done
    fi
done

# =================================================
#  4. BOOTING DOCKER FEDERATION                    
# =================================================
cd "$PROJECT_ROOT" || exit 1
docker compose up -d
docker compose logs -f > "$LOG_DIR/system/docker_mesh.log" 2>&1 &
PIDS+=($!)

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