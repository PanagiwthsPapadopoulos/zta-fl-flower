#!/bin/bash
# =========================================================
# generate_compose.sh
# Generates the docker-compose.yml file dynamically based
# on the extracted topology.
# =========================================================

INSECURE_MODE=$1
HOST_TZ=$2
EDGES_ARRAY=($3)

# Global Mount Variables 
SHARED_DATA_MOUNT="./data"
SHARED_CONFIG_MOUNT="./config"
SHARED_RESULTS_MOUNT="./results"
CLOUD_LOG_MOUNT="./logs/nodes/cloud"

cat <<EOF> "$COMPOSE_FILE"
name: "zta-fl"

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
cat <<EOF>> "$COMPOSE_FILE"
      - "--ssl-certfile=/app/certs/cloud_server/certificates.pem"
      - "--ssl-keyfile=/app/certs/cloud_server/private-key.pem"
      - "--ssl-ca-certfile=/app/certs/cloud_ca/ca.crt"
    volumes:
      - ${CLOUD_LOG_MOUNT}:/app/logs
      - ${SHARED_DATA_MOUNT}:/app/data
      - "$CERTS_DIR:/app/certs:ro"
EOF
else
cat <<EOF>> "$COMPOSE_FILE"
      - "--insecure"
    volumes:
      - ${CLOUD_LOG_MOUNT}:/app/logs
      - ${SHARED_DATA_MOUNT}:/app/data
EOF
fi

cat <<EOF>> "$COMPOSE_FILE"
    networks: [flwr-network]
    ports: 
      - "${CLOUD_CTRL}:${CLOUD_CTRL}"
    healthcheck:
      test: ["CMD-SHELL", "python3 -c \"import socket; socket.create_connection(('127.0.0.1', ${CLOUD_CTRL}))\" || exit 1"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 5s

  cloud-serverapp:
    image: panagiotispapadopoulos/zta-cloud-node:latest
    environment: 
      - TZ=${HOST_TZ}
    command:
      - "--insecure"
      - "--plugin-type"
      - "serverapp"
      - "--appio-api-address"
      - "cloud-superlink:${CLOUD_SA}"
    networks: [flwr-network]
    depends_on:
      cloud-superlink:
        condition: service_healthy
    volumes:
      - ${CLOUD_LOG_MOUNT}:/app/logs
      - ${SHARED_DATA_MOUNT}:/app/data
      - ${SHARED_RESULTS_MOUNT}:/app/results
      - ${SHARED_CONFIG_MOUNT}:/app/config:ro
EOF

if [ "$INSECURE_MODE" = false ]; then
    cat <<EOF>> "$COMPOSE_FILE"

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
    CURRENT_EDGES=${EDGES_ARRAY[$((i-1))]:-0}
    FOG_LOG_MOUNT="./logs/nodes/fog_${i}"
    
    if [ "$INSECURE_MODE" = false ]; then
        FOG_INTERNAL_FL=$((FOG_FL_BASE + 10000 + i))
    else
        FOG_INTERNAL_FL=$FOG_FL
    fi

    cat <<EOF>> "$COMPOSE_FILE"

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
    cat <<EOF>> "$COMPOSE_FILE"
      - "--root-certificates=/app/certs/cloud_ca/ca.crt"
    volumes:
      - ${FOG_LOG_MOUNT}:/app/logs
      - ${SHARED_DATA_MOUNT}:/app/data:ro
      - "$CERTS_DIR:/app/certs:ro"
EOF
    else
    cat <<EOF>> "$COMPOSE_FILE"
      - "--insecure"
    volumes:
      - ${FOG_LOG_MOUNT}:/app/logs
      - ${SHARED_DATA_MOUNT}:/app/data:ro
EOF
    fi
    cat <<EOF>> "$COMPOSE_FILE"
    networks: [flwr-network]
    depends_on:
      cloud-superlink:
        condition: service_healthy

  fog-${i}-clientapp:
    image: panagiotispapadopoulos/zta-cloud-node:latest
    environment: [TZ=${HOST_TZ}, FOG_SERVER_HOST=fog-${i}-serverapp, IPC_PORT=${FOG_CLIENT_IO}]
    command:
      - "--insecure"
      - "--plugin-type"
      - "clientapp"
      - "--appio-api-address"
      - "fog-${i}-supernode:${FOG_CLIENT_IO}"
    networks: [flwr-network]
    depends_on: [fog-${i}-supernode]
    volumes:
      - ${FOG_LOG_MOUNT}:/app/logs
      - ${SHARED_DATA_MOUNT}:/app/data:ro
      - ${SHARED_CONFIG_MOUNT}:/app/config:ro

  fog-${i}-superlink:
    image: flwr/superlink:1.30.0
    environment:
      - TZ=${HOST_TZ}
    command:
      - "--isolation"
      - "process"
      - "--insecure"
      - "--serverappio-api-address"
      - "0.0.0.0:${FOG_SA}"
      - "--fleet-api-address"
      - "0.0.0.0:${FOG_INTERNAL_FL}"
      - "--control-api-address"
      - "0.0.0.0:${FOG_CTRL}"
    networks: [flwr-network]
    ports: 
      - "${FOG_CTRL}:${FOG_CTRL}"
    healthcheck:
      test: ["CMD-SHELL", "python3 -c \"import socket; socket.create_connection(('127.0.0.1', ${FOG_CTRL}))\" || exit 1"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 5s
    volumes:
      - ${FOG_LOG_MOUNT}:/app/logs
      - ${SHARED_DATA_MOUNT}:/app/data:ro

  fog-${i}-serverapp:
    image: panagiotispapadopoulos/zta-cloud-node:latest
    environment: [TZ=${HOST_TZ}, IPC_PORT=${FOG_SA}]
    command:
      - "--insecure"
      - "--plugin-type"
      - "serverapp"
      - "--appio-api-address"
      - "fog-${i}-superlink:${FOG_SA}"
    networks: [flwr-network]
    depends_on:
      fog-${i}-superlink:
        condition: service_healthy
    volumes:
      - ${FOG_LOG_MOUNT}:/app/logs
      - ${SHARED_DATA_MOUNT}:/app/data:ro
      - ${SHARED_CONFIG_MOUNT}:/app/config:ro
      - ./runtime/tpm_state:/app/runtime/tpm_state:rw
      - ${SHARED_RESULTS_MOUNT}:/app/results
EOF

    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_CLIENT_IO=$((EDGE_CIO_BASE + (i * 100) + j))
            EDGE_PROXY_PORT=$((FOG_FL_BASE + 20000 + (i * 100) + j))
            EDGE_LOG_MOUNT="./logs/nodes/edge_${i}_${j}"
            
            if [ "$INSECURE_MODE" = false ]; then
                EDGE_UPLINK="nginx-proxy:${EDGE_PROXY_PORT}"
            else
                EDGE_UPLINK="fog-${i}-superlink:${FOG_INTERNAL_FL}"
            fi

            cat <<EOF>> "$COMPOSE_FILE"

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
      - "--insecure"
      - "--superlink"
      - "${EDGE_UPLINK}"
      - "--clientappio-api-address"
      - "0.0.0.0:${EDGE_CLIENT_IO}"
      - "--node-config"
      - "fog_num=${i} partition-id=${j}"
    networks: [flwr-network]
    depends_on:
      fog-${i}-superlink:
        condition: service_healthy
    volumes:
      - ${EDGE_LOG_MOUNT}:/app/logs
      - ${SHARED_DATA_MOUNT}:/app/data:ro

  edge-${i}-${j}-clientapp:
    image: panagiotispapadopoulos/zta-edge-node:latest
    environment: 
      - TZ=${HOST_TZ}
      - TPM2TOOLS_TCTI=swtpm:port=2321
    command:
      - "--insecure"
      - "--plugin-type"
      - "clientapp"
      - "--appio-api-address"
      - "edge-${i}-${j}-supernode:${EDGE_CLIENT_IO}"
    networks: [flwr-network]
    depends_on: [edge-${i}-${j}-supernode]
    volumes:
      - ${EDGE_LOG_MOUNT}:/app/logs
      - ${SHARED_DATA_MOUNT}:/app/data:ro
      - ./runtime/tpm_state/edge_${i}_${j}:/app/runtime/tpm_state/edge_${i}_${j}
      - ${SHARED_CONFIG_MOUNT}:/app/config:ro
EOF
        done
    fi
done