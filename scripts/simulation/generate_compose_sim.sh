#!/bin/bash

# =========================================================
# generate_compose_sim.sh
# =========================================================

HOST_TZ=$1
EDGES_ARRAY=($2)

SHARED_DATA_MOUNT="./data"
SHARED_CONFIG_MOUNT="./config"
SHARED_RESULTS_MOUNT="./results"
BATCH_ENV=""
if [ "${ZTA_BATCH_MANAGED:-0}" = "1" ]; then
    BATCH_ENV="      - FLWR_HOME=/app/runtime/batch/flower
      - RAY_TMPDIR=/app/runtime/batch/ray"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_AVAILABLE=true
    GPU_CONFIG="    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]"
else
    GPU_AVAILABLE=false
    GPU_CONFIG=""
fi

mkdir -p "$PROJECT_ROOT/logs/nodes/cloud"
mkdir -p "$PROJECT_ROOT/runtime/tpm_state"

cat <<EOF > "$COMPOSE_FILE"
name: "zta-fl"

networks:
  flwr-network:
    driver: bridge

services:
  cloud:
    image: panagiotispapadopoulos/zta-cloud-node:latest
    working_dir: /app
    init: true
    container_name: sim-cloud
    hostname: cloud
    labels:
      zta.batch_attempt: "${ZTA_BATCH_ATTEMPT:-standalone}"
    environment:
      - TZ=${HOST_TZ}
      - TIER=cloud
      - ZTA_TIER=cloud
      - ZTA_GPU_AVAILABLE=${GPU_AVAILABLE}
      - NUM_FOGS=${NUM_FOGS}
      - PYTHONPATH=/app
${BATCH_ENV}
    volumes:
      - ./logs/nodes/cloud:/app/logs
      - ./logs/system:/app/logs/system
      - ${SHARED_DATA_MOUNT}:/app/data
      - ${SHARED_RESULTS_MOUNT}:/app/results
      - ${SHARED_CONFIG_MOUNT}:/app/config:ro
      - ./runtime/tpm_state:/app/runtime/tpm_state
      - ./runtime/batch:/app/runtime/batch
      - ./src:/app/src:ro
      - ./scripts:/app/scripts:ro
      - ./pyproject.toml:/app/pyproject.toml:ro
      - ./.gitignore:/app/.gitignore:ro
    networks: [flwr-network]
    entrypoint: []
    command: ["sleep", "infinity"]
${GPU_CONFIG}
EOF

for i in $(seq 1 $NUM_FOGS); do
    CURRENT_EDGES=${EDGES_ARRAY[$((i-1))]:-0}

    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            mkdir -p "$PROJECT_ROOT/runtime/tpm_state/edge_${i}_${j}"
            
            cat <<EOF >> "$COMPOSE_FILE"

  tpm-${i}-${j}:
    image: panagiotispapadopoulos/zta-edge-node:latest
    working_dir: /app
    init: true
    container_name: sim-tpm-${i}-${j}
    hostname: tpm-${i}-${j}
    labels:
      zta.batch_attempt: "${ZTA_BATCH_ATTEMPT:-standalone}"
    environment:
      - TZ=${HOST_TZ}
      - TPM2TOOLS_TCTI=swtpm:port=2321
      - PYTHONPATH=/app
    user: "0:0"
    volumes:
      - ./src:/app/src:ro
      - ./scripts:/app/scripts:ro
      - ${SHARED_CONFIG_MOUNT}:/app/config:ro
      - ./runtime/tpm_state/edge_${i}_${j}:/app/runtime/tpm_state/edge_${i}_${j}
    networks: [flwr-network]
    entrypoint: []
    command: sh -c "swtpm socket --tpmstate dir=/app/runtime/tpm_state/edge_${i}_${j} --tpm2 --server type=tcp,port=2321,bindaddr=0.0.0.0 --ctrl type=tcp,port=2322,bindaddr=0.0.0.0 --flags startup-clear --daemon ; i=0 ; until nc -z 127.0.0.1 2321 ; do i=\$\$((i+1)) ; [ \$\$i -ge 30 ] && break ; sleep 1 ; done ; export TPM2TOOLS_TCTI=swtpm:port=2321 ; tpm2_startup -c || true ; cd /app && PYTHONPATH=/app python3 -c 'import logging; from src.tier_edge.tpm_attestation import TPMAttestation; logging.basicConfig(level=logging.INFO); logger=logging.getLogger(\"Bootstrapper\"); tpm=TPMAttestation(logger=logger); tpm.generate_attestation_token(nonce=\"FACTORY_BOOT_NONCE\", software_label=\"[EDGE ${i}_${j}]\", round_num=0)' ; echo 'TPM ready' ; sleep infinity"
    restart: unless-stopped
EOF
        done
    fi
done
