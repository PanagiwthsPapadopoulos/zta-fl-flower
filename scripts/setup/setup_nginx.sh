#!/bin/bash
# =========================================================
# setup_nginx.sh
# Generates the NGINX configuration file for mTLS reverse proxying.
# It maps public-facing Fog ports to internal insecure Flower SuperNode ports,
# and provisions local sidecars for Edge nodes to route outbound traffic securely.
#
# ARGUMENTS:
#   $1 (num_fogs)      : Integer. The total number of Fog nodes.
#   $2 (edges_array)   : String. Space-separated list of edges per fog (e.g., "2 0 4").
#   $3 (broker_ip)     : String (Optional). IP address of the Cloud server. Defaults to 127.0.0.1.
#   $4 (fog_fl_base)   : Integer (Optional). Base port for Fog Fleet APIs. Defaults to 9290.
# =========================================================

echo "[DEBUG NGINX] ---------------------------------------"
echo "[DEBUG NGINX] Starting NGINX Setup Script"

# Validate required positional arguments
if [ "$#" -lt 2 ]; then
    echo "[FATAL NGINX] Missing arguments. Aborting." > /dev/tty
    exit 1
fi

NUM_FOGS=$1
EDGES_ARRAY=($2)
BROKER_IP=${3:-127.0.0.1}

# Dynamically retrieved from the boot script to prevent port mismatches
FOG_FL_BASE=${4:-9290}

echo "[DEBUG NGINX] Arguments: NUM_FOGS=$1 | EDGES_ARRAY=$2 | BROKER_IP=$3 | FOG_FL_BASE=$FOG_FL_BASE"

# Resolve absolute paths for Nginx configuration generation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CERTS_DIR="$PROJECT_ROOT/src/network/certs"
LOG_DIR="$PROJECT_ROOT/logs/system"
NGINX_CONF="$PROJECT_ROOT/src/network/nginx.conf"

echo "[DEBUG NGINX] Creating NGINX Conf at: $NGINX_CONF"

# Initialize global Nginx event and HTTP block configurations
cat <<EOF > "$NGINX_CONF"
worker_processes auto;
pid $LOG_DIR/nginx.pid;

events {
    worker_connections 1024;
}

http {
    access_log $LOG_DIR/nginx_access.log;
    error_log $LOG_DIR/nginx_error.log debug;

    # Allow large ML model weight transfers and prevent gRPC session timeouts
    client_max_body_size 50M;
    grpc_read_timeout 1d;
    grpc_send_timeout 1d;

EOF

# Generate server blocks mapping Edge traffic to respective Fog backend ports
for i in $(seq 1 "$NUM_FOGS"); do
    FOG_FL=$((FOG_FL_BASE + i))
    FOG_INTERNAL_FL=$((FOG_FL_BASE + 10000 + i))
    echo "[DEBUG NGINX] Writing Fog $i - Public Port: $FOG_FL -> Internal: $FOG_INTERNAL_FL"

    # Fog reverse proxy server block for mTLS termination
    cat <<EOF >> "$NGINX_CONF"
    server {
        listen $FOG_FL ssl;
        http2 on;
        server_name $BROKER_IP localhost;

        # Present identity to connecting Edge nodes
        ssl_certificate $CERTS_DIR/fog_${i}/nginx.crt;
        ssl_certificate_key $CERTS_DIR/fog_${i}/nginx.key;

        # Enforce mutual TLS authentication against Edge root CA
        ssl_client_certificate $CERTS_DIR/edge_ca/ca.crt;
        ssl_verify_client on;
        ssl_protocols TLSv1.3;

        location / {
            # Route authenticated traffic to the insecure internal Flower server
            grpc_pass grpc://$BROKER_IP:$FOG_INTERNAL_FL;
        }
    }
EOF

    CURRENT_EDGES=${EDGES_ARRAY[$((i-1))]:-0}
    echo "[DEBUG NGINX] Fog $i has $CURRENT_EDGES Edge Nodes"
    
    # Generate individualized local sidecars for each Edge node to handle outbound mTLS
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_PROXY_PORT=$((FOG_FL_BASE + 20000 + (i * 100) + j))
            echo "[DEBUG NGINX] Writing Edge ${i}_${j} Sidecar at Port: $EDGE_PROXY_PORT"
            
            cat <<EOF >> "$NGINX_CONF"
    server {
        listen 127.0.0.1:$EDGE_PROXY_PORT;
        http2 on;
        server_name localhost;

        location / {
            # Upstream target: Secure Fog port
            grpc_pass grpcs://$BROKER_IP:$FOG_FL;
            
            # Mount Edge client certificate for outbound authentication
            grpc_ssl_certificate $CERTS_DIR/edge_${i}_${j}/client.crt;
            grpc_ssl_certificate_key $CERTS_DIR/edge_${i}_${j}/client.key;
            
            # Verify the Fog server against the Edge CA
            grpc_ssl_trusted_certificate $CERTS_DIR/edge_ca/ca.crt;
            grpc_ssl_verify on;
            grpc_ssl_server_name on;
            grpc_ssl_name fog-${i}-nginx;
        }
    }
EOF
        done
    fi
done

echo "}" >> "$NGINX_CONF"
echo "[DEBUG NGINX] NGINX Setup completed."
exit 0