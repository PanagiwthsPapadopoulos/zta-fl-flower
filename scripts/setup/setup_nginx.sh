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
#   $5 (is_docker)     : Boolean (Optional). Set to 'true' to generate internal Docker paths. Defaults to false.
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
IS_DOCKER=${5:-false}

echo "[DEBUG NGINX] Arguments: FOGS=$1 | EDGES=$2 | IP=$3 | BASE=$4 | DOCKER=$5"

# Resolve absolute paths for Nginx configuration generation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CERTS_DIR="$PROJECT_ROOT/runtime/certs"
LOG_DIR="$PROJECT_ROOT/logs/system"
export NGINX_CONF="$PROJECT_ROOT/runtime/infra/nginx.conf"

# =========================================================
# DYNAMIC PATH & BINDING RESOLUTION
# =========================================================
# Inherit CLOUD_FL from the environment (set by parse_topology.py), default to 9002
CLOUD_FL=${CLOUD_FL:-9002}

if [ "$IS_DOCKER" = "true" ]; then
    CONF_CERTS_DIR="/etc/nginx/certs"
    CONF_PID="/var/run/nginx.pid"
    # In Alpine Docker, NGINX symlinks these to stdout/stderr automatically
    CONF_ACCESS_LOG="/var/log/nginx/access.log"
    CONF_ERROR_LOG="/var/log/nginx/error.log"
    # Docker sidecars must bind to 0.0.0.0 to accept traffic from Edge containers
    SIDECAR_BIND=""
    CLOUD_UPSTREAM="cloud-superlink"
else
    CONF_CERTS_DIR="$CERTS_DIR"
    CONF_PID="$LOG_DIR/nginx.pid"
    CONF_ACCESS_LOG="$LOG_DIR/nginx_access.log"
    CONF_ERROR_LOG="$LOG_DIR/nginx_error.log"
    # Local Mac sidecars bind securely to localhost
    SIDECAR_BIND="127.0.0.1:"
    CLOUD_UPSTREAM="$BROKER_IP"
fi

echo "[DEBUG NGINX] Creating NGINX Conf at: $NGINX_CONF"

# Initialize global Nginx event and HTTP block configurations
cat <<EOF > "$NGINX_CONF"
worker_processes auto;
pid $CONF_PID;

events {
    worker_connections 1024;
}

http {
    access_log $CONF_ACCESS_LOG;
    error_log $CONF_ERROR_LOG debug;

    client_max_body_size 50M;
    grpc_read_timeout 1d;
    grpc_send_timeout 1d;

EOF

# =========================================================
# CLOUD TIER: STANDARD TLS TERMINATION
# =========================================================
echo "[DEBUG NGINX] Writing Cloud Proxy - Public Port: $CLOUD_FL -> Internal: 1${CLOUD_FL}"

cat <<EOF >> "$NGINX_CONF"
    server {
        listen $CLOUD_FL ssl;
        http2 on;
        server_name $BROKER_IP localhost cloud-superlink;

        # Standard TLS Server Identity
        ssl_certificate $CONF_CERTS_DIR/cloud_server/certificates.pem;
        ssl_certificate_key $CONF_CERTS_DIR/cloud_server/private-key.pem;
        ssl_protocols TLSv1.3;

        location / {
            # Route authenticated traffic to the insecure internal Cloud server
            grpc_pass grpc://$CLOUD_UPSTREAM:1${CLOUD_FL};
        }
    }
EOF

# Generate server blocks mapping Edge traffic to respective Fog backend ports
for i in $(seq 1 "$NUM_FOGS"); do
    FOG_FL=$((FOG_FL_BASE + i))
    FOG_INTERNAL_FL=$((FOG_FL_BASE + 10000 + i))
    
    # =========================================================
    # DYNAMIC UPSTREAM ROUTING RESOLUTION
    # =========================================================
    if [ "$IS_DOCKER" = "true" ]; then
        # NGINX must route to the Docker container name, not 127.0.0.1
        FOG_UPSTREAM="fog-${i}-superlink"
        # The NGINX sidecar proxies to the Fog port hosted on its OWN container
        SIDECAR_UPSTREAM="127.0.0.1"
    else
        FOG_UPSTREAM="$BROKER_IP"
        SIDECAR_UPSTREAM="$BROKER_IP"
    fi

    echo "[DEBUG NGINX] Writing Fog $i - Public Port: $FOG_FL -> Internal: $FOG_INTERNAL_FL"

    # =========================================================
    # FOG OUTBOUND TIER: STANDARD TLS SIDECAR (TO CLOUD)
    # =========================================================
    CLOUD_PROXY_PORT=$((CLOUD_FL + 20000 + i))
    
    echo "[DEBUG NGINX] Writing Fog $i Outbound Sidecar - Local Port: $CLOUD_PROXY_PORT -> Secure Cloud: $CLOUD_FL"
    
    cat <<EOF >> "$NGINX_CONF"
    server {
        listen ${SIDECAR_BIND}$CLOUD_PROXY_PORT;
        http2 on;
        server_name localhost;

        location / {
            # Upstream target: Secure Cloud port
            grpc_pass grpcs://$SIDECAR_UPSTREAM:$CLOUD_FL;
            
            # Verify the Cloud server against the Cloud CA (Standard TLS)
            # Notice: No client certificate is provided here.
            grpc_ssl_trusted_certificate $CONF_CERTS_DIR/cloud_ca/ca.crt;
            grpc_ssl_verify on;
            grpc_ssl_server_name on;
            grpc_ssl_name cloud-superlink;
        }
    }
EOF

    # Fog reverse proxy server block for mTLS termination
    cat <<EOF >> "$NGINX_CONF"
    server {
        listen $FOG_FL ssl;
        http2 on;
        server_name $BROKER_IP localhost;

        # Present identity to connecting Edge nodes
        ssl_certificate $CONF_CERTS_DIR/fog_${i}/nginx.crt;
        ssl_certificate_key $CONF_CERTS_DIR/fog_${i}/nginx.key;

        # Enforce mutual TLS authentication against Edge root CA
        ssl_client_certificate $CONF_CERTS_DIR/edge_ca/ca.crt;
        ssl_verify_client on;
        ssl_protocols TLSv1.3;

        location / {
            # Route authenticated traffic to the insecure internal Flower server
            grpc_pass grpc://$FOG_UPSTREAM:$FOG_INTERNAL_FL;
        }
    }
EOF

    CURRENT_EDGES=${EDGES_ARRAY[$((i-1))]:-0}
    
    # Generate individualized local sidecars for each Edge node to handle outbound mTLS
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_PROXY_PORT=$((FOG_FL_BASE + 20000 + (i * 100) + j))
            
            cat <<EOF >> "$NGINX_CONF"
    server {
        listen ${SIDECAR_BIND}$EDGE_PROXY_PORT;
        http2 on;
        server_name localhost;

        location / {
            # Upstream target: Secure Fog port
            grpc_pass grpcs://$SIDECAR_UPSTREAM:$FOG_FL;
            
            # Mount Edge client certificate for outbound authentication
            grpc_ssl_certificate $CONF_CERTS_DIR/edge_${i}_${j}/client.crt;
            grpc_ssl_certificate_key $CONF_CERTS_DIR/edge_${i}_${j}/client.key;
            
            # Verify the Fog server against the Edge CA
            grpc_ssl_trusted_certificate $CONF_CERTS_DIR/edge_ca/ca.crt;
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