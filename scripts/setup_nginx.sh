#!/bin/bash
echo "[DEBUG NGINX] ---------------------------------------"
echo "[DEBUG NGINX] Ξεκινάει το NGINX Setup Script"

if [ "$#" -lt 2 ]; then
    echo "[FATAL NGINX] Λείπουν arguments. Σταματάω." > /dev/tty
    exit 1
fi

NUM_FOGS=$1
EDGES_ARRAY=($2)
BROKER_IP=${3:-127.0.0.1}
# ΠΛΕΟΝ ΤΟ ΠΑΙΡΝΕΙ ΔΥΝΑΜΙΚΑ ΑΠΟ ΤΟ BOOT SCRIPT ΓΙΑ ΝΑ ΜΗΝ ΥΠΑΡΧΕΙ ΑΣΥΜΦΩΝΙΑ
FOG_FL_BASE=${4:-9290}

echo "[DEBUG NGINX] Arguments: NUM_FOGS=$1 | EDGES_ARRAY=$2 | BROKER_IP=$3 | FOG_FL_BASE=$FOG_FL_BASE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CERTS_DIR="$PROJECT_ROOT/src/network/certs"
LOG_DIR="$PROJECT_ROOT/logs/system"
NGINX_CONF="$PROJECT_ROOT/src/network/nginx.conf"

echo "[DEBUG NGINX] Δημιουργία NGINX Conf στο: $NGINX_CONF"

cat <<EOF > "$NGINX_CONF"
worker_processes auto;
pid $LOG_DIR/nginx.pid;

events {
    worker_connections 1024;
}

http {
    access_log $LOG_DIR/nginx_access.log;
    error_log $LOG_DIR/nginx_error.log debug;

    client_max_body_size 50M;
    grpc_read_timeout 1d;
    grpc_send_timeout 1d;

EOF

for i in $(seq 1 "$NUM_FOGS"); do
    FOG_FL=$((FOG_FL_BASE + i))
    FOG_INTERNAL_FL=$((FOG_FL_BASE + 10000 + i))
    echo "[DEBUG NGINX] Γράφω Fog $i - Public Port: $FOG_FL -> Internal: $FOG_INTERNAL_FL"

    cat <<EOF >> "$NGINX_CONF"
    server {
        listen $FOG_FL ssl;
        http2 on;
        server_name $BROKER_IP localhost;

        ssl_certificate $CERTS_DIR/fog_${i}/nginx.crt;
        ssl_certificate_key $CERTS_DIR/fog_${i}/nginx.key;

        ssl_client_certificate $CERTS_DIR/edge_ca/ca.crt;
        ssl_verify_client on;
        ssl_protocols TLSv1.3;

        location / {
            grpc_pass grpc://$BROKER_IP:$FOG_INTERNAL_FL;
        }
    }
EOF

    CURRENT_EDGES=${EDGES_ARRAY[$((i-1))]:-0}
    echo "[DEBUG NGINX] O Fog $i έχει $CURRENT_EDGES Edge Nodes"
    
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_PROXY_PORT=$((FOG_FL_BASE + 20000 + (i * 100) + j))
            echo "[DEBUG NGINX] Γράφω Edge ${i}_${j} Sidecar στο Port: $EDGE_PROXY_PORT"
            
            cat <<EOF >> "$NGINX_CONF"
    server {
        listen 127.0.0.1:$EDGE_PROXY_PORT;
        http2 on;
        server_name localhost;

        location / {
            grpc_pass grpcs://$BROKER_IP:$FOG_FL;
            
            grpc_ssl_certificate $CERTS_DIR/edge_${i}_${j}/client.crt;
            grpc_ssl_certificate_key $CERTS_DIR/edge_${i}_${j}/client.key;
            
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
echo "[DEBUG NGINX] Το NGINX Setup ολοκληρώθηκε."
exit 0