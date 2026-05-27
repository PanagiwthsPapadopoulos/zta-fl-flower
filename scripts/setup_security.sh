#!/bin/bash
# =========================================================
# setup_security.sh
# Generates static TLS certificates for the Cloud/Fog boundary
# and mTLS certificates for the NGINX/Edge boundary.
#
# ARGUMENTS:
#   $1 (num_fogs)             : Integer. The total number of Fog nodes.
#   $2 (edges_per_fog_string) : String. Space-separated list of edges per fog (e.g., "2 0 4").
#   $3 (broker_ip)            : String (Optional). IP address of the Cloud server. Defaults to 127.0.0.1.
#
# USAGE EXAMPLE:
#   ./setup_security.sh 3 "2 0 4" 192.168.1.100
# =========================================================

# 1. Enforce strict arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <num_fogs> <edges_per_fog_string> [broker_ip]" > /dev/tty
    exit 1
fi

NUM_FOGS=$1
read -r -a EDGES_ARRAY <<< "$2"
BROKER_IP=${3:-127.0.0.1}

# Path resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CERTS_DIR="$PROJECT_ROOT/src/network/certs"

# =========================================================
# 2. PROTECTION CHECK FOR STATIC IDENTITIES
# =========================================================
if [ -d "$CERTS_DIR" ] && [ "$(ls -A "$CERTS_DIR" 2>/dev/null)" ]; then
    # Route read strictly to /dev/tty so the prompt survives output capture
    echo "⚠️  Existing certificates found in $CERTS_DIR." > /dev/tty
    read -p "Do you want to wipe them and regenerate a new identity? (y/n) " -n 1 -r < /dev/tty
    echo > /dev/tty
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        # This string is caught by boot_network.sh
        echo "STATUS:KEPT" 
        exit 0
    fi
    rm -rf "$CERTS_DIR"
fi

mkdir -p "$CERTS_DIR/cloud_ca"
mkdir -p "$CERTS_DIR/cloud_server"
mkdir -p "$CERTS_DIR/edge_ca"

# =========================================================
# PHASE 1: CLOUD-FOG STANDARD TLS (FLOWER)
# =========================================================
openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
    -keyout "$CERTS_DIR/cloud_ca/ca.key" -out "$CERTS_DIR/cloud_ca/ca.crt" \
    -subj "/CN=ZTA-Cloud-Root-CA" 2>/dev/null

openssl req -nodes -newkey rsa:2048 \
    -keyout "$CERTS_DIR/cloud_server/private-key.pem" -out "$CERTS_DIR/cloud_server/server.csr" \
    -subj "/CN=cloud-superlink" 2>/dev/null
    
# Create isolated SAN extension for the Cloud Server
cat <<EOF > "$CERTS_DIR/cloud_server/san.ext"
subjectAltName = IP:$BROKER_IP,DNS:localhost,DNS:cloud-superlink
EOF
    
openssl x509 -req -in "$CERTS_DIR/cloud_server/server.csr" \
    -CA "$CERTS_DIR/cloud_ca/ca.crt" -CAkey "$CERTS_DIR/cloud_ca/ca.key" -CAcreateserial \
    -out "$CERTS_DIR/cloud_server/certificates.pem" -days 3650 -extfile "$CERTS_DIR/cloud_server/san.ext" 2>/dev/null

# =========================================================
# PHASE 2: EDGE-FOG MUTUAL TLS (NGINX)
# =========================================================
openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
    -keyout "$CERTS_DIR/edge_ca/ca.key" -out "$CERTS_DIR/edge_ca/ca.crt" \
    -subj "/CN=ZTA-Edge-Root-CA" 2>/dev/null

for i in $(seq 1 "$NUM_FOGS"); do
    # Fallback to 0 if the array is shorter than NUM_FOGS
    CURRENT_EDGES=${EDGES_ARRAY[$((i-1))]:-0}
    
    FOG_DIR="$CERTS_DIR/fog_${i}"
    mkdir -p "$FOG_DIR"
    
    # Generate the NGINX Server Certificate for this Fog Node
    openssl req -nodes -newkey rsa:2048 \
        -keyout "$FOG_DIR/nginx.key" -out "$FOG_DIR/nginx.csr" \
        -subj "/CN=fog-${i}-nginx" 2>/dev/null
        
    # Create isolated SAN extension for this specific Fog Nginx Sidecar
    cat <<EOF > "$FOG_DIR/san.ext"
subjectAltName = IP:$BROKER_IP,DNS:localhost,DNS:fog-${i}-nginx
EOF
        
    openssl x509 -req -in "$FOG_DIR/nginx.csr" \
        -CA "$CERTS_DIR/edge_ca/ca.crt" -CAkey "$CERTS_DIR/edge_ca/ca.key" -CAcreateserial \
        -out "$FOG_DIR/nginx.crt" -days 3650 -extfile "$FOG_DIR/san.ext" 2>/dev/null

    # Only attempt to generate Edge certificates if this Fog actually has Edges
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_DIR="$CERTS_DIR/edge_${i}_${j}"
            mkdir -p "$EDGE_DIR"
            
            openssl req -nodes -newkey rsa:2048 \
                -keyout "$EDGE_DIR/client.key" -out "$EDGE_DIR/client.csr" \
                -subj "/CN=edge-agent-${i}-${j}" 2>/dev/null
                
            openssl x509 -req -in "$EDGE_DIR/client.csr" \
                -CA "$CERTS_DIR/edge_ca/ca.crt" -CAkey "$CERTS_DIR/edge_ca/ca.key" -CAcreateserial \
                -out "$EDGE_DIR/client.crt" -days 3650 2>/dev/null
        done
    fi
done

# Cleanup temporary generation files
find "$CERTS_DIR" -type f -name "*.csr" -delete
find "$CERTS_DIR" -type f -name "*.srl" -delete
find "$CERTS_DIR" -type f -name "*.ext" -delete

# This string is caught by boot_network.sh
echo "STATUS:GENERATED"
exit 0