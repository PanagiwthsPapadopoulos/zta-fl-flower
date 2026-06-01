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
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CERTS_DIR="$PROJECT_ROOT/src/network/certs"

# Configure dedicated security debug logging
LOG_DIR="$PROJECT_ROOT/logs/system"
mkdir -p "$LOG_DIR"
SEC_LOG="$LOG_DIR/security_setup.log"
echo "[SECURITY SETUP INITIATED]" > "$SEC_LOG"

# =========================================================
# 2. PROTECTION CHECK FOR STATIC IDENTITIES
# =========================================================
# Evaluate existence of active certificate directories to avoid overwriting established Trust anchors
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

# Provision hierarchy paths for Cloud and Edge Certificate Authorities
mkdir -p "$CERTS_DIR/cloud_ca"
mkdir -p "$CERTS_DIR/cloud_server"
mkdir -p "$CERTS_DIR/edge_ca"

# =========================================================
# PHASE 1: CLOUD-FOG STANDARD TLS (FLOWER)
# =========================================================
echo "[LOG] Generating Cloud CA..." >> "$SEC_LOG"
# Establish root Certificate Authority for the central Cloud orchestrator
openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
    -keyout "$CERTS_DIR/cloud_ca/ca.key" -out "$CERTS_DIR/cloud_ca/ca.crt" \
    -subj "/CN=ZTA-Cloud-Root-CA" >> "$SEC_LOG" 2>&1

echo "[LOG] Generating Cloud SuperLink CSR..." >> "$SEC_LOG"
# Generate CSR and private key for the Cloud SuperLink
openssl req -nodes -newkey rsa:2048 \
    -keyout "$CERTS_DIR/cloud_server/private-key.pem" -out "$CERTS_DIR/cloud_server/server.csr" \
    -subj "/CN=cloud-superlink" >> "$SEC_LOG" 2>&1
    
# Create isolated SAN (Subject Alternative Name) extension for the Cloud Server to pass gRPC domain validation
cat <<EOF > "$CERTS_DIR/cloud_server/san.ext"
subjectAltName = IP:$BROKER_IP,DNS:localhost,DNS:cloud-superlink
EOF
    
echo "[LOG] Signing Cloud SuperLink Certificate..." >> "$SEC_LOG"
# Issue signed Cloud Server certificate
openssl x509 -req -in "$CERTS_DIR/cloud_server/server.csr" \
    -CA "$CERTS_DIR/cloud_ca/ca.crt" -CAkey "$CERTS_DIR/cloud_ca/ca.key" -CAcreateserial \
    -out "$CERTS_DIR/cloud_server/certificates.pem" -days 3650 -extfile "$CERTS_DIR/cloud_server/san.ext" >> "$SEC_LOG" 2>&1

# =========================================================
# PHASE 2: EDGE-FOG MUTUAL TLS (NGINX)
# =========================================================
echo "[LOG] Generating Edge CA..." >> "$SEC_LOG"
# Establish independent Certificate Authority specifically for Edge-level mTLS validation
openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
    -keyout "$CERTS_DIR/edge_ca/ca.key" -out "$CERTS_DIR/edge_ca/ca.crt" \
    -subj "/CN=ZTA-Edge-Root-CA" >> "$SEC_LOG" 2>&1

for i in $(seq 1 "$NUM_FOGS"); do
    # Fallback to 0 if the array is shorter than NUM_FOGS
    CURRENT_EDGES=${EDGES_ARRAY[$((i-1))]:-0}
    
    FOG_DIR="$CERTS_DIR/fog_${i}"
    mkdir -p "$FOG_DIR"
    
    echo "[LOG] Generating Fog $i NGINX CSR..." >> "$SEC_LOG"
    # Generate the NGINX Server Certificate for this Fog Node
    openssl req -nodes -newkey rsa:2048 \
        -keyout "$FOG_DIR/nginx.key" -out "$FOG_DIR/nginx.csr" \
        -subj "/CN=fog-${i}-nginx" >> "$SEC_LOG" 2>&1
        
    # Create isolated SAN extension for this specific Fog Nginx Sidecar
    cat <<EOF > "$FOG_DIR/san.ext"
subjectAltName = IP:$BROKER_IP,DNS:localhost,DNS:fog-${i}-nginx
EOF
        
    echo "[LOG] Signing Fog $i NGINX Certificate..." >> "$SEC_LOG"
    # Sign Nginx server certificate with the Edge CA
    openssl x509 -req -in "$FOG_DIR/nginx.csr" \
        -CA "$CERTS_DIR/edge_ca/ca.crt" -CAkey "$CERTS_DIR/edge_ca/ca.key" -CAcreateserial \
        -out "$FOG_DIR/nginx.crt" -days 3650 -extfile "$FOG_DIR/san.ext" >> "$SEC_LOG" 2>&1

    # Only attempt to generate Edge certificates if this Fog actually has Edges
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            EDGE_DIR="$CERTS_DIR/edge_${i}_${j}"
            mkdir -p "$EDGE_DIR"
            
            echo "[LOG] Generating Edge ${i}_${j} Client CSR..." >> "$SEC_LOG"
            # Generate cryptographic identity for an individual Edge ClientApp
            openssl req -nodes -newkey rsa:2048 \
                -keyout "$EDGE_DIR/client.key" -out "$EDGE_DIR/client.csr" \
                -subj "/CN=edge-agent-${i}-${j}" >> "$SEC_LOG" 2>&1
                
            echo "[LOG] Signing Edge ${i}_${j} Client Certificate..." >> "$SEC_LOG"
            # Issue signed client certificate required for backend authentication
            openssl x509 -req -in "$EDGE_DIR/client.csr" \
                -CA "$CERTS_DIR/edge_ca/ca.crt" -CAkey "$CERTS_DIR/edge_ca/ca.key" -CAcreateserial \
                -out "$EDGE_DIR/client.crt" -days 3650 >> "$SEC_LOG" 2>&1
        done
    fi
done

# Cleanup temporary generation files to leave the PKI directory strictly production-ready
find "$CERTS_DIR" -type f -name "*.csr" -delete
find "$CERTS_DIR" -type f -name "*.srl" -delete
find "$CERTS_DIR" -type f -name "*.ext" -delete

echo "[SECURITY SETUP COMPLETED SUCCESSFULLY]" >> "$SEC_LOG"

# This string is caught by boot_network.sh
echo "STATUS:GENERATED"
exit 0