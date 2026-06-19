#!/bin/bash
# =========================================================
# setup_security.sh
# Generates strict RFC 5280 compliant TLS certificates for the Cloud/Fog boundary
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

# 1. Enforce safe fallbacks to prevent syntax crashes if boot script fails
NUM_FOGS=${1:-1}
if ! [[ "$NUM_FOGS" =~ ^[0-9]+$ ]] || [ "$NUM_FOGS" -lt 1 ]; then 
    NUM_FOGS=1 
fi

EDGES_STRING=${2:-"1"}
read -r -a EDGES_ARRAY <<< "$EDGES_STRING"
BROKER_IP=${3:-127.0.0.1}

# Path resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CERTS_DIR="$PROJECT_ROOT/runtime/certs"

# Configure dedicated security debug logging
LOG_DIR="$PROJECT_ROOT/logs/system"
mkdir -p "$LOG_DIR"
SEC_LOG="$LOG_DIR/security_setup.log"
echo "[SECURITY SETUP INITIATED]" > "$SEC_LOG"

# =========================================================
# 2. PROTECTION CHECK FOR STATIC IDENTITIES
# =========================================================
rm -rf "$CERTS_DIR"

# Provision hierarchy paths for Cloud and Edge Certificate Authorities
mkdir -p "$CERTS_DIR/cloud_ca"
mkdir -p "$CERTS_DIR/cloud_server"
mkdir -p "$CERTS_DIR/edge_ca"

# =========================================================
# PHASE 1: CLOUD-FOG STANDARD TLS (FLOWER)
# =========================================================
echo "[LOG] Generating Cloud CA..." >> "$SEC_LOG"

# OS-Agnostic OpenSSL CA Config
cat <<EOF > "$CERTS_DIR/cloud_ca/ca.cnf"
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_ca
prompt = no

[req_distinguished_name]
CN = ZTA-Cloud-Root-CA

[v3_ca]
basicConstraints = critical, CA:TRUE
keyUsage = critical, digitalSignature, cRLSign, keyCertSign
EOF

openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
    -keyout "$CERTS_DIR/cloud_ca/ca.key" -out "$CERTS_DIR/cloud_ca/ca.crt" \
    -config "$CERTS_DIR/cloud_ca/ca.cnf" -extensions v3_ca >> "$SEC_LOG" 2>&1

echo "[LOG] Generating Cloud SuperLink CSR..." >> "$SEC_LOG"
# Generate CSR and private key for the Cloud SuperLink
openssl req -nodes -newkey rsa:2048 \
    -keyout "$CERTS_DIR/cloud_server/private-key.pem" -out "$CERTS_DIR/cloud_server/server.csr" \
    -subj "/CN=cloud-superlink" >> "$SEC_LOG" 2>&1
    
# Create isolated SAN (Subject Alternative Name) extension for the Cloud Server to pass gRPC domain validation
cat <<EOF > "$CERTS_DIR/cloud_server/san.ext"
basicConstraints = critical, CA:FALSE
keyUsage = critical, nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth
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

cat <<EOF > "$CERTS_DIR/edge_ca/ca.cnf"
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_ca
prompt = no

[req_distinguished_name]
CN = ZTA-Edge-Root-CA

[v3_ca]
basicConstraints = critical, CA:TRUE
keyUsage = critical, digitalSignature, cRLSign, keyCertSign
EOF

openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
    -keyout "$CERTS_DIR/edge_ca/ca.key" -out "$CERTS_DIR/edge_ca/ca.crt" \
    -config "$CERTS_DIR/edge_ca/ca.cnf" -extensions v3_ca >> "$SEC_LOG" 2>&1

for (( i=1; i<=NUM_FOGS; i++ )); do
    CURRENT_EDGES=${EDGES_ARRAY[$((i-1))]:-1} 

    FOG_DIR="$CERTS_DIR/fog_${i}"
    mkdir -p "$FOG_DIR"
    
    echo "[LOG] Generating Fog $i NGINX CSR..." >> "$SEC_LOG"
    # Generate the NGINX Server Certificate for this Fog Node
    openssl req -nodes -newkey rsa:2048 \
        -keyout "$FOG_DIR/nginx.key" -out "$FOG_DIR/nginx.csr" \
        -subj "/CN=fog-${i}-nginx" >> "$SEC_LOG" 2>&1
        
# CRITICAL FIX: EOF must be completely flush left
cat <<EOF > "$FOG_DIR/san.ext"
basicConstraints = critical, CA:FALSE
keyUsage = critical, nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = IP:$BROKER_IP,DNS:localhost,DNS:fog-${i}-nginx
EOF
        
    echo "[LOG] Signing Fog $i NGINX Certificate..." >> "$SEC_LOG"
    # Sign Nginx server certificate with the Edge CA
    openssl x509 -req -in "$FOG_DIR/nginx.csr" \
        -CA "$CERTS_DIR/edge_ca/ca.crt" -CAkey "$CERTS_DIR/edge_ca/ca.key" -CAcreateserial \
        -out "$FOG_DIR/nginx.crt" -days 3650 -extfile "$FOG_DIR/san.ext" >> "$SEC_LOG" 2>&1

    # Only attempt to generate Edge certificates if this Fog actually has Edges
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for (( j=1; j<=CURRENT_EDGES; j++ )); do
            EDGE_DIR="$CERTS_DIR/edge_${i}_${j}"
            mkdir -p "$EDGE_DIR"
            
            echo "[LOG] Generating Edge ${i}_${j} Client CSR..." >> "$SEC_LOG"
            # Generate cryptographic identity for an individual Edge ClientApp
            openssl req -nodes -newkey rsa:2048 \
                -keyout "$EDGE_DIR/client.key" -out "$EDGE_DIR/client.csr" \
                -subj "/CN=edge-agent-${i}-${j}" >> "$SEC_LOG" 2>&1

                        
            # Add strict clientAuth extension to the Edge Client
cat <<EOF > "$EDGE_DIR/client.ext"
basicConstraints = CA:FALSE
keyUsage = critical, nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth, serverAuth
EOF

                
            echo "[LOG] Signing Edge ${i}_${j} Client Certificate..." >> "$SEC_LOG"
            # Issue signed client certificate required for backend authentication
            openssl x509 -req -in "$EDGE_DIR/client.csr" \
                -CA "$CERTS_DIR/edge_ca/ca.crt" -CAkey "$CERTS_DIR/edge_ca/ca.key" -CAcreateserial \
                -out "$EDGE_DIR/client.crt" -days 3650 -extfile "$EDGE_DIR/client.ext" >> "$SEC_LOG" 2>&1
        done
    fi
done

# Cleanup
find "$CERTS_DIR" -type f \( -name "*.csr" -o -name "*.srl" -o -name "*.ext" -o -name "*.cnf" \) -delete

echo "[SECURITY SETUP COMPLETED SUCCESSFULLY]" >> "$SEC_LOG"

echo "STATUS:GENERATED"
exit 0