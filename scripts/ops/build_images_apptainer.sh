#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
IMAGE_DIR="$PROJECT_ROOT/images"

mkdir -p "$IMAGE_DIR"
cd "$IMAGE_DIR"

echo "=== Building SIF images from Docker Registries ==="

# Build Cloud and Edge nodes from Docker Hub
if [ ! -f "cloud_node.sif" ]; then
    echo "Converting panagiotispapadopoulos/zta-cloud-node:latest -> cloud_node.sif..."
    apptainer build cloud_node.sif docker://panagiotispapadopoulos/zta-cloud-node:latest
fi

if [ ! -f "edge_node.sif" ]; then
    echo "Converting panagiotispapadopoulos/zta-edge-node:latest -> edge_node.sif..."
    apptainer build edge_node.sif docker://panagiotispapadopoulos/zta-edge-node:latest
fi

# Build Flower infrastructure images
if [ ! -f "superlink.sif" ]; then
    echo "Converting flwr/superlink:1.30.0 -> superlink.sif..."
    apptainer build superlink.sif docker://flwr/superlink:1.30.0
fi

if [ ! -f "supernode.sif" ]; then
    echo "Converting flwr/supernode:1.30.0 -> supernode.sif..."
    apptainer build supernode.sif docker://flwr/supernode:1.30.0
fi

# Build NGINX for secure TLS/mTLS termination
if [ ! -f "nginx.sif" ]; then
    echo "Converting nginx:alpine -> nginx.sif..."
    apptainer build nginx.sif docker://nginx:alpine
fi

echo "=== All Apptainer images are ready in $IMAGE_DIR ==="
