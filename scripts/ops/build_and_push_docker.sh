#!/bin/bash

# Define your registry details (Change this to your actual Docker Hub username)
DOCKER_USER="panagiotispapadopoulos"

# ----------------------------------------
# 1. Build and Push the Cloud Image
# ----------------------------------------
echo "Building Cloud Image..."
docker build -t $DOCKER_USER/zta-cloud-node:latest -f docker/cloud.Dockerfile .

echo "Pushing Cloud Image..."
docker push $DOCKER_USER/zta-cloud-node:latest

# ----------------------------------------
# 2. Build and Push the Edge Image
# ----------------------------------------
echo "Building Edge Image..."
docker build -t $DOCKER_USER/zta-edge-node:latest -f docker/edge.Dockerfile .

echo "Pushing Edge Image..."
docker push $DOCKER_USER/zta-edge-node:latest

echo "✅ Both images have been successfully built and pushed to $DOCKER_USER!"