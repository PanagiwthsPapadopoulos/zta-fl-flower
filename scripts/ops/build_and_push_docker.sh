#!/bin/bash

# Use "docker buildx imagetools inspect panagiotispapadopoulos/zta-cloud-node:latest" to inspect the images.

set -e

DOCKER_USER="panagiotispapadopoulos"

# Ensure a buildx builder exists and is active for multi-arch compilation
docker buildx create --use --name zta-builder 2>/dev/null || docker buildx use zta-builder

echo "Building and Pushing Cloud Image (Multi-Arch with Registry Caching)..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t $DOCKER_USER/zta-cloud-node:latest \
  --cache-from type=registry,ref=$DOCKER_USER/zta-cloud-node:buildcache \
  --cache-to type=registry,ref=$DOCKER_USER/zta-cloud-node:buildcache,mode=max \
  -f docker/cloud.Dockerfile \
  --push .

echo "Building and Pushing Edge Image (Multi-Arch with Registry Caching)..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t $DOCKER_USER/zta-edge-node:latest \
  --cache-from type=registry,ref=$DOCKER_USER/zta-edge-node:buildcache \
  --cache-to type=registry,ref=$DOCKER_USER/zta-edge-node:buildcache,mode=max \
  -f docker/edge.Dockerfile \
  --push .

echo "Multi-arch images successfully built and pushed to $DOCKER_USER!"