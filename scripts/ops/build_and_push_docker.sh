#!/bin/bash

# Use "docker buildx imagetools inspect panagiotispapadopoulos/zta-cloud-node:latest" to inspect the images.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

DOCKER_USER="panagiotispapadopoulos"
BUILDER_NAME="zta-builder"

if ! docker info >/dev/null; then
  echo "Docker engine is unavailable. Start Docker Desktop and wait for it to be ready before retrying." >&2
  exit 1
fi
docker buildx version

# Reuse the named builder or create it without suppressing creation errors.
if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
  docker buildx create --name "$BUILDER_NAME" --driver docker-container
fi
docker buildx inspect "$BUILDER_NAME" --bootstrap

echo "Building and Pushing Cloud Image (Multi-Arch with Registry Caching)..."
docker buildx build \
  --builder "$BUILDER_NAME" \
  --platform linux/amd64,linux/arm64 \
  -t $DOCKER_USER/zta-cloud-node:latest \
  --cache-from type=registry,ref=$DOCKER_USER/zta-cloud-node:buildcache \
  --cache-to type=registry,ref=$DOCKER_USER/zta-cloud-node:buildcache,mode=max \
  -f docker/cloud.Dockerfile \
  --push .

echo "Building and Pushing Edge Image (Multi-Arch with Registry Caching)..."
docker buildx build \
  --builder "$BUILDER_NAME" \
  --platform linux/amd64,linux/arm64 \
  -t $DOCKER_USER/zta-edge-node:latest \
  --cache-from type=registry,ref=$DOCKER_USER/zta-edge-node:buildcache \
  --cache-to type=registry,ref=$DOCKER_USER/zta-edge-node:buildcache,mode=max \
  -f docker/edge.Dockerfile \
  --push .

echo "Multi-arch images successfully built and pushed to $DOCKER_USER!"
