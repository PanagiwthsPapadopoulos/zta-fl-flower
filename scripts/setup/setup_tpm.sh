#!/bin/bash

# =========================================================
# setup_tpm.sh

# Initializes the local NVRAM storage directories for the
# Software TPM 2.0 emulators.
# =========================================================

NUM_FOGS=$1
EDGES_ARRAY=($2)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TPM_BASE_DIR="$PROJECT_ROOT/runtime/tpm_state"

echo "[DEBUG TPM] ---------------------------------------"
echo "[DEBUG TPM] Provisioning Hardware State Directories"

for i in $(seq 1 "$NUM_FOGS"); do
CURRENT_EDGES=${EDGES_ARRAY[$((i-1))]:-0}
if [ "$CURRENT_EDGES" -gt 0 ]; then
for j in $(seq 1 "$CURRENT_EDGES"); do
mkdir -p "$TPM_BASE_DIR/edge_${i}_${j}"
echo "[DEBUG TPM] Created NVRAM volume for Edge ${i}_${j}"
done
fi
done

echo "[DEBUG TPM] TPM Volume Provisioning Complete."
exit 0