#!/bin/bash
set -e

echo "🛡️ Starting Hardware Root of Trust (TPM 2.0)..."

# Ensure TPM state directory exists
mkdir -p /runtime/tpm_state

# Boot the daemon. By pointing it at the directory, swtpm 
# will auto-create the necessary files if they are missing.
echo "🔌 Booting swtpm daemon..."
swtpm socket --tpmstate dir="/runtime/tpm_state" \
             --tpm2 \
             --server type=tcp,port=2321 \
             --ctrl type=tcp,port=2322 \
             --flags startup-clear \
             --daemon

sleep 2

export TPM2TOOLS_TCTI="swtpm:port=2321"
echo "✅ TPM TCTI configured."

echo "🌸 Booting Flower Client..."
exec flower-superexec "$@"