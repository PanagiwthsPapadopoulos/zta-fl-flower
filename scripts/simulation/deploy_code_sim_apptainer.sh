#!/bin/bash
# =========================================================
#   deploy_code_sim_apptainer.sh
#   Deploys code bundles (FABs) to the Apptainer Simulation
#   Engine and connects to the background swtpm daemons.
# =========================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/logs"
IMAGE_DIR="$PROJECT_ROOT/images"
LOCKS_DIR="$PROJECT_ROOT/runtime/locks"
BIN_FLWR="/python/venv/bin/flwr"
FEDERATION=local
FLWR_TMP_HOME="/tmp/${USER}_flwr"
MANIFEST_FILE=/tmp/pyproject.cloud.toml
PATCH_FILE=/tmp/tpm_socket_patch.py
EXTRA_ARGS=()
if [ "${ZTA_BATCH_MANAGED:-0}" = "1" ]; then
    mkdir -p "$PROJECT_ROOT/runtime/batch"
    FLWR_TMP_HOME="$PROJECT_ROOT/runtime/batch/flower"
    MANIFEST_FILE="$PROJECT_ROOT/runtime/batch/pyproject.cloud.toml"
    PATCH_FILE="$PROJECT_ROOT/runtime/batch/tpm_socket_patch.py"
    EXTRA_ARGS=(--bind "$PROJECT_ROOT/runtime/batch:$PROJECT_ROOT/runtime/batch"
        --env "RAY_TMPDIR=/app/runtime/batch/ray"
        --env "ZTA_BATCH_ATTEMPT=$ZTA_BATCH_ATTEMPT")
    printf 'preparing\n' > "$ZTA_BATCH_PHASE_FILE"
fi

export APPTAINERENV_PATH="/python/venv/bin:$PATH"
export APPTAINERENV_PYTHONPATH=/app

if [ -L /etc/localtime ]; then
    HOST_TZ=$(readlink /etc/localtime | sed 's#^.*zoneinfo/##')
else
    HOST_TZ="UTC"
fi
export TZ="$HOST_TZ"

# =========================================================
# PRE-FLIGHT DEPENDENCY & HEALTH CHECK
# =========================================================
if [ ! -f "$PROJECT_ROOT/runtime/infra/.network_ready" ]; then
    echo "  FATAL: The simulation network is not running or failed to boot."
    echo "   Please run 'boot_network_sim_apptainer.sh' first and wait for the success message."
    exit 1
fi

if ! pgrep -f "swtpm" > /dev/null; then
    echo "  FATAL: State mismatch detected. The lock file exists, but the Apptainer swtpm network is offline."
    echo "   Cleaning up stale lock file..."
    rm -f "$PROJECT_ROOT/runtime/infra/.network_ready"
    echo "   Please run 'boot_network_sim_apptainer.sh' to boot the network."
    exit 1
fi

# =========================================================
# HOST GPU CAPABILITY DETECTION
# =========================================================
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_ARGS="--nv"
    ZTA_GPU_AVAILABLE="true"
else
    GPU_ARGS=""
    ZTA_GPU_AVAILABLE="false"
fi

# =========================================================
# 1. LOG ROTATION & TOPOLOGY ANALYSIS
# =========================================================
echo "================================================="
echo " 1. LOG ROTATION & TOPOLOGY ANALYSIS"
echo "================================================="

echo "  Clearing stale Flower and Ray processes..."
if [ "${ZTA_BATCH_MANAGED:-0}" != "1" ]; then
pkill -9 -f 'flower-superlink|flower-supernode|flower-superexec|flwr-serverapp|flwr-clientapp|flwr run|flwr-simulation|ray' 2>/dev/null || true

echo "  Clearing old node logs and Ray temporary sessions safely..."
mkdir -p "$LOG_DIR/nodes/cloud"
mkdir -p "$LOG_DIR/system"
mkdir -p "$LOCKS_DIR"
find "$LOG_DIR/nodes" -type f -delete 2>/dev/null || true
rm -f "$LOG_DIR"/system/run_*.log 2>/dev/null || true
rm -rf "$PROJECT_ROOT/.flwr" "$FLWR_TMP_HOME" /tmp/ray 2>/dev/null || true
fi
mkdir -p "$LOG_DIR/nodes/cloud" "$LOG_DIR/system" "$LOCKS_DIR"

cd "$PROJECT_ROOT" || exit 1
CONFIG_VARS=$(python3 "$PROJECT_ROOT/scripts/setup/parse_topology.py")
eval "$CONFIG_VARS"

TOTAL_EDGES=0
for nodes in "${EDGES_PER_FOG_ARRAY[@]}"; do
    TOTAL_EDGES=$((TOTAL_EDGES + nodes))
done

if [ "$TOTAL_EDGES" -eq 0 ]; then
    echo "  FATAL: TOTAL_EDGES evaluated to 0 from network.yaml."
    exit 1
fi

echo "  Execution environment refreshed."
echo "  Detected ${TOTAL_EDGES} Edge Nodes in network.yaml (${NUM_FOGS} Fogs: ${EDGES_PER_FOG_ARRAY[*]})."

echo "================================================="
echo " SYSTEM ARCHITECTURE & TOPOLOGY SUMMARY       "
echo "================================================="
echo ""
echo "  Apptainer Host Network"
echo "  │"
echo "  └── Image: cloud_node.sif (The Simulation Engine)"
echo "      │"
echo "      └── [CLOUD] Global Aggregator (Tier 1)"

for i in $(seq 1 "$NUM_FOGS"); do
    echo "          │"
    if [ "$i" -eq "$NUM_FOGS" ]; then
        FOG_PREFIX="          └──"
        SPACER="              "
    else
        FOG_PREFIX="          ├──"
        SPACER="          │  "
    fi

    echo "${FOG_PREFIX} [FOG $i] Virtual Aggregator (Tier 2)"

    CURRENT_EDGES=${EDGES_PER_FOG_ARRAY[$((i-1))]:-0}
    if [ "$CURRENT_EDGES" -gt 0 ]; then
        for j in $(seq 1 "$CURRENT_EDGES"); do
            if [ "$j" -eq "$CURRENT_EDGES" ]; then
                EDGE_PREFIX="└──"
            else
                EDGE_PREFIX="├──"
            fi
            echo "${SPACER} ${EDGE_PREFIX} [EDGE ${i}_${j}] Ephemeral Ray Actor  ----(UNIX)---->   swtpm: edge_${i}_${j}/swtpm.sock"
        done
    else
        echo "${SPACER} └── (No Edge Nodes Assigned)"
    fi
done
echo ""
echo "  External Hardware Emulators (${TOTAL_EDGES} Apptainer swtpm Daemons)"

# =========================================================
# 2. DATASET ARTIFACT PREPARATION
# =========================================================
echo "================================================="
echo "   COMPILING DATASET ARTIFACTS (IF MISSING)      "
echo "================================================="

if [ -z "${ZTA_BATCH_SESSION:-}" ]; then
if ! apptainer exec --pwd /app \
    --env N_SIM="${TOTAL_EDGES}" \
    --bind "$PROJECT_ROOT/data:/app/data" \
    --bind "$PROJECT_ROOT/config:/app/config:ro" \
    --bind "$PROJECT_ROOT/src:/app/src:ro" \
    --bind "$PROJECT_ROOT/scripts:/app/scripts:ro" \
    "$IMAGE_DIR/cloud_node.sif" \
    python3 /app/scripts/setup/build_artifacts.py; then

    echo "  FATAL: Artifact compilation failed! Aborting network boot."
    exit 1
fi
echo "  Dataset Artifacts Verified!"
fi

# =========================================================
# 3. SIMULATION DEPLOYMENT DISPATCH
# =========================================================
echo "================================================="
echo " 3. SIMULATION DEPLOYMENT DISPATCH"
echo "================================================="

TEMPLATE="$PROJECT_ROOT/pyproject.toml"
echo "  Copying application manifest; simulation node count is passed explicitly to Flower."
cp "$TEMPLATE" "$MANIFEST_FILE"

# Pre-create FLWR_HOME/config.toml
mkdir -p "$FLWR_TMP_HOME"
cat <<EOF > "$FLWR_TMP_HOME/config.toml"
[superlink]
default = "local"

[superlink.local]
address = ":local:"
EOF

# Route tpm2_* calls from simulation TCP addresses to the Apptainer UNIX sockets.
# Flower's timeout is read from config/network.yaml by the shared server code.
cat << 'EOF' > "$PATCH_FILE"
import subprocess, os, re

_orig_run = subprocess.run
def _patched_run(cmd, *args, **kwargs):
    if isinstance(cmd, (list, tuple)) and cmd and os.path.basename(str(cmd[0])).startswith("tpm2_"):
        env = kwargs.get("env")
        if env is None:
            env = os.environ
        tcti = env.get("TPM2TOOLS_TCTI", "")
        m = re.search(r'tpm-(\d+)-(\d+)', tcti)
        if m:
            f_num, e_num = m.groups()
            sock_path = f"/app/runtime/tpm_state/edge_{f_num}_{e_num}/swtpm.sock"
            # TPMAttestation passes env explicitly; update that child environment
            # without leaking one Edge's socket into the next Ray actor call.
            kwargs["env"] = {**env, "TPM2TOOLS_TCTI": f"swtpm:path={sock_path}"}
    return _orig_run(cmd, *args, **kwargs)

subprocess.run = _patched_run

EOF

touch "$PROJECT_ROOT/config/training.yaml"
echo "  [cloud] sim: 1 Central Server + ${TOTAL_EDGES} Edge Actors"

launch_simulation() {
apptainer exec --writable-tmpfs $GPU_ARGS ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} --pwd /app \
    --env TZ="${HOST_TZ}" \
    --env TIER=cloud \
    --env ZTA_TIER=cloud \
    --env ZTA_GPU_AVAILABLE="${ZTA_GPU_AVAILABLE}" \
    --env NUM_FOGS="${NUM_FOGS}" \
    --env N_SIM="${TOTAL_EDGES}" \
    --env ZTA_BATCH_SESSION="${ZTA_BATCH_SESSION:-}" \
    --env FLWR_HOME="$FLWR_TMP_HOME" \
    --env PYTHONPATH="/tmp:/app" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 \
    --env TORCH_NUM_THREADS=1 \
    --env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    --env RAY_INCLUDE_DASHBOARD=0 \
    --env RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    --bind "$PROJECT_ROOT:/app" \
    --bind "$LOG_DIR/nodes/cloud:/app/logs" \
    --bind "$LOG_DIR/system:/app/logs/system" \
    --bind "$PROJECT_ROOT/data:/app/data" \
    --bind "$PROJECT_ROOT/results:/app/results" \
    --bind "$PROJECT_ROOT/config:/app/config:ro" \
    --bind "$PROJECT_ROOT/runtime/tpm_state:/app/runtime/tpm_state:rw" \
    --bind "$LOCKS_DIR:/app/runtime/locks:rw" \
    --bind "$PROJECT_ROOT/src:/app/src:ro" \
    --bind "$MANIFEST_FILE:/app/pyproject.toml" \
    --bind "$PATCH_FILE:/tmp/sitecustomize.py:ro" \
    "$IMAGE_DIR/cloud_node.sif" \
    bash -e -c '
        sed -i "s/dashboard_process\.children()/()/g" /python/venv/lib/python3.13/site-packages/ray/_private/node.py
        sed -i "s/raise AccessDenied(pid) from err/pass/g" /python/venv/lib/python3.13/site-packages/ray/thirdparty_files/psutil/_pslinux.py

        if [ -n "${ZTA_BATCH_SESSION:-}" ]; then
            exec /python/venv/bin/python3 /app/scripts/ops/batch_worker.py "$ZTA_BATCH_SESSION"
        fi
        FED_CFG=$(/python/venv/bin/python3 -c "
import os, tomllib, pathlib
data = tomllib.loads(pathlib.Path(\"/app/pyproject.toml\").read_text())
local = data.get(\"tool\", {}).get(\"flwr\", {}).get(\"federations\", {}).get(\"local\", {})
opts = local.get(\"options\", local)
backend = opts.get(\"backend\", {})
cres = backend.get(\"client-resources\", opts.get(\"client-resources\", {}))
iargs = backend.get(\"init-args\", opts.get(\"init-args\", {}))
cfg = [\"num-supernodes=\" + str(int(os.environ[\"N_SIM\"]))]
if \"num-cpus\" in cres: cfg.append(f\"client-resources-num-cpus={cres['\''num-cpus'\'']}\")
if \"num-gpus\" in cres: cfg.append(f\"client-resources-num-gpus={cres['\''num-gpus'\'']}\")
if \"num-cpus\" in iargs: cfg.append(f\"init-args-num-cpus={iargs['\''num-cpus'\'']}\")
if \"num-gpus\" in iargs: cfg.append(f\"init-args-num-gpus={iargs['\''num-gpus'\'']}\")
if \"verbose\" in opts: cfg.append(f\"verbose={str(opts['\''verbose'\'']).lower()}\")
print(\" \".join(cfg))
")
        echo "PATCH: Ray, psutil, and dynamic TPM UNIX socket routing applied successfully."
        echo "CONFIG: Loaded federation config from pyproject.toml -> ${FED_CFG}"
        /python/venv/bin/flwr run /app local --stream --run-config "mode=\"sim\" tier=\"cloud\" min_clients=${N_SIM}" --federation-config "${FED_CFG}"
    '
}
if [ "${ZTA_BATCH_MANAGED:-0}" = "1" ]; then
    printf 'training\n' > "$ZTA_BATCH_PHASE_FILE"
    if [ -n "${ZTA_BATCH_SESSION:-}" ]; then
        launch_simulation
    else
        launch_simulation > "$LOG_DIR/system/run_cloud.log" 2>&1
    fi
else
    launch_simulation > "$LOG_DIR/system/run_cloud.log" 2>&1 &
fi

echo ""
echo "  Global synchronization dispatched!"
echo "All output is safely redirected. Your terminal is now clean."
echo ""
echo "To monitor the background training, run:"
echo "cat logs/nodes/cloud/cloud_server.jsonl"
