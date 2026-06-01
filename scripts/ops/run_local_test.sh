#!/bin/bash

# =========================================================
# MASTER ORCHESTRATION: PIPELINE-FIRST AUDIT
# =========================================================

# Setup cleanup trap to handle normal exit and Ctrl+C (SIGINT/SIGTERM)
cleanup() {
    echo -e "\n================================================="
    echo " 🧹 CLEANUP: Restoring Environment"
    echo "================================================="
    
    echo "🛑 Shutting down network components..."
    if [ -n "$BOOT_PID" ]; then
        kill $BOOT_PID 2>/dev/null
    fi
    
    if [ -f "pyproject.toml.bak" ]; then
        echo "📦 Restoring original pyproject.toml..."
        mv pyproject.toml.bak pyproject.toml
    fi
    
    echo "✅ Cleanup complete."
}

# Register the trap for EXIT (normal completion) and route SIGINT/SIGTERM to exit so the trap catches them
trap cleanup EXIT
trap 'exit 0' SIGINT SIGTERM

# 1. Environment Preparation
echo "🧹 Cleaning previous artifacts..."
rm -rf logs/

# We replace 'rm -f pyproject.toml' with a backup command so VS Code detects it.
echo "📦 Creating backup of pyproject.toml..."
if [ -f "pyproject.toml" ]; then
    cp pyproject.toml pyproject.toml.bak
fi

echo "🎲 Generating random topology and configuration..."
python3 tools/generate_random_toml.py
if [ $? -ne 0 ]; then echo "❌ Failed to generate TOML"; exit 1; fi

# 2. Boot the Network
echo "🚀 Booting 3-Tier Architecture..."
echo "y" | ./scripts/ops/boot_network.sh &

BOOT_PID=$!

# Get target rounds from TOML
TARGET_ROUNDS=$(grep "num_rounds" pyproject.toml | awk -F'=' '{print $2}' | tr -d ' ')
echo "Target rounds to verify: $TARGET_ROUNDS"

echo "⏳ Waiting for network to initialize..."
sleep 10

# 3. Continuous Pipeline Monitoring (Blocking)
echo "================================================="
echo " 🔎 MONITORING MODE: Starting Pipeline Audit"
echo "================================================="
echo "Monitoring pipeline execution. Press Ctrl+C when finished."

# Run the monitor in the foreground
while true; do
    # Run the monitor and capture output
    # We pipe to 'tee' to display in terminal while filtering for the success string
    SUCCESS_FOUND=$(python3 verification/verify_pipeline.py | tee /dev/tty | grep "Round $TARGET_ROUNDS verified successfully")
    
    if [[ -n "$SUCCESS_FOUND" ]]; then
        echo -e "\n✅ Target round $TARGET_ROUNDS reached. Initiating final audit..."
        break
    else
        echo "⏳ Round $TARGET_ROUNDS not yet reached. Retrying in 5 seconds..."
        sleep 5
    fi
done

# 4. Post-Monitoring Audit
echo ""
echo "================================================="
echo " 🏁 MONITORING STOPPED: Executing Final Audit"
echo "================================================="

echo "🔍 Running one-time Configuration Audit..."
python3 verification/verify_configs.py

if [ $? -eq 0 ]; then
    echo "✅ Final audit completed successfully."
else
    echo "⚠️ Final audit completed with discrepancies."
fi

# 5. Shutdown is now handled entirely by the cleanup trap at the top!