#!/bin/bash

# ==============================================================================
# OVERNIGHT BATCH ORCHESTRATOR
# ==============================================================================
# DESCRIPTION:
# This script orchestrates multiple consecutive federated learning runs without
# user intervention. It operates by hot-swapping YAML configuration files into
# the live `config/` directory, dispatching the application code, and monitoring
# the dynamically generated output JSON files.
#
# PREREQUISITES:
# 1. Create an `experiments_queue/` directory at the project root.
# 2. Inside `experiments_queue/`, create a folder for each run (e.g., `run_01`).
# 3. Place ONLY the YAML files you wish to override (e.g., `training.yaml`) into 
#    these subfolders. The script will overlay them onto the default configs.
#
# BEHAVIOR:
# - Backs up the original `config/` directory.
# - Clears TPM state and boots the Docker network in the background.
# - Copies the override YAMLs for the active run into the live `config/` directory.
# - Triggers `scripts/ops/deploy_code_docker.sh`.
# - Dynamically finds the latest .json file in `results/` to monitor round progress.
# - Sends SIGTERM to the boot script to trigger clean network teardown.
#
# STATE TRACKING:
# The script maintains `results/batch_archives/batch_status.log` to track progress. 
# If interrupted, re-running the script will automatically skip completed runs.
# ==============================================================================

QUEUE_DIR="experiments_queue"
RESULTS_DIR="results"
ARCHIVE_DIR="$RESULTS_DIR/batch_archives"
STATUS_FILE="$ARCHIVE_DIR/batch_status.log"

# Failsafe timeout (7 hours per run)
TIMEOUT_SECONDS=25200 

mkdir -p "$QUEUE_DIR"
mkdir -p "$ARCHIVE_DIR"
touch "$STATUS_FILE"

BOOT_PID=""

# ==============================================================================
# EMERGENCY CLEANUP TRAP
# ==============================================================================
emergency_cleanup() {
    echo -e "\n================================================="
    echo " 🚨 INTERRUPT DETECTED: Executing Emergency Teardown"
    echo "================================================="
    
    # 1. Process Purge
    echo "🛑 Terminating deployments..."
    pkill -9 -f "deploy_code_docker.sh" 2>/dev/null
    
    # Trigger the programmatic Ctrl+C on the boot script
    if [ -n "$BOOT_PID" ] && kill -0 $BOOT_PID 2>/dev/null; then
        echo "🛑 Sending SIGTERM to network boot script to spin down Docker..."
        kill -TERM $BOOT_PID
        wait $BOOT_PID 2>/dev/null || true
    fi
    
    # 2. Configuration Rollback
    if [ -d "config_backup/" ]; then
        echo "♻️  Rolling back configurations to pristine baseline..."
        rm -rf config/
        mv config_backup/ config/
    fi
    
    # 3. State Resolution
    if [ -n "$RUN_NAME" ]; then
        echo "📝 Marking current run ($RUN_NAME) as CANCELLED..."
        echo "$RUN_NAME: CANCELLED" >> "$STATUS_FILE"
    fi
    
    echo "✅ Teardown complete. Exiting safely."
    exit 1
}

# Register the trap for SIGINT (Ctrl+C) and SIGTERM (Kill)
trap emergency_cleanup SIGINT SIGTERM

echo "================================================="
echo " 🌌 STARTING BATCH ORCHESTRATOR"
echo "================================================="

# Iterate through the queue subdirectories
for run_path in "$QUEUE_DIR"/*/; do
    # Skip if directory is empty or doesn't exist
    [ -d "$run_path" ] || continue
    
    RUN_NAME=$(basename "$run_path")

    # Check state tracker
    if grep -q "^$RUN_NAME: SUCCESS" "$STATUS_FILE"; then
        echo "⏭️  Skipping $RUN_NAME - Already marked as SUCCESS."
        continue
    fi

    echo -e "\n================================================="
    echo " 🚀 INITIATING RUN: $RUN_NAME"
    echo "================================================="

    # 1. State Reset & Config Hot-Swap
    echo "📦 Backing up baseline configs..."
    cp -r config/ config_backup/
    
    echo "🔄 Injecting YAML overrides from $RUN_NAME..."
    cp -r "$run_path"*.yaml config/ 2>/dev/null || true

    # Extract target rounds dynamically (Aggressively sanitized)
    TARGET_ROUNDS=$(grep "num_rounds:" config/training.yaml | awk -F':' '{print $2}' | cut -d'#' -f1 | tr -d ' \r\n' || echo "0")
    if [ "$TARGET_ROUNDS" == "0" ]; then
        echo "⚠️ Could not parse target rounds. Defaulting to safe fallback."
        TARGET_ROUNDS=10 # Fallback
    fi
    echo "🎯 Target rounds for $RUN_NAME: $TARGET_ROUNDS"

    # 2. Infrastructure Boot
    echo "🧹 Wiping legacy TPM State..."
    rm -rf runtime/tpm_state 2>/dev/null
    mkdir -p runtime/tpm_state

    echo "🏗️ Booting fresh Docker Federation in background..."
    ./scripts/ops/boot_network_docker.sh > "$ARCHIVE_DIR/boot_$RUN_NAME.log" 2>&1 &
    BOOT_PID=$!

    # Dynamically wait for the network to emit the LIVE signal
    echo "⏳ Waiting for network to stabilize (this may take a few minutes)..."
    NETWORK_READY=0
    for i in {1..120}; do # Failsafe timeout of 10 minutes (120 loops * 5s)
        if grep -q "ENGINE IS LIVE" "$ARCHIVE_DIR/boot_$RUN_NAME.log" 2>/dev/null; then
            echo "✅ Network initialization confirmed!"
            NETWORK_READY=1
            break
        fi
        
        # Check if the background boot script crashed prematurely
        if ! kill -0 $BOOT_PID 2>/dev/null; then
            echo "❌ Boot script died unexpectedly!"
            break
        fi
        sleep 5
    done

    # Abort this run if the network never booted successfully
    if [ "$NETWORK_READY" -eq 0 ]; then
        echo "⚠️ Network failed to boot. Marking as FAILED_BOOT and skipping..."
        echo "$RUN_NAME: FAILED_BOOT" >> "$STATUS_FILE"
        
        if kill -0 $BOOT_PID 2>/dev/null; then
            kill -TERM $BOOT_PID
            wait $BOOT_PID 2>/dev/null || true
        fi
        
        rm -rf config/
        mv config_backup/ config/
        continue
    fi

    # 3. Dispatch
    echo "🚢 Dispatching FAB to nodes..."
    
    # Record the strict UNIX epoch timestamp exactly when the run starts
    RUN_START_TIME=$(date +%s)
    
    ./scripts/ops/deploy_code_docker.sh > "$ARCHIVE_DIR/deploy_$RUN_NAME.log" 2>&1 &
    
    # 4. Polling Loop - Dynamic File Discovery
    ELAPSED=0
    SUCCESS_FLAG=0
    echo "⏳ Monitoring for the latest JSON output in $RESULTS_DIR..."

    while [ $ELAPSED -lt $TIMEOUT_SECONDS ]; do
        # Retrieve the newest file AND its timestamp
        NEWEST_INFO=$( (find "$RESULTS_DIR" -name "*.json" -type f -exec stat -c "%Y %n" {} + 2>/dev/null || find "$RESULTS_DIR" -name "*.json" -type f -exec stat -f "%m %N" {} + 2>/dev/null) | sort -nr | head -n1 )
        
        if [ -n "$NEWEST_INFO" ]; then
            FILE_TIME=$(echo "$NEWEST_INFO" | awk '{print $1}')
            ACTIVE_JSON=$(echo "$NEWEST_INFO" | cut -d' ' -f2-)
            
            # Strictly verify the file was created AFTER the run started
            if [ "$FILE_TIME" -ge "$RUN_START_TIME" ] && [ -f "$ACTIVE_JSON" ]; then
                # macOS-bulletproof POSIX regex
                if grep -Eq '"round"[[:space:]]*:[[:space:]]*'"$TARGET_ROUNDS" "$ACTIVE_JSON"; then
                    echo "✅ Target round $TARGET_ROUNDS reached in $ACTIVE_JSON."
                    SUCCESS_FLAG=1
                    break
                fi
            fi
        fi

        sleep 10
        ELAPSED=$((ELAPSED + 10))
    done

    # 5. End of Run State Resolution
    if [ $ELAPSED -ge $TIMEOUT_SECONDS ]; then
        echo "⏳ RUN TIMEOUT: Exceeded $TIMEOUT_SECONDS seconds."
        echo "$RUN_NAME: TIMEOUT" >> "$STATUS_FILE"
    elif [ $SUCCESS_FLAG -eq 1 ]; then
        echo "✅ RUN SUCCESS"
        echo "$RUN_NAME: SUCCESS" >> "$STATUS_FILE"
    else
        echo "❌ RUN FAILED (Unknown Error)"
        echo "$RUN_NAME: FAILED" >> "$STATUS_FILE"
    fi

    # 6. Archival of Configurations
    if [ "$SUCCESS_FLAG" -eq 1 ] && [ -n "$ACTIVE_JSON" ] && [ -f "$ACTIVE_JSON" ]; then
        ACTIVE_RUN_DIR=$(dirname "$ACTIVE_JSON")
        echo "📦 Archiving executed configs into $ACTIVE_RUN_DIR..."
        cp -r config/ "$ACTIVE_RUN_DIR/executed_config/"
    fi

    # 7. Teardown Application Layer & Network
    echo "🛑 Programmatically pressing Ctrl+C on the network boot script..."
    if [ -n "$BOOT_PID" ] && kill -0 $BOOT_PID 2>/dev/null; then
        kill -TERM $BOOT_PID
        # Wait for the teardown trap in the boot script to fully finish
        wait $BOOT_PID 2>/dev/null || true
    fi
    
    # 8. Restore Baseline Configs
    echo "♻️  Restoring baseline configs..."
    rm -rf config/
    mv config_backup/ config/

    echo "🏁 Completed processing for $RUN_NAME. Cooling down..."
    sleep 5
done

echo -e "\n🎉 All queued experiments have been processed."