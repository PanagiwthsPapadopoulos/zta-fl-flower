#!/bin/bash

echo "================================================="
echo " 🔍 FLOWER PORT MEDIC: Scanning for Zombies...    "
echo "================================================="

# Use lsof to find any TCP sockets actively LISTENING in our specific ranges
# 9000-9999 (Standard), 19000-19999 (Shifted), 29000-29999 (Proxy)
STUCK_PROCESSES=$(lsof -iTCP -sTCP:LISTEN -P -n | awk '$9 ~ /:(9[0-9]{3}|19[0-9]{3}|29[0-9]{3})$/')

if [ -z "$STUCK_PROCESSES" ]; then
    echo "✅ All clear! Your ports are completely free."
    exit 0
fi

echo "⚠️  WARNING: Found ghost processes holding your simulation ports:"
echo ""
echo "COMMAND     PID      USER       NODE NAME (PORT)"
echo "---------------------------------------------------------"
echo "$STUCK_PROCESSES" | awk '{printf "%-10s %-8s %-10s %-20s\n", $1, $2, $3, $9}'
echo "---------------------------------------------------------"
echo ""

read -p "💀 Do you want to execute (kill -9) all of these processes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Extract unique PIDs from the 2nd column and kill them
    PIDS=$(echo "$STUCK_PROCESSES" | awk '{print $2}' | sort -u)
    
    echo ""
    for pid in $PIDS; do
        echo "🔫 Sniping PID: $pid..."
        kill -9 $pid 2>/dev/null
    done
    
    echo "✅ Purge complete. Run this script one more time to verify the kernel released them."
else
    echo "Exiting. Ports are still locked."
fi