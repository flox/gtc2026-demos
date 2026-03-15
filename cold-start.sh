#!/bin/bash
set -euo pipefail

LOG_PREFIX="[gtc2026-cold-start]"

echo "$LOG_PREFIX === Resetting to cold start state ==="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for d in cuda-cpp-cmake multi-gpu-workflow; do
    echo "$LOG_PREFIX Removing cache for $d..."
    rm -rf "$SCRIPT_DIR/$d/.flox/cache"
    echo "$LOG_PREFIX Removing run state for $d..."
    rm -rf "$SCRIPT_DIR/$d/.flox/run"
    echo "$LOG_PREFIX $d cleaned."
    echo ""
done

echo "$LOG_PREFIX Running Flox garbage collection..."
flox gc
echo ""

echo "$LOG_PREFIX === Cold start reset complete ==="
