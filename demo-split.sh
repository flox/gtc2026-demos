#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="gtc2026"

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with first pane in cuda-cpp-cmake
tmux new-session -d -s "$SESSION" -c "$SCRIPT_DIR/cuda-cpp-cmake"
tmux send-keys -t "$SESSION" "flox activate" Enter

# Split horizontally and open multi-gpu-workflow
tmux split-window -h -t "$SESSION" -c "$SCRIPT_DIR/multi-gpu-workflow"
tmux send-keys -t "$SESSION" "flox activate" Enter

# Attach to the session
tmux attach -t "$SESSION"
