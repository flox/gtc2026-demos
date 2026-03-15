#!/bin/bash
set -euo pipefail

LOG_PREFIX="[gtc2026-setup]"

echo "$LOG_PREFIX === GTC 2026 Demos Setup ==="
echo ""

# Install Flox if not already installed
if command -v flox &>/dev/null; then
    echo "$LOG_PREFIX Flox already installed: $(flox --version)"
else
    echo "$LOG_PREFIX Flox not found, installing..."
    echo "$LOG_PREFIX Downloading .deb package to $HOME..."
    wget -P ~ https://downloads.flox.dev/by-env/nightly/deb/flox-1.10.0-35-gfed448a4.x86_64-linux.deb
    echo "$LOG_PREFIX Installing package..."
    sudo apt install -y ~/flox-1.10.0-35-gfed448a4.x86_64-linux.deb
    echo "$LOG_PREFIX Cleaning up .deb file..."
    rm -f ~/flox-1.10.0-35-gfed448a4.x86_64-linux.deb
    echo "$LOG_PREFIX Installed: $(flox --version)"
fi

echo ""

# Determine repo root: if running from within the repo, use it; otherwise clone
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/cuda-cpp-cmake" ] && [ -d "$SCRIPT_DIR/multi-gpu-workflow" ]; then
    REPO_DIR="$SCRIPT_DIR"
    echo "$LOG_PREFIX Running inside existing clone: $REPO_DIR"
else
    echo "$LOG_PREFIX Not inside repo, cloning..."
    git clone https://github.com/flox/gtc2026-demos.git
    REPO_DIR="$(pwd)/gtc2026-demos"
    echo "$LOG_PREFIX Cloned to: $REPO_DIR"
fi

echo ""

# Pre-activate each environment to pull dependencies, then clean up caches
for d in cuda-cpp-cmake multi-gpu-workflow; do
    echo "$LOG_PREFIX Pre-activating $d to pull dependencies..."
    flox activate -d "$REPO_DIR/$d" -- true
    echo "$LOG_PREFIX Cleaning cache for $d..."
    rm -rf "$REPO_DIR/$d/.flox/cache"
    echo "$LOG_PREFIX $d ready."
    echo ""
done

echo "$LOG_PREFIX === Setup complete ==="
