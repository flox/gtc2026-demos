#!/bin/bash
set -euo pipefail

LOG_PREFIX="[gtc2026-setup]"

echo "$LOG_PREFIX === GTC 2026 Demos Setup ==="
echo ""

# Install Flox if not already installed
if command -v flox &>/dev/null; then
    echo "$LOG_PREFIX Flox already installed: $(flox --version)"
else
    TMPDIR="$(mktemp -d)"
    echo "$LOG_PREFIX Flox not found, installing..."
    echo "$LOG_PREFIX Downloading .deb package to $TMPDIR..."
    wget -P "$TMPDIR" https://downloads.flox.dev/by-env/nightly/deb/flox-1.10.0-35-gfed448a4.x86_64-linux.deb
    echo "$LOG_PREFIX Installing package..."
    sudo apt install -y "$TMPDIR/flox-1.10.0-35-gfed448a4.x86_64-linux.deb"
    echo "$LOG_PREFIX Cleaning up temp directory..."
    rm -rf "$TMPDIR"
    echo "$LOG_PREFIX Installed: $(flox --version)"
fi

echo ""

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$LOG_PREFIX Repo directory: $REPO_DIR"

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
