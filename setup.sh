#!/bin/bash
set -euo pipefail

# Install Flox
wget https://downloads.flox.dev/by-env/nightly/deb/flox-1.10.0-35-gfed448a4.x86_64-linux.deb
sudo apt install -y ./flox-1.10.0-35-gfed448a4.x86_64-linux.deb
rm -f ./flox-1.10.0-35-gfed448a4.x86_64-linux.deb

# Determine repo root: if running from within the repo, use it; otherwise clone
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/cuda-cpp-cmake" ] && [ -d "$SCRIPT_DIR/multi-gpu-workflow" ]; then
    REPO_DIR="$SCRIPT_DIR"
else
    git clone https://github.com/flox/gtc2026-demos.git
    REPO_DIR="$(pwd)/gtc2026-demos"
fi

# Pre-activate each environment to pull dependencies, then clean up caches
for d in cuda-cpp-cmake multi-gpu-workflow; do
    flox activate -d "$REPO_DIR/$d" -- true
    rm -rf "$REPO_DIR/$d/.flox/cache"
done
