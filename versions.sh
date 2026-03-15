#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

check() {
    local label="$1" cmd="$2" version
    version=$(eval "$cmd" 2>/dev/null | head -1)
    if [ -n "$version" ]; then
        printf "  ${GREEN}%-14s${RESET} %s\n" "$label" "$version"
    else
        printf "  ${RED}%-14s${RESET} not found\n" "$label"
    fi
}

echo ""
echo -e "${BOLD}${CYAN}Environment Versions${RESET}"
echo ""
check "Ubuntu"    "lsb_release -ds"
check "GCC"       "gcc --version"
check "CMake"     "cmake --version"
check "Python"    "python3 --version"
check "pip"       "pip --version"
check "uv"        "uv --version"
check "nvcc"      "nvcc --version | grep release"
check "Driver"     "nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits"
echo ""
