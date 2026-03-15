#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

ENV_DIR="${1:-$(pwd)}"

if [ ! -d "$ENV_DIR/.flox" ]; then
    echo "No .flox directory found in: $ENV_DIR"
    echo "Run this from a directory with a Flox environment, or pass one as an argument."
    exit 1
fi

ENV_NAME="$(basename "$ENV_DIR")"

get_version() {
    local cmd="$1"
    local result
    result=$(eval "$cmd" 2>/dev/null | head -1)
    if [ -n "$result" ]; then
        echo "$result"
    else
        echo "-"
    fi
}

# Collect host versions
host_ubuntu=$(get_version "lsb_release -ds")
host_gcc=$(get_version "gcc --version | grep -oP '\d+\.\d+\.\d+'")
host_cmake=$(get_version "cmake --version | awk '{print \$3}'")
host_python=$(get_version "python3 --version | awk '{print \$2}'")
host_pip=$(get_version "pip --version | awk '{print \$2}'")
host_uv=$(get_version "uv --version | awk '{print \$2}'")
host_nvcc=$(get_version "nvcc --version | grep -oP 'release \K[\d.]+'")
host_driver=$(get_version "nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")

# Collect flox versions
flox_ubuntu=$(flox activate -d "$ENV_DIR" -- bash -c "$(declare -f get_version); get_version 'lsb_release -ds'" 2>/dev/null)
flox_gcc=$(flox activate -d "$ENV_DIR" -- bash -c "$(declare -f get_version); get_version \"gcc --version | grep -oP '\d+\.\d+\.\d+'\"" 2>/dev/null)
flox_cmake=$(flox activate -d "$ENV_DIR" -- bash -c "$(declare -f get_version); get_version \"cmake --version | awk '{print \\\$3}'\"" 2>/dev/null)
flox_python=$(flox activate -d "$ENV_DIR" -- bash -c "$(declare -f get_version); get_version \"python3 --version | awk '{print \\\$2}'\"" 2>/dev/null)
flox_pip=$(flox activate -d "$ENV_DIR" -- bash -c "$(declare -f get_version); get_version \"pip --version | awk '{print \\\$2}'\"" 2>/dev/null)
flox_uv=$(flox activate -d "$ENV_DIR" -- bash -c "$(declare -f get_version); get_version \"uv --version | awk '{print \\\$2}'\"" 2>/dev/null)
flox_nvcc=$(flox activate -d "$ENV_DIR" -- bash -c "$(declare -f get_version); get_version \"nvcc --version | grep -oP 'release \K[\d.]+'\"" 2>/dev/null)
flox_driver=$(flox activate -d "$ENV_DIR" -- bash -c "$(declare -f get_version); get_version 'nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits'" 2>/dev/null)

# Display side-by-side
col1=22
col2=22

colorize() {
    local val="$1"
    if [ "$val" = "-" ]; then
        printf "${RED}%-${col2}s${RESET}" "$val"
    else
        printf "${GREEN}%-${col2}s${RESET}" "$val"
    fi
}

echo ""
printf "  ${BOLD}${CYAN}%-14s${RESET} ${DIM}%-${col1}s${RESET} ${BOLD}%-${col2}s${RESET}\n" "" "Host" "Flox ($ENV_NAME)"
printf "  ${DIM}%-14s %-${col1}s %-${col2}s${RESET}\n" "──────────────" "──────────────────────" "──────────────────────"

row() {
    local label="$1" host="$2" flox="$3"
    printf "  ${BOLD}%-14s${RESET} " "$label"
    colorize "$host"
    printf " "
    colorize "$flox"
    echo ""
}

row "Ubuntu"     "$host_ubuntu" "$flox_ubuntu"
row "GCC"        "$host_gcc"    "$flox_gcc"
row "CMake"      "$host_cmake"  "$flox_cmake"
row "Python"     "$host_python" "$flox_python"
row "pip"        "$host_pip"    "$flox_pip"
row "uv"         "$host_uv"    "$flox_uv"
row "nvcc"       "$host_nvcc"   "$flox_nvcc"
row "GPU Driver" "$host_driver" "$flox_driver"
echo ""
