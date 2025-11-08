#!/bin/bash
# Script 0: Setup AlphaStar environment and install dependencies

set -e

echo "=========================================="
echo "Setup: Installing AlphaStar Dependencies"
echo "=========================================="

ALPHASTAR_DIR="${HOME}/projects/alphastar"

cd "${ALPHASTAR_DIR}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    brew install uv
fi

# Create virtual environment
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

echo "Creating virtual environment with uv..."
uv venv

echo "Installing dependencies..."
# Install packages needed for data processing
# Note: dm-acme and related packages are not available for ARM64 macOS
uv pip install \
    pysc2 \
    s2clientprotocol \
    ml-collections \
    tensorflow \
    chex \
    dm-tree \
    numpy \
    protobuf \
    apache-beam \
    absl-py \
    pandas

echo ""
echo "✓ AlphaStar environment setup complete!"
echo ""
echo "Installed packages for data processing."
echo ""
echo "NOTE: Full training dependencies (dm-acme, jax, etc.) are not available"
echo "for ARM64 macOS. For full training, use a Linux x86_64 machine."
