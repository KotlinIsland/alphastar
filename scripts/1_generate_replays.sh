#!/bin/bash
# Script 1: Generate 3 replays using the 'why' bot

set -e

echo "=========================================="
echo "Step 1: Generating 3 Replays"
echo "=========================================="

# Configuration
WHY_BOT_DIR="${HOME}/projects/sc2-why"
REPLAY_DIR="${HOME}/sc2_replays/test/raw"
NUM_REPLAYS=3

# Create replay directory
mkdir -p "${REPLAY_DIR}"

# Change to why bot directory
cd "${WHY_BOT_DIR}"

# Check if venv exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment for 'why' bot..."
    /opt/homebrew/bin/python3.12 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install poetry
    .venv/bin/poetry install
fi

# Run the replay generation script
echo "Running replay generation (this may take 10-30 minutes)..."
.venv/bin/python generate_replays.py

echo ""
echo "✓ Replay generation complete!"
echo "Replays saved to: ${REPLAY_DIR}"
ls -lh "${REPLAY_DIR}"
