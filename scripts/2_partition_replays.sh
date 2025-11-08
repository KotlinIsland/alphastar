#!/bin/bash
# Script 2: Partition the replays for parallel processing

set -e

echo "=========================================="
echo "Step 2: Partitioning Replays"
echo "=========================================="

# Configuration
ALPHASTAR_DIR="${HOME}/projects/alphastar"
REPLAY_DIR="${HOME}/sc2_replays/test/raw"
CONVERTED_DIR="${HOME}/sc2_replays/test/converted"
PARTITION_DIR="${HOME}/sc2_replays/test/partitions"
NUM_PARTITIONS=1

cd "${ALPHASTAR_DIR}"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Error: AlphaStar venv not found. Run setup_alphastar.sh first."
    exit 1
fi

# Create directories
mkdir -p "${CONVERTED_DIR}"
mkdir -p "${PARTITION_DIR}"

# Run partitioning
echo "Partitioning replays..."
.venv/bin/python alphastar/unplugged/data/generate_partitions.py \
    --sc2_replay_path="${REPLAY_DIR}" \
    --converted_path="${CONVERTED_DIR}" \
    --num_partitions="${NUM_PARTITIONS}" \
    --partition_path="${PARTITION_DIR}"

echo ""
echo "✓ Partitioning complete!"
echo "Partition files created in: ${PARTITION_DIR}"
ls -lh "${PARTITION_DIR}"
echo ""
echo "Partition contents:"
cat "${PARTITION_DIR}/partition_0"
