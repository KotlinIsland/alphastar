#!/bin/bash
# Script 3: Convert replays to TFRecord format
# NOTE: This step requires SC2 to re-render replays and may take a long time

set -e

echo "=========================================="
echo "Step 3: Converting Replays to TFRecords"
echo "=========================================="
echo "WARNING: This step is computationally intensive!"
echo "It may take 30+ minutes per replay."
echo ""

# Configuration
ALPHASTAR_DIR="${HOME}/projects/alphastar"
REPLAY_DIR="${HOME}/sc2_replays/test/raw"
CONVERTED_DIR="${HOME}/sc2_replays/test/converted"
PARTITION_DIR="${HOME}/sc2_replays/test/partitions"
CONVERTER_SETTINGS="${ALPHASTAR_DIR}/alphastar/unplugged/configs/alphastar_supervised_converter_settings.pbtxt"

cd "${ALPHASTAR_DIR}"

# Kill any existing SC2 processes to avoid mutex issues
echo "Cleaning up any existing SC2 processes..."
pkill -9 "SC2" 2>/dev/null || true
sleep 3

# Set PYTHONPATH
export PYTHONPATH="${ALPHASTAR_DIR}:${PYTHONPATH}"

# Convert each partition
for partition_file in "${PARTITION_DIR}"/partition_*; do
    partition_name=$(basename "${partition_file}")
    echo ""
    echo "Converting ${partition_name}..."
    echo "Started at: $(date)"

    .venv/bin/python alphastar/unplugged/data/generate_dataset.py \
        --sc2_replay_path="${REPLAY_DIR}" \
        --converted_path="${CONVERTED_DIR}" \
        --partition_file="${partition_file}" \
        --converter_settings="${CONVERTER_SETTINGS}" \
        --logtostderr

    echo "Completed ${partition_name} at: $(date)"
done

echo ""
echo "✓ Conversion complete!"
echo "TFRecord files saved to: ${CONVERTED_DIR}"
echo ""
echo "Checking converted files:"
find "${CONVERTED_DIR}" -name "*.tfrecord" -exec ls -lh {} \;
