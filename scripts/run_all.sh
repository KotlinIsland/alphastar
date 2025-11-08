#!/bin/bash
# Master script: Run the complete AlphaStar training pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔════════════════════════════════════════════════════════╗"
echo "║   AlphaStar Training Pipeline                          ║"
echo "║   Training on 'why' bot replays                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Step 0: Setup environment
echo "══════════════════════════════════════════════════════════"
read -p "Run Step 0 (Setup AlphaStar environment)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "${SCRIPT_DIR}/0_setup_alphastar.sh"
fi

# Step 1: Generate replays
echo ""
echo "══════════════════════════════════════════════════════════"
read -p "Run Step 1 (Generate 3 replays)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "${SCRIPT_DIR}/1_generate_replays.sh"
fi

# Step 2: Partition replays
echo ""
echo "══════════════════════════════════════════════════════════"
read -p "Run Step 2 (Partition replays)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "${SCRIPT_DIR}/2_partition_replays.sh"
fi

# Step 3: Convert to tfrecords
echo ""
echo "══════════════════════════════════════════════════════════"
echo "WARNING: Step 3 (Convert to TFRecords) is time-consuming!"
echo "It may take 30+ minutes per replay and uses SC2 extensively."
read -p "Run Step 3 (Convert to TFRecords)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "${SCRIPT_DIR}/3_convert_to_tfrecords.sh"
fi

# Step 4: Setup training
echo ""
echo "══════════════════════════════════════════════════════════"
read -p "Run Step 4 (Setup training config)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "${SCRIPT_DIR}/4_setup_training.sh"
fi

# Step 5: Train model
echo ""
echo "══════════════════════════════════════════════════════════"
echo "NOTE: Step 5 (Training) may fail due to ARM64 incompatibilities"
read -p "Run Step 5 (Train model)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "${SCRIPT_DIR}/5_train_model.sh"
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║   Pipeline Complete!                                   ║"
echo "╚════════════════════════════════════════════════════════╝"
