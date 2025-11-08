#!/bin/bash
# Script 5: Train AlphaStar model on converted replays
# NOTE: Full training requires JAX and other dependencies that may not work on ARM64

set -e

echo "=========================================="
echo "Step 5: Training AlphaStar Model"
echo "=========================================="
echo "NOTE: This is a minimal training run for testing."
echo "Full training requires significant compute resources."
echo ""

# Configuration
ALPHASTAR_DIR="${HOME}/projects/alphastar"
CONFIG_FILE="${ALPHASTAR_DIR}/alphastar/unplugged/configs/alphastar_supervised.py"

cd "${ALPHASTAR_DIR}"

# Set PYTHONPATH
export PYTHONPATH="${ALPHASTAR_DIR}:${PYTHONPATH}"

# Check if paths.py exists
if [ ! -f "alphastar/unplugged/data/paths.py" ]; then
    echo "Error: paths.py not found. Run script 4 first."
    exit 1
fi

# Minimal training configuration
echo "Starting training with minimal configuration..."
echo "This will train on a very small number of frames for testing."
echo ""

.venv/bin/python alphastar/unplugged/scripts/train.py \
    --config="${CONFIG_FILE}:alphastar.dummy" \
    --config.train.max_number_of_frames=1000 \
    --config.train.learner_kwargs.batch_size=2 \
    --config.train.datasource.kwargs.shuffle_buffer_size=16 \
    --config.train.optimizer_kwargs.lr_frames_before_decay=100 \
    --config.train.learner_kwargs.unroll_len=3 \
    --logtostderr

echo ""
echo "✓ Training complete (or encountered expected errors due to missing JAX/Acme dependencies)"
echo ""
echo "NOTE: Full training requires:"
echo "  - JAX with GPU support"
echo "  - dm-acme (x86_64 Linux only)"
echo "  - Significant computational resources"
echo "  - Many more training frames"
