#!/bin/bash
# Script 4: Setup AlphaStar training configuration

set -e

echo "=========================================="
echo "Step 4: Setting Up Training Configuration"
echo "=========================================="

# Configuration
ALPHASTAR_DIR="${HOME}/projects/alphastar"
CONVERTED_DIR="${HOME}/sc2_replays/test/converted"

cd "${ALPHASTAR_DIR}"

# Create paths.py
echo "Creating paths.py configuration..."
cat > alphastar/unplugged/data/paths.py <<EOF
"""Paths configuration for AlphaStar training data."""

import os

BASE_PATH = os.path.expanduser('~/sc2_replays/test')

# A mapping where the keys are (replay_versions, data_split, player_min_mmr)
# tuples, and the values are glob expressions specifying where the corresponding
# converted replay files are located relative to BASE_PATH.
RELATIVE_PATHS = {
    # Using all available tfrecords for training
    # MMR set to 0 to include all skill levels
    (('4.9.2',), 'train', 0): 'converted/**/*.tfrecord',
    (('4.9.2',), 'test', 0): 'converted/**/*.tfrecord',
    (('4.9.2',), 'debug', 0): 'converted/**/*.tfrecord',
}
EOF

echo ""
echo "✓ Configuration created!"
echo "Created: alphastar/unplugged/data/paths.py"
echo ""
echo "Contents:"
cat alphastar/unplugged/data/paths.py
