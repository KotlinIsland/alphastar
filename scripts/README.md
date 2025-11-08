# AlphaStar Training Pipeline Scripts

This directory contains scripts to train an AlphaStar model on replays from the "why" bot.

## Prerequisites

- **macOS** with Apple Silicon or Intel
- **Python 3.12** installed via Homebrew
- **StarCraft II** installed at `/Applications/StarCraft II/`
- **uv** package manager (will be installed by script if missing)
- **why bot** located at `~/projects/sc2-why`

## Quick Start

Run the complete pipeline:

```bash
cd ~/projects/alphastar/scripts
chmod +x *.sh
./run_all.sh
```

The master script will prompt you for each step.

## Individual Steps

### Step 0: Setup Environment

```bash
./0_setup_alphastar.sh
```

Creates virtual environment and installs AlphaStar dependencies using `uv pip`.

**Duration:** ~2-5 minutes

### Step 1: Generate Replays

```bash
./1_generate_replays.sh
```

Runs 3 games of "why" bot vs Computer AI and saves replays to `~/sc2_replays/test/raw/`.

**Duration:** ~10-30 minutes (depends on game length)

### Step 2: Partition Replays

```bash
./2_partition_replays.sh
```

Partitions the replays for parallel processing. Creates partition files in `~/sc2_replays/test/partitions/`.

**Duration:** < 1 minute

### Step 3: Convert to TFRecords

```bash
./3_convert_to_tfrecords.sh
```

⚠️ **Most time-consuming step!** Converts replays to TensorFlow record format by re-rendering them with SC2.

**Duration:** ~30-60 minutes per replay

**Known Issues:**
- May hit SC2 mutex locks if SC2 is already running
- Solution: Kill all SC2 processes or restart machine before running

### Step 4: Setup Training Config

```bash
./4_setup_training.sh
```

Creates `alphastar/unplugged/data/paths.py` configuration file pointing to converted data.

**Duration:** < 1 minute

### Step 5: Train Model

```bash
./5_train_model.sh
```

Trains AlphaStar model on the converted data (minimal configuration for testing).

**Duration:** Varies

**Known Issues:**
- Full training requires JAX, dm-acme, and other dependencies not available for ARM64 macOS
- This step demonstrates the training pipeline but may not complete successfully on Apple Silicon
- For production training, use a Linux x86_64 machine with GPU

## Directory Structure

After running all scripts:

```
~/sc2_replays/test/
├── raw/                    # Original .SC2Replay files
│   ├── why_vs_computer_1.SC2Replay
│   ├── why_vs_computer_2.SC2Replay
│   └── why_vs_computer_3.SC2Replay
├── partitions/             # Partition files
│   └── partition_0
└── converted/              # TFRecord files
    └── *.tfrecord
```

## Troubleshooting

### SC2 Mutex Lock Issues

If Step 3 hangs with "Lock blocking" errors:

```bash
pkill -9 SC2
# Or restart your Mac
```

### ARM64 Compatibility Issues

Some DeepMind packages (dm-acme, dm-reverb, dm-launchpad) don't support ARM64 macOS:

- **Data processing (Steps 0-4):** Should work ✅
- **Full training (Step 5):** May fail ⚠️

**Solution:** Use a Linux x86_64 machine for full training, or wait for ARM64 support.

### Missing Dependencies

If you get import errors:

```bash
cd ~/projects/alphastar
.venv/bin/python -c "import <module_name>"
uv pip install <module_name>
```

## Platform Compatibility

| Step | ARM64 macOS | x86_64 Linux |
|------|-------------|--------------|
| 0-2  | ✅ Works     | ✅ Works      |
| 3    | ⚠️ May work  | ✅ Works      |
| 4    | ✅ Works     | ✅ Works      |
| 5    | ⚠️ Limited   | ✅ Works      |

## What This Demonstrates

This pipeline shows the complete workflow for training an AlphaStar agent:

1. ✅ Replay generation from bot games
2. ✅ Data partitioning for parallel processing
3. ⚠️ Replay-to-TFRecord conversion (works but slow)
4. ✅ Training configuration
5. ⚠️ Model training (limited on ARM64)

Even with just 3 replays, this demonstrates the full pipeline. Production training would use:
- Thousands of replays
- Multiple GPUs/TPUs
- Days/weeks of training time
- x86_64 Linux machines

## Notes

- **3 replays is minimal:** Production AlphaStar was trained on millions of games
- **Training is experimental:** Full AlphaStar training requires significant resources
- **This is a proof-of-concept:** Demonstrates the pipeline, not production training
