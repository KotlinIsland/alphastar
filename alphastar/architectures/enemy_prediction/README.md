# Enemy Prediction Network

A parallel neural network that predicts hidden enemy unit locations, types, and provides calibrated confidence scores. This augments the main AlphaStar agent with additional information about enemy positions behind fog-of-war.

## Overview

The Enemy Prediction Network processes the same observations as the main AlphaStar architecture but focuses on predicting where enemy units are likely to be, even when they're not visible. The predictions are then merged with real observations to provide richer input to the main agent.

### Key Features

- **Parallel Architecture**: Processes observations independently with its own encoders and LSTM
- **Multi-Output Prediction**: Predicts location, unit type, and existence for multiple units
- **Calibrated Confidence**: Learns to output confidence scores that match actual accuracy
- **Seamless Integration**: Predictions merge with real observations as additional unit features

## Architecture

```
Observable State
    ↓
┌─────────────────────────────────┐
│  Enemy Prediction Network        │
│                                  │
│  ┌──────────────────────────┐   │
│  │ Visual Encoder           │   │
│  │ (minimap, visibility)    │   │
│  └──────────────────────────┘   │
│              ↓                   │
│  ┌──────────────────────────┐   │
│  │ Vector Encoder           │   │
│  │ (game state, timing)     │   │
│  └──────────────────────────┘   │
│              ↓                   │
│  ┌──────────────────────────┐   │
│  │ Unit Encoder             │   │
│  │ (observed units)         │   │
│  └──────────────────────────┘   │
│              ↓                   │
│  ┌──────────────────────────┐   │
│  │ LSTM (temporal tracking) │   │
│  └──────────────────────────┘   │
│              ↓                   │
│  ┌──────────────────────────┐   │
│  │ Prediction Heads         │   │
│  │ • Location (x, y)        │   │
│  │ • Unit Type              │   │
│  │ • Existence              │   │
│  │ • Confidence             │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
    ↓
Predicted Enemy Units
    ↓
[Merge with Real Observations]
    ↓
Augmented Unit Features
    ↓
Main AlphaStar Architecture
```

## Usage

### 1. Basic Prediction

```python
from alphastar.architectures.enemy_prediction import EnemyPredictionNetwork
import haiku as hk

# Create network
def forward_fn(inputs):
    network = EnemyPredictionNetwork(
        obs_spec=obs_spec,
        action_spec=action_spec,
        max_predicted_units=50,
        hidden_size=256,
        lstm_size=256
    )
    outputs, _ = network._forward(inputs)
    return outputs

# Transform and initialize
network = hk.without_apply_rng(hk.transform(forward_fn))
params = network.init(rng, sample_input)

# Run prediction
predictions = network.apply(params, observations)
# predictions contains:
# - 'predicted_units_locations': [50, 2]
# - 'predicted_units_types': [50]
# - 'predicted_units_confidence': [50]
# - 'predicted_units_exists': [50]
```

### 2. Merge with Real Observations

```python
from alphastar.architectures.enemy_prediction import merge_predicted_units

# Get predictions
predictions = network.apply(params, observations)

# Merge with real units
augmented_units = merge_predicted_units(
    raw_units=observations['observation', 'raw_units'],
    predicted_locations=predictions['predicted_units_locations'],
    predicted_types=predictions['predicted_units_types'],
    predicted_confidence=predictions['predicted_units_confidence'],
    predicted_exists=predictions['predicted_units_exists'],
    existence_threshold=0.5
)

# augmented_units now has 2 extra features:
# - is_predicted: 0 for real units, 1 for predicted
# - confidence: 1.0 for real units, [0,1] for predicted

# Pass to main architecture
outputs = main_network.apply(main_params, {'observation': {'raw_units': augmented_units}})
```

### 3. Training

```python
from alphastar.architectures.enemy_prediction.losses import enemy_prediction_loss

# Forward pass
predictions = network.apply(params, inputs)

# Compute loss
ground_truth = {
    'locations': true_enemy_locations,  # [batch, max_units, 2]
    'types': true_enemy_types,          # [batch, max_units]
    'exists': true_enemy_exists,        # [batch, max_units]
}

total_loss, loss_dict = enemy_prediction_loss(
    predictions, ground_truth,
    location_weight=1.0,
    type_weight=1.0,
    existence_weight=1.0,
    calibration_weight=0.5
)

# loss_dict contains:
# - 'location_loss': MSE for positions
# - 'type_loss': Cross-entropy for unit types
# - 'existence_loss': BCE for existence
# - 'calibration_loss': MSE between confidence and accuracy
```

## Training Strategy

### Data Preparation

The network is trained on replays with fog-of-war **enabled** (realistic games), but with access to ground truth enemy positions:

```python
# From replay:
observed_state = replay.get_observation()  # With fog-of-war
true_enemies = replay.get_ground_truth()   # Hidden enemy units

# Training batch:
inputs = observed_state
labels = true_enemies
```

### Loss Functions

The training uses **multi-objective supervised learning**:

1. **Location Loss** (MSE): Minimize distance between predicted and true positions
2. **Type Loss** (Cross-Entropy): Correctly classify unit types
3. **Existence Loss** (BCE): Predict whether a unit exists at that position
4. **Calibration Loss** (MSE): Match confidence to actual accuracy

```
total_loss = location_loss + type_loss + existence_loss + λ * calibration_loss
```

### Confidence Calibration

The calibration loss encourages the network to output:
- **High confidence** when predictions are likely correct
- **Low confidence** when uncertain

This is achieved by penalizing the difference between predicted confidence and actual accuracy:

```
calibration_loss = (confidence - actual_accuracy)²
```

Where `actual_accuracy` is computed from:
- Location accuracy (inverse of distance error)
- Type accuracy (correct/incorrect)
- Existence accuracy (correct/incorrect)

## Visualization

Real-time visualization of predictions:

```python
from alphastar.visualization.prediction_visualizer import PredictionVisualizer

visualizer = PredictionVisualizer(window_size=(800, 800))

for step in training_loop:
    predictions = network.apply(params, inputs)

    visualizer.update(
        predictions=predictions,
        ground_truth=ground_truth,
        minimap=observations['minimap_visibility_map']
    )
```

**Controls:**
- `G`: Toggle ground truth display
- `P`: Toggle predictions display
- `UP/DOWN`: Adjust confidence threshold
- `SPACE`: Pause
- `Q`: Quit

## Integration with Main Architecture

### Option A: Input-Level Augmentation (Recommended)

Inject predicted units at the input level:

```python
# 1. Get predictions
predicted_units = prediction_network.apply(pred_params, obs)

# 2. Merge with observations
augmented_units = merge_predicted_units(
    raw_units=obs['raw_units'],
    predicted_locations=predicted_units['predicted_units_locations'],
    # ...
)

# 3. Pass to main network
obs_augmented = obs.copy()
obs_augmented['raw_units'] = augmented_units

actions = main_network.apply(main_params, obs_augmented)
```

### Option B: Feature-Level Augmentation

Inject at intermediate feature level (requires modifying main architecture).

## Training Script

```bash
python examples/train_enemy_prediction.py \
    --tfrecord_dir=/path/to/replays \
    --checkpoint_dir=./checkpoints \
    --batch_size=32 \
    --learning_rate=1e-4 \
    --max_predicted_units=50 \
    --visualize
```

## Performance Considerations

- **Overhead**: Minimal during inference (~5-10% additional compute)
- **Memory**: +2 features per unit (is_predicted, confidence)
- **Accuracy**: Depends on training data quality and enemy predictability

## Future Improvements

1. **Ensemble Uncertainty**: Use multiple forward passes for better confidence estimates
2. **Attention Mechanisms**: Better temporal tracking of enemy movements
3. **Multi-Scale Predictions**: Predict at different spatial resolutions
4. **Adversarial Training**: Train against agents that try to be unpredictable
5. **Meta-Learning**: Adapt quickly to opponent play styles

## References

- Calibrated confidence: Guo et al., "On Calibration of Modern Neural Networks" (2017)
- Temporal modeling: Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- StarCraft II: Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning" (2019)
