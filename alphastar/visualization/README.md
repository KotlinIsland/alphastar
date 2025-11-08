# AlphaStar Feature Map Visualization

Real-time visualization of convolutional neural network activations during training and inference.

## Features

- **Real-time display**: Interactive pygame-based viewer showing feature maps as they're computed
- **Multi-layer visualization**: View different convolutional layers and switch between them
- **Feature plane grid**: See all channels/feature planes at once in a grid layout
- **Interactive controls**: Pause, navigate, and inspect activations frame-by-frame

## Installation

## Quick Start

### 1. Basic Usage with Existing Components

Wrap your visual components with visualization-enabled versions:

```python
from alphastar.visualization import ActivationCapture, FeatureVisualizer
from alphastar.visualization.visual_wrapper import VisualizableFeatureEncoder

# Create activation capture
activation_capture = ActivationCapture()

# Create visualizer
visualizer = FeatureVisualizer(
    window_size=(1600, 900),
    feature_map_scale=4,
    colormap='viridis'
)

# Use visualizable components instead of regular ones
encoder = VisualizableFeatureEncoder(
    input_name='minimap',
    output_name='minimap_encoded',
    input_spatial_size=128,
    input_feature_size=None,
    downscale_factor=2,
    output_features_size=32,
    kernel_size=4,
    activation_capture=activation_capture,
    name='minimap_encoder'
)

# Run your network
outputs, _ = encoder._forward(inputs)

# Update visualization
activations = activation_capture.get_activations()
visualizer.update(activations)
```

### 2. Integration with Training Loop

```python
# In your training loop
for step, batch in enumerate(train_dataset):
    # Clear previous step's activations
    activation_capture.clear()

    # Forward pass (activations captured automatically)
    loss, metrics = train_step(batch)

    # Update visualization every N steps
    if step % 10 == 0:
        activations = activation_capture.get_activations()
        if not visualizer.update(activations, step=step):
            break  # User quit

    if visualizer.is_paused:
        # Wait while paused
        while visualizer.is_paused and visualizer.is_running:
            visualizer.update(activations, step=step)

visualizer.close()
```

### 3. Integration with Replay Processing

```python
from alphastar.visualization import ActivationCapture, FeatureVisualizer

def process_replay_with_visualization(replay_path, network):
    activation_capture = ActivationCapture(
        layer_names=['encoder_1', 'resnet_1', 'downscale_2']  # Optional filter
    )
    visualizer = FeatureVisualizer()

    # Process each frame
    for frame in replay_iterator(replay_path):
        activation_capture.clear()

        # Run network
        output = network(frame, activation_capture=activation_capture)

        # Visualize
        activations = activation_capture.get_activations()
        if not visualizer.update(activations):
            break

    visualizer.close()
```

## Visualization Controls

When the pygame window is open:

- **SPACE**: Pause/unpause the visualization
- **LEFT ARROW**: Switch to previous layer
- **RIGHT ARROW**: Switch to next layer
- **Q**: Quit visualization

## Available Components

All standard visual components have visualizable wrappers:

- `VisualizableFeatureEncoder` - Initial feature encoding
- `VisualizableEmbedding` - Embedding layers
- `VisualizableDownscale` - Downscaling layers
- `VisualizableUpscale` - Upscaling layers
- `VisualizableResnet` - ResNet blocks

## Configuration Options

### FeatureVisualizer

```python
FeatureVisualizer(
    window_size=(1600, 900),      # Pygame window dimensions
    target_fps=30,                 # Target frame rate
    feature_map_scale=4,           # Scale factor for small feature maps
    colormap='viridis'             # 'viridis', 'gray', or 'hot'
)
```

### ActivationCapture

```python
ActivationCapture(
    layer_names=['layer1', 'layer2']  # Optional: only capture specific layers
)
```

## Example Script

See `examples/visualize_network.py` for a complete working example:

```bash
python examples/visualize_network.py \
    --tfrecord_path=/path/to/replay.tfrecord \
    --window_width=1920 \
    --window_height=1080 \
    --feature_map_scale=4 \
    --colormap=viridis
```

## Advanced: Custom Integration

To add visualization to custom components:

```python
from alphastar.visualization.activation_capture import ActivationCapture

class MyCustomLayer:
    def __init__(self, activation_capture=None):
        self.activation_capture = activation_capture

    def forward(self, x):
        # Your computation
        output = self.compute(x)

        # Capture activation
        if self.activation_capture is not None:
            self.activation_capture.capture('my_layer', output)

        return output
```

## Performance Notes

- Visualization adds minimal overhead when disabled
- Use `activation_capture.disable()` during evaluation if not visualizing
- Consider visualizing every N steps during training to maintain speed
- Feature map scale affects rendering performance

## Troubleshooting

**Black/empty feature maps**: May indicate dead neurons or need for different colormap

**Slow visualization**: Reduce `feature_map_scale` or `target_fps`

**"No activations captured"**: Ensure components have `activation_capture` parameter set

**Import errors**: Make sure pygame is installed: `pip install pygame`
