#!/usr/bin/env python3

"""Example: Real-time visualization of network feature maps during inference.

This script demonstrates how to visualize convolutional feature activations
from AlphaStar's visual encoders while processing replay data.

Usage:
  python examples/visualize_network.py --tfrecord_path=/path/to/data.tfrecord

Controls:
  SPACE: Pause/unpause
  LEFT/RIGHT: Switch between different layers
  Q: Quit
"""

import os
from absl import app
from absl import flags
from absl import logging
from alphastar.visualization import ActivationCapture, FeatureVisualizer
from alphastar.visualization.visual_wrapper import (
    VisualizableFeatureEncoder,
    VisualizableDownscale,
    VisualizableResnet
)
from alphastar.architectures.components import visual
from alphastar import types
import tensorflow as tf
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'tfrecord_path', None,
    'Path to a TFRecord file to visualize.', required=True)
flags.DEFINE_integer(
    'window_width', 1600,
    'Width of visualization window.')
flags.DEFINE_integer(
    'window_height', 900,
    'Height of visualization window.')
flags.DEFINE_integer(
    'feature_map_scale', 4,
    'Scale factor for displaying feature maps.')
flags.DEFINE_string(
    'colormap', 'viridis',
    'Colormap to use: viridis, gray, or hot.')


def create_simple_visual_encoder(activation_capture: ActivationCapture):
  """Create a simple visual processing network with visualization hooks.

  Args:
    activation_capture: ActivationCapture instance for recording activations.

  Returns:
    List of visual components forming a simple encoder.
  """
  # This is a simplified example - adjust based on your actual architecture
  components = []

  # Initial feature encoder for minimap
  components.append(
      VisualizableFeatureEncoder(
          input_name=('observation', 'minimap_height_map'),
          output_name='minimap_encoded',
          input_spatial_size=128,
          input_feature_size=None,
          downscale_factor=2,
          output_features_size=32,
          kernel_size=4,
          activation_capture=activation_capture,
          name='minimap_encoder'
      )
  )

  # Downscale layer
  components.append(
      VisualizableDownscale(
          input_name='minimap_encoded',
          output_name='minimap_downscaled',
          input_spatial_size=64,
          input_features_size=32,
          output_features_size=64,
          downscale_factor=2,
          kernel_size=4,
          activation_capture=activation_capture,
          name='downscale_1'
      )
  )

  # Resnet processing
  components.append(
      VisualizableResnet(
          input_name='minimap_downscaled',
          output_name='minimap_processed',
          input_spatial_size=32,
          input_features_size=64,
          num_resblocks=2,
          activation_capture=activation_capture,
          name='resnet_processor'
      )
  )

  return components


def load_episode_from_tfrecord(tfrecord_path: str):
  """Load a single episode from a TFRecord file.

  Args:
    tfrecord_path: Path to the TFRecord file.

  Returns:
    Dictionary containing episode data.
  """
  logging.info(f'Loading episode from {tfrecord_path}')

  dataset = tf.data.TFRecordDataset([tfrecord_path])

  for raw_record in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    # Extract minimap data (this is simplified - adjust based on actual format)
    # The actual parsing depends on how the data was serialized
    episode_data = {}

    # For demonstration, create dummy data
    # In practice, you'd parse the actual TFRecord format
    logging.warning('Using dummy data - implement actual TFRecord parsing')
    episode_data['minimap_height_map'] = np.random.randint(
        0, 256, size=(100, 128, 128), dtype=np.uint8
    )

    return episode_data

  raise ValueError(f'No data found in {tfrecord_path}')


def main(argv):
  del argv

  # Initialize visualization components
  activation_capture = ActivationCapture()
  visualizer = FeatureVisualizer(
      window_size=(FLAGS.window_width, FLAGS.window_height),
      target_fps=30,
      feature_map_scale=FLAGS.feature_map_scale,
      colormap=FLAGS.colormap
  )

  # Create network components with visualization hooks
  encoder_components = create_simple_visual_encoder(activation_capture)

  # Load episode data
  episode_data = load_episode_from_tfrecord(FLAGS.tfrecord_path)
  num_steps = len(episode_data['minimap_height_map'])

  logging.info(f'Loaded episode with {num_steps} steps')
  logging.info('Starting visualization...')
  logging.info('Controls: SPACE=pause, LEFT/RIGHT=switch layers, Q=quit')

  # Main visualization loop
  step_idx = 0
  while visualizer.is_running and step_idx < num_steps:
    # Prepare input for this timestep
    inputs = types.StreamDict({
        ('observation', 'minimap_height_map'): episode_data['minimap_height_map'][step_idx]
    })

    # Clear previous activations
    activation_capture.clear()

    # Run forward pass through components (simplified)
    # In practice, you'd run the full network forward pass
    current_input = inputs
    for component in encoder_components:
      try:
        # Note: This is simplified - actual usage requires proper initialization
        # and may need haiku transform
        logging.info(f'Processing component: {component.name}')
        # outputs, _ = component._forward(current_input)
        # current_input = outputs
      except Exception as e:
        logging.warning(f'Skipping component due to error: {e}')

    # For demonstration, create dummy activations
    # Remove this when you have actual network running
    dummy_activations = {
        'minimap_encoder_minimap_encoded': np.random.randn(64, 64, 32).astype(np.float32),
        'downscale_1_minimap_downscaled': np.random.randn(32, 32, 64).astype(np.float32),
        'resnet_processor_minimap_processed': np.random.randn(32, 32, 64).astype(np.float32),
    }

    # Update visualization
    if not visualizer.is_paused:
      step_idx += 1

    # Use actual activations if available, otherwise use dummy
    activations = activation_capture.get_activations()
    if not activations:
      activations = dummy_activations

    if not visualizer.update(activations, step=step_idx):
      break

  # Cleanup
  visualizer.close()
  logging.info('Visualization complete')


if __name__ == '__main__':
  app.run(main)
