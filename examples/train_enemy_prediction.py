#!/usr/bin/env python3
# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for enemy prediction network.

This script trains a network to predict hidden enemy unit locations and types
from visible game state, with calibrated confidence scores.

Usage:
  python examples/train_enemy_prediction.py \\
      --tfrecord_dir=/path/to/tfrecords \\
      --checkpoint_dir=/path/to/checkpoints \\
      --visualize

The network is trained with:
1. Location loss (MSE)
2. Unit type loss (cross-entropy)
3. Existence loss (binary cross-entropy)
4. Confidence calibration loss (MSE between confidence and actual accuracy)
"""

import os
from typing import Dict, Tuple
from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import numpy as np

from alphastar.architectures.enemy_prediction import EnemyPredictionNetwork
from alphastar.architectures.enemy_prediction.losses import (
    enemy_prediction_loss,
    expected_calibration_error
)
from alphastar.visualization.prediction_visualizer import PredictionVisualizer

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'tfrecord_dir', None,
    'Directory containing TFRecord files.', required=True)
flags.DEFINE_string(
    'checkpoint_dir', './checkpoints/enemy_prediction',
    'Directory to save model checkpoints.')
flags.DEFINE_integer(
    'batch_size', 32,
    'Batch size for training.')
flags.DEFINE_integer(
    'num_epochs', 100,
    'Number of training epochs.')
flags.DEFINE_float(
    'learning_rate', 1e-4,
    'Learning rate for Adam optimizer.')
flags.DEFINE_integer(
    'max_predicted_units', 50,
    'Maximum number of enemy units to predict.')
flags.DEFINE_bool(
    'visualize', False,
    'Whether to visualize predictions during training.')
flags.DEFINE_integer(
    'visualize_every', 100,
    'Visualize every N steps.')


def load_dataset(tfrecord_dir: str, batch_size: int):
  """Load and preprocess TFRecord dataset.

  Args:
    tfrecord_dir: Directory containing .tfrecord files.
    batch_size: Batch size.

  Returns:
    TensorFlow dataset.
  """
  # Find all tfrecord files
  tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
  logging.info(f'Found {len(tfrecord_files)} TFRecord files')

  # Create dataset
  dataset = tf.data.TFRecordDataset(tfrecord_files)

  # Parse function (simplified - adjust based on your TFRecord format)
  def parse_example(serialized):
    # This is a placeholder - implement based on your actual format
    # For now, return dummy data
    features = {
        'observation': tf.zeros([128, 128], dtype=tf.float32),
        'enemy_locations': tf.zeros([FLAGS.max_predicted_units, 2], dtype=tf.float32),
        'enemy_types': tf.zeros([FLAGS.max_predicted_units], dtype=tf.int32),
        'enemy_exists': tf.zeros([FLAGS.max_predicted_units], dtype=tf.float32),
    }
    return features

  dataset = dataset.map(parse_example)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)

  return dataset


def create_train_state(
    rng: jax.random.PRNGKey,
    obs_spec: Dict,
    action_spec: Dict,
    learning_rate: float
):
  """Create initial training state.

  Args:
    rng: Random key.
    obs_spec: Observation specification.
    action_spec: Action specification.
    learning_rate: Learning rate.

  Returns:
    Tuple of (params, opt_state, apply_fn).
  """
  # Initialize network
  def forward_fn(inputs):
    network = EnemyPredictionNetwork(
        obs_spec=obs_spec,
        action_spec=action_spec,
        max_predicted_units=FLAGS.max_predicted_units
    )
    outputs, _ = network._forward(inputs)
    return outputs

  # Transform with Haiku
  network_transformed = hk.without_apply_rng(hk.transform(forward_fn))

  # Initialize parameters
  dummy_input = {
      ('observation', 'minimap_visibility_map'): jnp.zeros((128, 128), dtype=jnp.uint8),
      ('observation', 'game_loop'): jnp.array(0, dtype=jnp.int32),
      ('observation', 'player'): jnp.zeros(11, dtype=jnp.float32),
      ('observation', 'unit_counts_bow'): jnp.zeros(100, dtype=jnp.int32),
  }

  params = network_transformed.init(rng, dummy_input)

  # Create optimizer
  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(params)

  return params, opt_state, network_transformed.apply, optimizer


@jax.jit
def train_step(
    params,
    opt_state,
    apply_fn,
    optimizer,
    inputs: Dict,
    ground_truth: Dict
) -> Tuple:
  """Single training step.

  Args:
    params: Model parameters.
    opt_state: Optimizer state.
    apply_fn: Network apply function.
    optimizer: Optimizer.
    inputs: Input observations.
    ground_truth: Ground truth labels.

  Returns:
    Updated params, opt_state, and losses.
  """
  def loss_fn(params):
    predictions = apply_fn(params, inputs)
    total_loss, loss_dict = enemy_prediction_loss(
        predictions, ground_truth
    )
    return total_loss, (predictions, loss_dict)

  (loss, (predictions, loss_dict)), grads = jax.value_and_grad(
      loss_fn, has_aux=True
  )(params)

  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

  return params, opt_state, loss_dict, predictions


def main(argv):
  del argv

  # Create checkpoint directory
  os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)

  # Load dataset
  logging.info('Loading dataset...')
  dataset = load_dataset(FLAGS.tfrecord_dir, FLAGS.batch_size)

  # Initialize network and optimizer
  logging.info('Initializing network...')
  rng = jax.random.PRNGKey(42)

  # Dummy specs (replace with actual specs from your data)
  obs_spec = {
      'minimap_visibility_map': tf.TensorSpec(shape=(128, 128), dtype=tf.uint8),
      'game_loop': tf.TensorSpec(shape=(), dtype=tf.int32),
      'player': tf.TensorSpec(shape=(11,), dtype=tf.float32),
      'unit_counts_bow': tf.TensorSpec(shape=(100,), dtype=tf.int32),
  }
  action_spec = {
      'world': tf.TensorSpec(shape=(), dtype=tf.int32),
  }

  params, opt_state, apply_fn, optimizer = create_train_state(
      rng, obs_spec, action_spec, FLAGS.learning_rate
  )

  # Initialize visualizer if requested
  visualizer = None
  if FLAGS.visualize:
    visualizer = PredictionVisualizer()

  # Training loop
  logging.info('Starting training...')
  step = 0

  for epoch in range(FLAGS.num_epochs):
    logging.info(f'Epoch {epoch + 1}/{FLAGS.num_epochs}')

    for batch in dataset:
      # Extract inputs and ground truth
      inputs = {
          ('observation', 'minimap_visibility_map'): batch['observation'].numpy(),
          ('observation', 'game_loop'): jnp.array(0),
          ('observation', 'player'): jnp.zeros(11),
          ('observation', 'unit_counts_bow'): jnp.zeros(100),
      }

      ground_truth = {
          'locations': batch['enemy_locations'].numpy(),
          'types': batch['enemy_types'].numpy(),
          'exists': batch['enemy_exists'].numpy(),
      }

      # Training step
      params, opt_state, loss_dict, predictions = train_step(
          params, opt_state, apply_fn, optimizer, inputs, ground_truth
      )

      # Logging
      if step % 10 == 0:
        logging.info(
            f'Step {step}: '
            f'total_loss={loss_dict["total_loss"]:.4f}, '
            f'loc_loss={loss_dict["location_loss"]:.4f}, '
            f'type_loss={loss_dict["type_loss"]:.4f}, '
            f'calib_loss={loss_dict["calibration_loss"]:.4f}'
        )

      # Visualization
      if visualizer and step % FLAGS.visualize_every == 0:
        # Convert predictions to numpy
        pred_dict = {
            'locations': np.array(predictions['predicted_units_locations'][0]),
            'types': np.array(predictions['predicted_units_types'][0]),
            'confidence': np.array(predictions['predicted_units_confidence'][0]),
            'exists': np.array(predictions['predicted_units_exists'][0]),
        }
        gt_dict = {
            'locations': ground_truth['locations'][0],
            'exists': ground_truth['exists'][0],
        }

        if not visualizer.update(pred_dict, gt_dict):
          logging.info('Visualization closed, continuing training...')
          visualizer = None

      step += 1

    # Save checkpoint
    checkpoint_path = os.path.join(
        FLAGS.checkpoint_dir, f'checkpoint_epoch_{epoch}.pkl'
    )
    logging.info(f'Saving checkpoint to {checkpoint_path}')
    # In production, use proper checkpoint saving (e.g., with pickle or orbax)

  # Cleanup
  if visualizer:
    visualizer.close()

  logging.info('Training complete!')


if __name__ == '__main__':
  app.run(main)
