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

"""Loss functions for enemy prediction network training."""

from typing import Dict, Tuple
import chex
import jax
import jax.numpy as jnp


def enemy_prediction_loss(
    predictions: Dict[str, chex.Array],
    ground_truth: Dict[str, chex.Array],
    location_weight: float = 1.0,
    type_weight: float = 1.0,
    existence_weight: float = 1.0,
    calibration_weight: float = 0.5
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
  """Compute total loss for enemy prediction network.

  This implements a multi-objective loss that:
  1. Maximizes accuracy of predictions (primary objective)
  2. Calibrates confidence to match actual accuracy (secondary objective)

  Args:
    predictions: Dictionary containing:
      - 'locations': [max_units, 2] predicted positions
      - 'type_logits': [max_units, num_types] logits for unit types
      - 'exists_logits': [max_units] logits for existence
      - 'confidence': [max_units] confidence scores in [0, 1]
    ground_truth: Dictionary containing:
      - 'locations': [max_units, 2] true positions
      - 'types': [max_units] true unit types
      - 'exists': [max_units] whether unit exists (0 or 1)
    location_weight: Weight for location loss.
    type_weight: Weight for type classification loss.
    existence_weight: Weight for existence prediction loss.
    calibration_weight: Weight for confidence calibration loss.

  Returns:
    Tuple of (total_loss, loss_dict) where loss_dict contains individual losses.
  """
  # 1. Location loss (MSE for existing units)
  loc_loss = location_loss(
      predictions['locations'],
      ground_truth['locations'],
      ground_truth['exists']
  )

  # 2. Unit type loss (cross-entropy for existing units)
  type_loss = unit_type_loss(
      predictions['type_logits'],
      ground_truth['types'],
      ground_truth['exists']
  )

  # 3. Existence loss (binary cross-entropy)
  exists_loss = existence_loss(
      predictions['exists_logits'],
      ground_truth['exists']
  )

  # 4. Confidence calibration loss
  # Compute actual accuracy per prediction
  actual_accuracy = compute_prediction_accuracy(
      predictions, ground_truth
  )
  calib_loss = calibration_loss(
      predictions['confidence'],
      actual_accuracy,
      ground_truth['exists']
  )

  # Combine losses
  total_loss = (
      location_weight * loc_loss +
      type_weight * type_loss +
      existence_weight * exists_loss +
      calibration_weight * calib_loss
  )

  loss_dict = {
      'total_loss': total_loss,
      'location_loss': loc_loss,
      'type_loss': type_loss,
      'existence_loss': exists_loss,
      'calibration_loss': calib_loss,
  }

  return total_loss, loss_dict


def location_loss(
    predicted_locations: chex.Array,
    true_locations: chex.Array,
    exists_mask: chex.Array
) -> chex.Array:
  """Compute MSE loss for unit locations.

  Args:
    predicted_locations: [max_units, 2] predicted (x, y) in [0, 1].
    true_locations: [max_units, 2] true (x, y) in [0, 1].
    exists_mask: [max_units] boolean mask for existing units.

  Returns:
    Scalar loss value.
  """
  chex.assert_equal_shape([predicted_locations, true_locations])
  chex.assert_rank(exists_mask, 1)

  # Squared error per unit
  squared_error = jnp.sum((predicted_locations - true_locations) ** 2, axis=-1)

  # Only compute loss for units that actually exist
  masked_error = squared_error * exists_mask

  # Average over existing units
  num_existing = jnp.maximum(jnp.sum(exists_mask), 1.0)
  return jnp.sum(masked_error) / num_existing


def unit_type_loss(
    type_logits: chex.Array,
    true_types: chex.Array,
    exists_mask: chex.Array
) -> chex.Array:
  """Compute cross-entropy loss for unit type classification.

  Args:
    type_logits: [max_units, num_types] logits for unit types.
    true_types: [max_units] true unit type indices.
    exists_mask: [max_units] boolean mask for existing units.

  Returns:
    Scalar loss value.
  """
  chex.assert_rank(type_logits, 2)
  chex.assert_rank(true_types, 1)
  chex.assert_rank(exists_mask, 1)

  # Cross-entropy loss per unit
  log_probs = jax.nn.log_softmax(type_logits, axis=-1)
  num_types = type_logits.shape[-1]

  # One-hot encode true types
  true_types_onehot = jax.nn.one_hot(true_types, num_types)

  # Negative log likelihood
  nll = -jnp.sum(log_probs * true_types_onehot, axis=-1)

  # Only compute loss for units that actually exist
  masked_nll = nll * exists_mask

  # Average over existing units
  num_existing = jnp.maximum(jnp.sum(exists_mask), 1.0)
  return jnp.sum(masked_nll) / num_existing


def existence_loss(
    exists_logits: chex.Array,
    true_exists: chex.Array
) -> chex.Array:
  """Compute binary cross-entropy for unit existence.

  Args:
    exists_logits: [max_units] logits for existence.
    true_exists: [max_units] true existence (0 or 1).

  Returns:
    Scalar loss value.
  """
  chex.assert_equal_shape([exists_logits, true_exists])

  # Binary cross-entropy
  bce = jnp.maximum(exists_logits, 0) - exists_logits * true_exists + \
        jnp.log(1 + jnp.exp(-jnp.abs(exists_logits)))

  return jnp.mean(bce)


def calibration_loss(
    confidence: chex.Array,
    actual_accuracy: chex.Array,
    exists_mask: chex.Array
) -> chex.Array:
  """Compute calibration loss to match confidence with actual accuracy.

  This encourages the network to output confidence ≈ accuracy, which means:
  - High confidence when predictions are likely correct
  - Low confidence when predictions are uncertain

  Args:
    confidence: [max_units] predicted confidence in [0, 1].
    actual_accuracy: [max_units] computed accuracy in [0, 1].
    exists_mask: [max_units] boolean mask for existing units.

  Returns:
    Scalar loss value (MSE between confidence and accuracy).
  """
  chex.assert_equal_shape([confidence, actual_accuracy, exists_mask])

  # Squared error between confidence and actual accuracy
  calibration_error = (confidence - actual_accuracy) ** 2

  # Only compute for existing units
  masked_error = calibration_error * exists_mask

  # Average over existing units
  num_existing = jnp.maximum(jnp.sum(exists_mask), 1.0)
  return jnp.sum(masked_error) / num_existing


def compute_prediction_accuracy(
    predictions: Dict[str, chex.Array],
    ground_truth: Dict[str, chex.Array]
) -> chex.Array:
  """Compute actual accuracy for each prediction.

  This computes a combined accuracy score based on:
  - Location accuracy (inverse of distance error)
  - Type accuracy (correct/incorrect)
  - Existence accuracy (correct/incorrect)

  Args:
    predictions: Dictionary with predictions.
    ground_truth: Dictionary with ground truth.

  Returns:
    [max_units] accuracy score in [0, 1] for each prediction.
  """
  # Location accuracy (1 - normalized_distance)
  location_error = jnp.sqrt(jnp.sum(
      (predictions['locations'] - ground_truth['locations']) ** 2,
      axis=-1
  ))
  # Normalize by diagonal of unit square (sqrt(2))
  location_accuracy = jnp.maximum(0.0, 1.0 - location_error / jnp.sqrt(2.0))

  # Type accuracy (binary: correct or not)
  predicted_types = jnp.argmax(predictions['type_logits'], axis=-1)
  type_accuracy = (predicted_types == ground_truth['types']).astype(jnp.float32)

  # Existence accuracy
  predicted_exists = (jax.nn.sigmoid(predictions['exists_logits']) > 0.5).astype(jnp.float32)
  existence_accuracy = (predicted_exists == ground_truth['exists']).astype(jnp.float32)

  # Combined accuracy (average of all components)
  # Only compute for actually existing units
  exists_mask = ground_truth['exists']
  combined_accuracy = (
      location_accuracy * 0.4 +
      type_accuracy * 0.4 +
      existence_accuracy * 0.2
  )

  # For non-existing units, accuracy is just existence accuracy
  combined_accuracy = jnp.where(
      exists_mask > 0,
      combined_accuracy,
      existence_accuracy
  )

  return combined_accuracy


def expected_calibration_error(
    confidence: chex.Array,
    actual_accuracy: chex.Array,
    exists_mask: chex.Array,
    num_bins: int = 10
) -> chex.Array:
  """Compute Expected Calibration Error (ECE) metric.

  ECE measures the difference between predicted confidence and actual accuracy
  across different confidence bins. Lower is better.

  Args:
    confidence: [max_units] predicted confidence.
    actual_accuracy: [max_units] actual accuracy.
    exists_mask: [max_units] mask for existing units.
    num_bins: Number of confidence bins.

  Returns:
    Scalar ECE value.
  """
  # Filter to existing units only
  valid_confidence = confidence * exists_mask
  valid_accuracy = actual_accuracy * exists_mask
  num_valid = jnp.sum(exists_mask)

  # Create bins
  bin_boundaries = jnp.linspace(0, 1, num_bins + 1)

  ece = 0.0
  for i in range(num_bins):
    # Find predictions in this bin
    in_bin = jnp.logical_and(
        valid_confidence >= bin_boundaries[i],
        valid_confidence < bin_boundaries[i + 1]
    )
    in_bin = jnp.logical_and(in_bin, exists_mask)

    num_in_bin = jnp.sum(in_bin)

    if num_in_bin > 0:
      # Average confidence and accuracy in this bin
      avg_confidence = jnp.sum(valid_confidence * in_bin) / num_in_bin
      avg_accuracy = jnp.sum(valid_accuracy * in_bin) / num_in_bin

      # Weighted contribution to ECE
      ece += (num_in_bin / num_valid) * jnp.abs(avg_confidence - avg_accuracy)

  return ece
