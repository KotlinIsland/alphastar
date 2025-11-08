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

"""Utilities for merging predicted enemy units with real observations."""

from typing import Dict
import chex
import jax.numpy as jnp
from pysc2.lib.features import FeatureUnit


# Feature indices for unit representation
# We add two new features at the end:
FEATURE_IS_PREDICTED = -2  # Boolean: is this a predicted unit?
FEATURE_CONFIDENCE = -1     # Float: confidence score for predictions


def merge_predicted_units(
    raw_units: chex.Array,
    predicted_locations: chex.Array,
    predicted_types: chex.Array,
    predicted_confidence: chex.Array,
    predicted_exists: chex.Array,
    existence_threshold: float = 0.5
) -> chex.Array:
  """Merge predicted enemy units with real observations.

  This function augments the raw_units observation with predicted enemy units.
  The predicted units are added as additional rows with special markers.

  Args:
    raw_units: Real unit observations [max_units, num_features].
    predicted_locations: Predicted unit locations [max_predicted, 2] (x, y in [0,1]).
    predicted_types: Predicted unit types [max_predicted].
    predicted_confidence: Confidence scores [max_predicted] in [0, 1].
    predicted_exists: Existence probabilities [max_predicted] in [0, 1].
    existence_threshold: Threshold for including a predicted unit (default 0.5).

  Returns:
    Augmented raw_units array with predicted units appended and two extra
    feature columns (is_predicted, confidence).
  """
  chex.assert_rank(raw_units, 2)
  chex.assert_rank(predicted_locations, 2)
  chex.assert_rank(predicted_types, 1)
  chex.assert_rank(predicted_confidence, 1)
  chex.assert_rank(predicted_exists, 1)

  max_units, num_features = raw_units.shape
  max_predicted = predicted_locations.shape[0]

  # Add two extra features to raw units (is_predicted, confidence)
  # Real units have is_predicted=0, confidence=1.0
  real_units_augmented = jnp.concatenate([
      raw_units,
      jnp.zeros((max_units, 1), dtype=raw_units.dtype),  # is_predicted = 0
      jnp.ones((max_units, 1), dtype=raw_units.dtype),   # confidence = 1.0
  ], axis=-1)

  # Create predicted unit entries
  # We need to convert predictions to the raw_units format
  predicted_units = _create_predicted_unit_features(
      predicted_locations=predicted_locations,
      predicted_types=predicted_types,
      predicted_confidence=predicted_confidence,
      predicted_exists=predicted_exists,
      num_features=num_features,
      existence_threshold=existence_threshold
  )

  # Concatenate real and predicted units
  merged_units = jnp.concatenate([real_units_augmented, predicted_units], axis=0)

  return merged_units


def _create_predicted_unit_features(
    predicted_locations: chex.Array,
    predicted_types: chex.Array,
    predicted_confidence: chex.Array,
    predicted_exists: chex.Array,
    num_features: int,
    existence_threshold: float
) -> chex.Array:
  """Convert predictions into raw_units format.

  Args:
    predicted_locations: [max_predicted, 2] in normalized [0,1] coordinates.
    predicted_types: [max_predicted] unit type IDs.
    predicted_confidence: [max_predicted] confidence scores.
    predicted_exists: [max_predicted] existence probabilities.
    num_features: Number of features in original raw_units.
    existence_threshold: Threshold for including units.

  Returns:
    Array of shape [max_predicted, num_features + 2] containing predicted units.
  """
  max_predicted = predicted_locations.shape[0]

  # Initialize with zeros (empty unit representation)
  predicted_units = jnp.zeros((max_predicted, num_features + 2), dtype=jnp.int32)

  # Only include predictions above existence threshold
  include_mask = predicted_exists >= existence_threshold

  # Set unit type
  predicted_units = predicted_units.at[:, FeatureUnit.unit_type].set(
      jnp.where(include_mask, predicted_types, 0)
  )

  # Set position (convert from [0,1] to world coordinates)
  # Assuming world size of 256 (adjust based on actual game settings)
  world_size = 256
  x_coords = (predicted_locations[:, 0] * world_size).astype(jnp.int32)
  y_coords = (predicted_locations[:, 1] * world_size).astype(jnp.int32)

  predicted_units = predicted_units.at[:, FeatureUnit.x].set(
      jnp.where(include_mask, x_coords, 0)
  )
  predicted_units = predicted_units.at[:, FeatureUnit.y].set(
      jnp.where(include_mask, y_coords, 0)
  )

  # Set alliance to Enemy (assuming 4 = enemy in SC2)
  predicted_units = predicted_units.at[:, FeatureUnit.alliance].set(
      jnp.where(include_mask, 4, 0)  # 4 = enemy alliance
  )

  # Set is_predicted flag (last two columns)
  predicted_units = predicted_units.at[:, -2].set(
      jnp.where(include_mask, 1, 0)  # is_predicted = 1
  )

  # Set confidence score
  predicted_units = predicted_units.at[:, -1].set(
      (predicted_confidence * 255).astype(jnp.int32)  # Store as uint8 in [0,255]
  )

  # Set other reasonable defaults for predicted units
  # Health (assume full health - we don't know better)
  predicted_units = predicted_units.at[:, FeatureUnit.health].set(
      jnp.where(include_mask, 255, 0)
  )

  # Visibility (assume 2 = snapshot, since we predicted it but don't see it)
  predicted_units = predicted_units.at[:, FeatureUnit.display_type].set(
      jnp.where(include_mask, 2, 0)
  )

  return predicted_units


def get_augmented_unit_encoder_config(
    base_num_features: int
) -> Dict[str, int]:
  """Get configuration for unit encoder that handles augmented features.

  Args:
    base_num_features: Original number of features in raw_units.

  Returns:
    Dictionary with configuration for augmented units encoder.
  """
  return {
      'num_raw_unit_features': base_num_features + 2,  # Add is_predicted + confidence
      'has_predicted_units': True,
  }


def extract_confidence_scores(augmented_units: chex.Array) -> chex.Array:
  """Extract confidence scores from augmented unit representation.

  Args:
    augmented_units: Augmented units with confidence in last column.

  Returns:
    Confidence scores as float32 in [0, 1].
  """
  # Confidence stored as int32 in [0, 255], convert back to [0, 1]
  confidence_int = augmented_units[:, -1]
  return confidence_int.astype(jnp.float32) / 255.0


def is_predicted_unit(augmented_units: chex.Array) -> chex.Array:
  """Check which units are predicted vs real observations.

  Args:
    augmented_units: Augmented units with is_predicted flag.

  Returns:
    Boolean array indicating which units are predictions.
  """
  return augmented_units[:, -2] > 0
