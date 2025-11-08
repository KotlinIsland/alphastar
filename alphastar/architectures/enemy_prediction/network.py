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

"""Enemy Prediction Network - predicts hidden enemy unit locations and types."""

from typing import Optional, Tuple
from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures.components import common, merge, units, vector, visual
from alphastar.architectures.standard import encoders
import chex
from dm_env import specs
import haiku as hk
import jax.numpy as jnp
import ml_collections


class EnemyPredictionNetwork(modular.BatchedComponent):
  """Parallel network that predicts enemy unit locations and types with confidence.

  This network processes the same observations as the main AlphaStar architecture
  but outputs predictions about hidden enemy units. The predictions are then merged
  with real observations to augment the main network's input.

  Architecture:
    Observable State → [Vector Encoder, Visual Encoder, Units Encoder]
                    → LSTM → Prediction Heads
                    → [Unit Locations, Unit Types, Confidence Scores]
  """

  def __init__(
      self,
      obs_spec: types.ObsSpec,
      action_spec: types.ActionSpec,
      max_predicted_units: int,
      hidden_size: int = 256,
      lstm_size: int = 256,
      visual_features_size: int = 32,
      num_unit_types: int = 256,
      name: Optional[str] = None
  ):
    """Initialize the Enemy Prediction Network.

    Args:
      obs_spec: Observation specification from environment.
      action_spec: Action specification from environment.
      max_predicted_units: Maximum number of enemy units to predict.
      hidden_size: Size of hidden layers.
      lstm_size: Size of LSTM hidden state.
      visual_features_size: Size of visual feature embeddings.
      num_unit_types: Number of possible unit types in the game.
      name: Name of this component.
    """
    super().__init__(name=name or 'enemy_prediction_network')
    self._obs_spec = obs_spec
    self._action_spec = action_spec
    self._max_predicted_units = max_predicted_units
    self._hidden_size = hidden_size
    self._lstm_size = lstm_size
    self._visual_features_size = visual_features_size
    self._num_unit_types = num_unit_types

  @property
  def input_spec(self) -> types.SpecDict:
    # Takes same observations as main architecture
    return types.SpecDict({
        ('observation', k): v for k, v in self._obs_spec.items()
    })

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        'predicted_units_locations': specs.Array(
            (self._max_predicted_units, 2), jnp.float32),
        'predicted_units_types': specs.Array(
            (self._max_predicted_units,), jnp.int32),
        'predicted_units_confidence': specs.Array(
            (self._max_predicted_units,), jnp.float32),
        'predicted_units_exists': specs.Array(
            (self._max_predicted_units,), jnp.float32),
    })

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    """Forward pass through the enemy prediction network.

    Args:
      inputs: Dictionary of input streams from observations.

    Returns:
      Tuple of (outputs, logs) where outputs contains predicted enemy units.
    """
    # 1. Extract key observable features
    # We focus on features that give hints about enemy presence:
    # - Minimap visibility (where we've scouted)
    # - Own units (what we can see)
    # - Game state (timing information)

    # Visual features (minimap)
    minimap_features = self._encode_visual_features(inputs)

    # Vector features (game state, timing)
    vector_features = self._encode_vector_features(inputs)

    # Unit features (what we currently see)
    unit_features = self._encode_unit_features(inputs)

    # 2. Combine all features
    combined = jnp.concatenate([
        minimap_features,
        vector_features,
        unit_features
    ], axis=-1)

    # 3. Process through LSTM to track enemy movements over time
    lstm_output = self._lstm_layer(combined)

    # 4. Prediction heads
    predictions = self._prediction_heads(lstm_output)

    outputs = types.StreamDict({
        'predicted_units_locations': predictions['locations'],
        'predicted_units_types': predictions['types'],
        'predicted_units_confidence': predictions['confidence'],
        'predicted_units_exists': predictions['exists'],
    })

    return outputs, {}

  def _encode_visual_features(self, inputs: types.StreamDict) -> chex.Array:
    """Encode visual features (minimap) into a vector.

    Args:
      inputs: Input stream dictionary.

    Returns:
      Encoded visual features as 1D vector.
    """
    # Simplified visual encoding - in practice, use proper visual encoder
    visibility_map = inputs['observation', 'minimap_visibility_map']

    # Simple downsampling and flattening
    # In production, use proper conv layers
    x = jnp.array(visibility_map, dtype=jnp.float32)
    x = hk.AvgPool(window_shape=8, strides=8, padding='VALID')(
        x[..., jnp.newaxis])
    x = jnp.reshape(x, [-1])
    x = hk.Linear(self._hidden_size)(x)
    x = jax.nn.relu(x)

    return x

  def _encode_vector_features(self, inputs: types.StreamDict) -> chex.Array:
    """Encode scalar/vector features (game state).

    Args:
      inputs: Input stream dictionary.

    Returns:
      Encoded vector features.
    """
    # Game loop (timing information is crucial for predicting enemy tech/army)
    game_loop = inputs['observation', 'game_loop']
    game_loop_embedding = jax.nn.one_hot(
        jnp.minimum(game_loop // 100, 300), 301)

    # Player features
    player_features = inputs['observation', 'player']
    player_features = jnp.array(player_features, dtype=jnp.float32)

    # Combine and project
    x = jnp.concatenate([game_loop_embedding, player_features], axis=-1)
    x = hk.Linear(self._hidden_size)(x)
    x = jax.nn.relu(x)

    return x

  def _encode_unit_features(self, inputs: types.StreamDict) -> chex.Array:
    """Encode observed units into aggregate features.

    Args:
      inputs: Input stream dictionary.

    Returns:
      Encoded unit features.
    """
    # Unit counts give us information about army composition we've seen
    unit_counts = inputs['observation', 'unit_counts_bow']
    unit_counts = jnp.array(unit_counts, dtype=jnp.float32)

    x = hk.Linear(self._hidden_size)(unit_counts)
    x = jax.nn.relu(x)

    return x

  def _lstm_layer(self, features: chex.Array) -> chex.Array:
    """Process features through LSTM to track temporal patterns.

    Args:
      features: Combined feature vector.

    Returns:
      LSTM output.
    """
    # Project to LSTM input size
    x = hk.Linear(self._lstm_size)(features)

    # LSTM core (tracks enemy movements over time)
    lstm = hk.LSTM(self._lstm_size)
    state = lstm.initial_state(batch_size=None)
    output, new_state = lstm(x, state)

    return output

  def _prediction_heads(self, lstm_output: chex.Array) -> dict:
    """Prediction heads for enemy units.

    Args:
      lstm_output: Output from LSTM layer.

    Returns:
      Dictionary with predictions for locations, types, confidence.
    """
    # Shared hidden layer
    hidden = hk.Linear(self._hidden_size)(lstm_output)
    hidden = jax.nn.relu(hidden)

    # Location head (x, y coordinates for each potential enemy unit)
    # Output shape: [max_predicted_units, 2]
    locations = hk.Linear(self._max_predicted_units * 2)(hidden)
    locations = jnp.reshape(locations, [self._max_predicted_units, 2])
    # Normalize to [0, 1] range
    locations = jax.nn.sigmoid(locations)

    # Unit type head (discrete unit type)
    # Output shape: [max_predicted_units, num_unit_types]
    type_logits = hk.Linear(self._max_predicted_units * self._num_unit_types)(hidden)
    type_logits = jnp.reshape(type_logits, [self._max_predicted_units, self._num_unit_types])
    predicted_types = jnp.argmax(type_logits, axis=-1)

    # Existence head (does this unit exist?)
    # Output shape: [max_predicted_units]
    exists_logits = hk.Linear(self._max_predicted_units)(hidden)
    exists_probs = jax.nn.sigmoid(exists_logits)

    # Confidence head (how confident are we about this prediction?)
    # Output shape: [max_predicted_units]
    confidence_logits = hk.Linear(self._max_predicted_units)(hidden)
    confidence = jax.nn.sigmoid(confidence_logits)

    return {
        'locations': locations,
        'types': predicted_types,
        'type_logits': type_logits,  # Keep for loss computation
        'confidence': confidence,
        'exists': exists_probs,
        'exists_logits': exists_logits,  # Keep for loss computation
    }
