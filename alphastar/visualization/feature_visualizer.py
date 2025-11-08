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

"""Real-time pygame-based visualization of convolutional feature maps."""

import math
from typing import Dict, Optional, Tuple
import numpy as np
import pygame
import chex


class FeatureVisualizer:
  """Interactive pygame visualizer for convolutional feature maps."""

  def __init__(
      self,
      window_size: Tuple[int, int] = (1600, 900),
      target_fps: int = 30,
      feature_map_scale: int = 2,
      colormap: str = 'viridis'
  ):
    """Initialize the feature visualizer.

    Args:
      window_size: Size of the pygame window (width, height).
      target_fps: Target frames per second for the display.
      feature_map_scale: Scale factor for displaying small feature maps.
      colormap: Colormap to use ('viridis', 'gray', 'hot').
    """
    pygame.init()
    self._window_size = window_size
    self._screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("AlphaStar Feature Map Visualizer")
    self._clock = pygame.time.Clock()
    self._target_fps = target_fps
    self._feature_map_scale = feature_map_scale
    self._colormap = colormap
    self._font = pygame.font.SysFont('monospace', 12)
    self._running = True
    self._paused = False
    self._current_layer_idx = 0
    self._layer_names = []

  def _normalize_feature_map(self, feature_map: np.ndarray) -> np.ndarray:
    """Normalize feature map to [0, 1] range.

    Args:
      feature_map: 2D array representing a single feature plane.

    Returns:
      Normalized feature map.
    """
    fmin, fmax = feature_map.min(), feature_map.max()
    if fmax - fmin < 1e-8:
      return np.zeros_like(feature_map)
    return (feature_map - fmin) / (fmax - fmin)

  def _apply_colormap(self, normalized: np.ndarray) -> np.ndarray:
    """Apply colormap to normalized values.

    Args:
      normalized: Normalized feature map in [0, 1].

    Returns:
      RGB image as uint8 array of shape (H, W, 3).
    """
    if self._colormap == 'gray':
      gray = (normalized * 255).astype(np.uint8)
      return np.stack([gray, gray, gray], axis=-1)
    elif self._colormap == 'hot':
      # Simple hot colormap: black -> red -> yellow -> white
      rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
      rgb[..., 0] = np.clip(normalized * 3 * 255, 0, 255)  # Red
      rgb[..., 1] = np.clip((normalized * 3 - 1) * 255, 0, 255)  # Green
      rgb[..., 2] = np.clip((normalized * 3 - 2) * 255, 0, 255)  # Blue
      return rgb
    else:  # viridis (simplified)
      # Simplified viridis-like colormap
      rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
      rgb[..., 0] = (normalized * 0.3 * 255).astype(np.uint8)  # R
      rgb[..., 1] = (normalized * 0.8 * 255).astype(np.uint8)  # G
      rgb[..., 2] = ((0.5 + normalized * 0.5) * 255).astype(np.uint8)  # B
      return rgb

  def _render_feature_planes(
      self,
      feature_map: chex.Array,
      layer_name: str
  ):
    """Render all feature planes from a convolutional layer.

    Args:
      feature_map: Array of shape (H, W, C) where C is number of channels.
      layer_name: Name of the layer being visualized.
    """
    self._screen.fill((0, 0, 0))

    # Handle different input shapes
    if len(feature_map.shape) == 2:
      # Single channel (H, W)
      feature_map = feature_map[..., np.newaxis]
    elif len(feature_map.shape) == 4:
      # Batched (B, H, W, C) - take first batch
      feature_map = feature_map[0]

    height, width, num_channels = feature_map.shape

    # Calculate grid layout for feature planes
    grid_cols = int(math.ceil(math.sqrt(num_channels)))
    grid_rows = int(math.ceil(num_channels / grid_cols))

    # Calculate size for each feature plane display
    available_width = self._window_size[0] - 20
    available_height = self._window_size[1] - 60  # Leave space for text
    plane_width = min(available_width // grid_cols, width * self._feature_map_scale)
    plane_height = min(available_height // grid_rows, height * self._feature_map_scale)

    # Render each feature plane
    for i in range(num_channels):
      row = i // grid_cols
      col = i % grid_cols

      # Extract and normalize this feature plane
      plane = np.array(feature_map[:, :, i])
      normalized = self._normalize_feature_map(plane)
      colored = self._apply_colormap(normalized)

      # Scale up for visibility
      if plane_width != width or plane_height != height:
        colored = np.repeat(np.repeat(colored,
                                      plane_height // height, axis=0),
                           plane_width // width, axis=1)

      # Convert to pygame surface
      surface = pygame.surfarray.make_surface(
          np.transpose(colored, (1, 0, 2))
      )

      # Position on screen
      x = 10 + col * (plane_width + 5)
      y = 50 + row * (plane_height + 5)
      self._screen.blit(surface, (x, y))

      # Label each plane
      label = self._font.render(f'{i}', True, (200, 200, 200))
      self._screen.blit(label, (x + 2, y + 2))

    # Render header text
    header = f"Layer: {layer_name} | Shape: {feature_map.shape} | Channels: {num_channels}"
    header_text = self._font.render(header, True, (255, 255, 255))
    self._screen.blit(header_text, (10, 10))

    # Render controls
    controls = "SPACE: pause | Q: quit | LEFT/RIGHT: switch layers (if multiple)"
    controls_text = self._font.render(controls, True, (150, 150, 150))
    self._screen.blit(controls_text, (10, 30))

    pygame.display.flip()

  def update(
      self,
      activations: Dict[str, chex.Array],
      step: Optional[int] = None
  ) -> bool:
    """Update the visualization with new activations.

    Args:
      activations: Dictionary mapping layer names to activation tensors.
      step: Optional step number to display.

    Returns:
      True if visualization should continue, False if user quit.
    """
    # Handle pygame events
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self._running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_q:
          self._running = False
        elif event.key == pygame.K_SPACE:
          self._paused = not self._paused
        elif event.key == pygame.K_LEFT:
          self._current_layer_idx = max(0, self._current_layer_idx - 1)
        elif event.key == pygame.K_RIGHT:
          self._current_layer_idx = min(
              len(self._layer_names) - 1, self._current_layer_idx + 1
          )

    if not self._running:
      return False

    # Update layer names list
    self._layer_names = sorted(activations.keys())
    if not self._layer_names:
      return True

    # Clamp current layer index
    self._current_layer_idx = min(self._current_layer_idx, len(self._layer_names) - 1)

    # Render current layer
    if self._layer_names:
      current_layer = self._layer_names[self._current_layer_idx]
      self._render_feature_planes(activations[current_layer], current_layer)

    self._clock.tick(self._target_fps)
    return True

  def close(self):
    """Clean up pygame resources."""
    pygame.quit()

  @property
  def is_running(self) -> bool:
    """Whether the visualizer is still running."""
    return self._running

  @property
  def is_paused(self) -> bool:
    """Whether the visualizer is paused."""
    return self._paused
