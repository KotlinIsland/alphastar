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

"""Visualization for enemy unit predictions."""

from typing import Dict, Optional, Tuple
import numpy as np
import pygame
import chex


class PredictionVisualizer:
  """Visualize predicted enemy units overlaid on game map."""

  def __init__(
      self,
      window_size: Tuple[int, int] = (800, 800),
      map_size: int = 256,
      target_fps: int = 30
  ):
    """Initialize prediction visualizer.

    Args:
      window_size: Size of pygame window (width, height).
      map_size: Size of game map (assumed square).
      target_fps: Target frames per second.
    """
    pygame.init()
    self._window_size = window_size
    self._map_size = map_size
    self._screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Enemy Prediction Visualizer")
    self._clock = pygame.time.Clock()
    self._target_fps = target_fps
    self._font = pygame.font.SysFont('monospace', 10)
    self._running = True
    self._paused = False
    self._show_ground_truth = True
    self._show_predictions = True
    self._confidence_threshold = 0.5

  def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
    """Convert world coordinates to screen coordinates.

    Args:
      x: World x coordinate (0-map_size).
      y: World y coordinate (0-map_size).

    Returns:
      Screen coordinates (sx, sy).
    """
    sx = int(x * self._window_size[0] / self._map_size)
    sy = int(y * self._window_size[1] / self._map_size)
    return sx, sy

  def _draw_unit(
      self,
      x: float,
      y: float,
      color: Tuple[int, int, int],
      radius: int = 5,
      confidence: Optional[float] = None
  ):
    """Draw a unit on the screen.

    Args:
      x: World x coordinate.
      y: World y coordinate.
      color: RGB color tuple.
      radius: Radius of the circle.
      confidence: Optional confidence score to display.
    """
    sx, sy = self._world_to_screen(x, y)

    # Draw circle
    pygame.draw.circle(self._screen, color, (sx, sy), radius)

    # Draw confidence if provided
    if confidence is not None:
      # Draw confidence ring
      ring_radius = int(radius * (1 + confidence))
      pygame.draw.circle(self._screen, color, (sx, sy), ring_radius, 1)

      # Draw confidence text
      conf_text = f"{int(confidence * 100)}%"
      text_surface = self._font.render(conf_text, True, color)
      self._screen.blit(text_surface, (sx + radius + 2, sy - 5))

  def update(
      self,
      predictions: Dict[str, chex.Array],
      ground_truth: Optional[Dict[str, chex.Array]] = None,
      minimap: Optional[np.ndarray] = None
  ) -> bool:
    """Update visualization with new predictions.

    Args:
      predictions: Dictionary containing:
        - 'locations': [N, 2] predicted positions (x, y in world coords)
        - 'types': [N] unit types
        - 'confidence': [N] confidence scores
        - 'exists': [N] existence probabilities
      ground_truth: Optional ground truth for comparison.
      minimap: Optional minimap to display as background.

    Returns:
      True if should continue, False if user quit.
    """
    # Handle events
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self._running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_q:
          self._running = False
        elif event.key == pygame.K_SPACE:
          self._paused = not self._paused
        elif event.key == pygame.K_g:
          self._show_ground_truth = not self._show_ground_truth
        elif event.key == pygame.K_p:
          self._show_predictions = not self._show_predictions
        elif event.key == pygame.K_UP:
          self._confidence_threshold = min(1.0, self._confidence_threshold + 0.1)
        elif event.key == pygame.K_DOWN:
          self._confidence_threshold = max(0.0, self._confidence_threshold - 0.1)

    if not self._running:
      return False

    # Clear screen
    self._screen.fill((30, 30, 30))

    # Draw minimap background if provided
    if minimap is not None:
      self._draw_minimap(minimap)

    # Draw ground truth units (green)
    if ground_truth is not None and self._show_ground_truth:
      self._draw_ground_truth_units(ground_truth)

    # Draw predicted units (red/yellow based on confidence)
    if self._show_predictions:
      self._draw_predicted_units(predictions)

    # Draw UI
    self._draw_ui()

    pygame.display.flip()
    self._clock.tick(self._target_fps)
    return True

  def _draw_minimap(self, minimap: np.ndarray):
    """Draw minimap as background.

    Args:
      minimap: 2D array representing minimap (e.g., visibility, terrain).
    """
    # Normalize and convert to RGB
    minimap_normalized = (minimap - minimap.min()) / (minimap.max() - minimap.min() + 1e-8)
    minimap_rgb = (minimap_normalized * 100).astype(np.uint8)
    minimap_rgb = np.stack([minimap_rgb] * 3, axis=-1)

    # Resize to window size
    minimap_surface = pygame.surfarray.make_surface(
        np.transpose(minimap_rgb, (1, 0, 2))
    )
    minimap_surface = pygame.transform.scale(minimap_surface, self._window_size)
    self._screen.blit(minimap_surface, (0, 0))

  def _draw_ground_truth_units(self, ground_truth: Dict[str, chex.Array]):
    """Draw ground truth enemy units.

    Args:
      ground_truth: Dictionary with 'locations' and 'exists'.
    """
    locations = np.array(ground_truth['locations'])
    exists = np.array(ground_truth['exists'])

    for i in range(len(locations)):
      if exists[i] > 0:
        x, y = locations[i]
        # Convert from normalized [0,1] to world coords
        x = x * self._map_size
        y = y * self._map_size
        self._draw_unit(x, y, color=(0, 255, 0), radius=6)  # Green

  def _draw_predicted_units(self, predictions: Dict[str, chex.Array]):
    """Draw predicted enemy units.

    Args:
      predictions: Dictionary with predictions.
    """
    locations = np.array(predictions['locations'])
    confidence = np.array(predictions['confidence'])
    exists = np.array(predictions['exists'])

    for i in range(len(locations)):
      # Only draw if above existence and confidence thresholds
      if exists[i] > 0.5 and confidence[i] > self._confidence_threshold:
        x, y = locations[i]
        # Convert from normalized [0,1] to world coords
        x = x * self._map_size
        y = y * self._map_size

        # Color based on confidence (red = low, yellow = high)
        conf = confidence[i]
        red = 255
        green = int(255 * conf)
        blue = 0
        color = (red, green, blue)

        self._draw_unit(x, y, color=color, radius=5, confidence=conf)

  def _draw_ui(self):
    """Draw UI elements (legend, controls)."""
    y_offset = 10

    # Legend
    legend_items = [
        ("GREEN: Ground Truth", (0, 255, 0)),
        ("RED-YELLOW: Predictions (confidence)", (255, 200, 0)),
    ]

    for text, color in legend_items:
      text_surface = self._font.render(text, True, color)
      self._screen.blit(text_surface, (10, y_offset))
      y_offset += 15

    # Controls
    y_offset += 10
    controls = [
        f"Threshold: {self._confidence_threshold:.1f} (UP/DOWN)",
        "G: Toggle ground truth",
        "P: Toggle predictions",
        "SPACE: Pause",
        "Q: Quit",
    ]

    for text in controls:
      text_surface = self._font.render(text, True, (150, 150, 150))
      self._screen.blit(text_surface, (10, y_offset))
      y_offset += 15

    # Status
    status = "PAUSED" if self._paused else "RUNNING"
    status_surface = self._font.render(status, True, (255, 255, 0))
    self._screen.blit(status_surface, (self._window_size[0] - 100, 10))

  def close(self):
    """Clean up pygame resources."""
    pygame.quit()

  @property
  def is_running(self) -> bool:
    """Whether visualizer is still running."""
    return self._running

  @property
  def is_paused(self) -> bool:
    """Whether visualizer is paused."""
    return self._paused
