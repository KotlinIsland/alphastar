"""Utilities for capturing intermediate layer activations."""

from typing import Dict, List, Optional
import jax.numpy as jnp
import chex


class ActivationCapture:
  """Captures and stores intermediate activations from network forward pass."""

  def __init__(self, layer_names: Optional[List[str]] = None):
    """Initialize activation capture.

    Args:
      layer_names: List of layer names to capture. If None, captures all.
    """
    self._layer_names = set(layer_names) if layer_names else None
    self._activations: Dict[str, chex.Array] = {}
    self._enabled = True

  def capture(self, layer_name: str, activation: chex.Array):
    """Capture an activation from a layer.

    Args:
      layer_name: Name of the layer producing this activation.
      activation: The activation tensor to capture.
    """
    if not self._enabled:
      return

    if self._layer_names is None or layer_name in self._layer_names:
      # Store a copy to avoid issues with mutation
      self._activations[layer_name] = jnp.array(activation)

  def get_activations(self) -> Dict[str, chex.Array]:
    """Get all captured activations.

    Returns:
      Dictionary mapping layer names to activation tensors.
    """
    return dict(self._activations)

  def clear(self):
    """Clear all captured activations."""
    self._activations.clear()

  def enable(self):
    """Enable activation capture."""
    self._enabled = True

  def disable(self):
    """Disable activation capture."""
    self._enabled = False

  @property
  def enabled(self) -> bool:
    """Whether capture is currently enabled."""
    return self._enabled
