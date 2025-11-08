"""Wrapper classes that add visualization hooks to visual components."""

from typing import Optional
from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures.components import visual
from alphastar.visualization.activation_capture import ActivationCapture


class VisualizableFeatureEncoder(visual.FeatureEncoder):
  """FeatureEncoder with activation capture for visualization."""

  def __init__(self, *args, activation_capture: Optional[ActivationCapture] = None, **kwargs):
    """Initialize with optional activation capture.

    Args:
      *args: Arguments passed to parent FeatureEncoder.
      activation_capture: Optional ActivationCapture instance for recording outputs.
      **kwargs: Keyword arguments passed to parent FeatureEncoder.
    """
    super().__init__(*args, **kwargs)
    self._activation_capture = activation_capture

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    outputs, logs = super()._forward(inputs)

    # Capture activation if enabled
    if self._activation_capture is not None:
      layer_name = f"{self.name or 'FeatureEncoder'}_{self._output_name}"
      self._activation_capture.capture(layer_name, outputs[self._output_name])

    return outputs, logs


class VisualizableEmbedding(visual.Embedding):
  """Embedding with activation capture for visualization."""

  def __init__(self, *args, activation_capture: Optional[ActivationCapture] = None, **kwargs):
    super().__init__(*args, **kwargs)
    self._activation_capture = activation_capture

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    outputs, logs = super()._forward(inputs)

    if self._activation_capture is not None:
      layer_name = f"{self.name or 'Embedding'}_{self._output_name}"
      self._activation_capture.capture(layer_name, outputs[self._output_name])

    return outputs, logs


class VisualizableDownscale(visual.Downscale):
  """Downscale with activation capture for visualization."""

  def __init__(self, *args, activation_capture: Optional[ActivationCapture] = None, **kwargs):
    super().__init__(*args, **kwargs)
    self._activation_capture = activation_capture

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    outputs, logs = super()._forward(inputs)

    if self._activation_capture is not None:
      layer_name = f"{self.name or 'Downscale'}_{self._output_name}"
      self._activation_capture.capture(layer_name, outputs[self._output_name])

    return outputs, logs


class VisualizableUpscale(visual.Upscale):
  """Upscale with activation capture for visualization."""

  def __init__(self, *args, activation_capture: Optional[ActivationCapture] = None, **kwargs):
    super().__init__(*args, **kwargs)
    self._activation_capture = activation_capture

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    outputs, logs = super()._forward(inputs)

    if self._activation_capture is not None:
      layer_name = f"{self.name or 'Upscale'}_{self._output_name}"
      self._activation_capture.capture(layer_name, outputs[self._output_name])

    return outputs, logs


class VisualizableResnet(visual.Resnet):
  """Resnet with activation capture for visualization."""

  def __init__(self, *args, activation_capture: Optional[ActivationCapture] = None, **kwargs):
    super().__init__(*args, **kwargs)
    self._activation_capture = activation_capture

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    outputs, logs = super()._forward(inputs)

    if self._activation_capture is not None:
      layer_name = f"{self.name or 'Resnet'}_{self._output_name}"
      self._activation_capture.capture(layer_name, outputs[self._output_name])

    return outputs, logs
