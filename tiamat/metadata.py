"""
Metadata on images.
"""
from dataclasses import dataclass
import numpy as np

# Image types
IMAGE_TYPE_SEGMENTATION = "segmentation"  # an image with discrete values, e.g., a mask or a segmentation.
IMAGE_TYPE_IMAGE = "image"  # an image with continuous values (e.g., a microscopy image).

# Meaning of channel dimension
CHANNEL_INTERPRETATION_COLOR = "color"
CHANNEL_INTERPRETATION_STACK = "stack"


@dataclass
class ImageMetadata:
    """
    Metadata of an image.
    """
    image_type: str
    shape: tuple
    value_range: tuple
    dtype: np.dtype
    spacing: float | tuple[float, ...] = None
    channel_dimension: int = 2
    channel_interpretation: str = CHANNEL_INTERPRETATION_COLOR
    additional_metadata: dict = None
