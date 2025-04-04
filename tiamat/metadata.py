"""
Metadata on images.
"""

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from itertools import repeat, product

# Image types
IMAGE_TYPE_SEGMENTATION = (
    "segmentation"  # an image with discrete values, e.g., a mask or a segmentation.
)
IMAGE_TYPE_IMAGE = (
    "image"  # an image with continuous values (e.g., a microscopy image).
)
IMAGE_TYPE_VECTOR = (
    "vector"  # an image with continuous vector values (e.g., a deformation field).
)

def get_dtype_limits(dtype):
    """Returns the min and max values for a given dtype.

    Supports numpy dtypes (integer or float)

    Args:
        dtype: numpy dtype

    Returns:
        tuple: min and max value of the dtype.
    """

    dtype = np.dtype(dtype)

    if issubclass(dtype.type, np.integer):
        info = np.iinfo(dtype)
    elif issubclass(dtype.type, np.floating):
        info = np.finfo(dtype)
    else:
        raise RuntimeError(f"{dtype} is not a valid dtype")
    return info.min, info.max


@dataclass
class ImageMetadata:
    """
    Metadata of an image.
    """

    image_type: str
    shape: tuple
    value_range: tuple
    dtype: np.dtype
    file_path: str | None = None
    spacing: float | tuple[float, ...] | None = None
    channel_dimension: int | None = None  # None means no channel dimension
    additional_metadata: dict | None = None
    scales: float | int | Iterable[float | int] | None = None

    @property
    def spatial_dimensions(self):
        """
        Returns indices of spatial axes, excluding channels
        """
        ch_dims = [self.channel_dimension] if self.channel_dimension is not None else []
        return sorted(list(set(range(len(self.shape))) - set(ch_dims)))
    
    @property
    def spatial_shape(self):
        return tuple(self.shape[i] for i in self.spatial_dimensions)
    
    @property
    def num_channels(self):
        if self.channel_dimension is None:
            return 0
        return self.shape[self.channel_dimension]

    @property
    def extents(self):
        """
        Extents of the image. Provided as a list of tuple of coordinates.
        Extrapolated by shape. Can be set, and will then return the set value.
        Delete the property to revert to the default behavior.
        """
        if hasattr(self, "_extents"):
            return self._extents
        extents = list(zip(repeat(0), self.shape))
        return list(product(*extents))

    @extents.setter
    def extents(self, val):
        self._extents = val

    @extents.deleter
    def extents(self):
        delattr(self, "_extents")
