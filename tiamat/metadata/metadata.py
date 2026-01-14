"""
Metadata on images.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product, repeat

import numpy as np

import tiamat.metadata.dimensions as dimensions

# Image types
IMAGE_TYPE_SEGMENTATION = "segmentation"  # an image with discrete values, e.g., a mask or a segmentation.
IMAGE_TYPE_IMAGE = "image"  # an image with continuous values (e.g., a microscopy image).
IMAGE_TYPE_VECTOR = "vector"  # an image with continuous vector values (e.g., a deformation field).


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
    dimensions: list[str, ...] | tuple[str, ...] = (dimensions.Y, dimensions.X, dimensions.C)
    file_path: Iterable[str] | str | None = None
    spacing: float | tuple[float, ...] | None = None
    additional_metadata: dict = field(default_factory=dict)
    scales: float | int | Iterable[float | int] | None = None

    @property
    def spatial_dimensions(self):
        """
        Returns indices of spatial axes, excluding channels
        """
        return tuple(
            self.dimensions.index(dimension)
            for dimension in dimensions.SPATIAL_DIMENSIONS
            if dimension in self.dimensions
        )

    @property
    def channel_dimensions(self):
        """
        Returns indices of channels, excluding spatial axes
        """
        return tuple(index for index, _ in enumerate(self.shape) if index not in self.spatial_dimensions)

    @property
    def spatial_shape(self):
        return tuple(self.shape[i] for i in self.spatial_dimensions)

    @spatial_shape.setter
    def spatial_shape(self, shape):
        shape_list = list(self.shape)
        for i, ix in enumerate(self.spatial_dimensions):
            shape_list[ix] = shape[i]
        self.shape = tuple(shape_list)

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

    @extents.deleter
    def extents(self):
        delattr(self, "_extents")

    def __repr__(self):
        return (
            f"ImageMetadata(\n"
            f"  image_type={self.image_type},\n"
            f"  shape={self.shape},\n"
            f"  value_range={self.value_range},\n"
            f"  dtype={self.dtype},\n"
            f"  spacing={self.spacing},\n"
            f"  dimensions=({','.join(self.dimensions)}),\n"
            f"  additional_metadata={self.additional_metadata},\n"
            f"  scales={self.scales}\n"
            f")"
        )

    def __str__(self):
        return self.__repr__()
