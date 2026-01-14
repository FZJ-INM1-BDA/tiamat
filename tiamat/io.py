"""
IO objects.
"""

from dataclasses import dataclass, field

import numpy as np

from tiamat.metadata import ImageMetadata


@dataclass
class ImageAccessor:
    """
    Defines a region and settings for accessing an image.

    Attributes:
        x, y, z: Coordinates or slices for each dimension. Can be an int, float, or a tuple
                  of (start, stop). Use None for unspecified.
        c: Channel selection. Can be an int, a tuple of (start, stop), or a dictionary
           mapping channel names to int or tuple.
        scale: Scale factor to retrieve the image.
        spacing: Spacing to retrieve the image. If None, original spacing is used.
        coordinate_scale: Scale of coordinates relative to original image.
        coordinate_spacing: Spacing of coordinates relative to original image.
        interpolation: Interpolation strategy (use tiamat.constants.INTERPOLATION_TYPE_* constants).
        anti_aliasing: Whether to apply Gaussian smoothing. Default is False.
        fill_value: Value for out-of-bounds padding. None means no padding.
        history: Dictionary storing history of operations or transformations.
    """

    x: tuple[int | float | None, int | float | None] | int = None
    y: tuple[int | float | None, int | float | None] | int = None
    z: tuple[int | float | None, int | float | None] | int = None
    c: (
        tuple[int | float | None, int | float | None]
        | int
        | dict[str : tuple[int | float | None, int | float | None] | int]
    ) = None
    # scale/spacing to retrieve
    scale: float | tuple[float, ...] = 1.0
    spacing: float | tuple[float, ...] = None
    # scale/spacing of coordinates
    coordinate_scale: float = 1.0
    coordinate_spacing: float = 1.0
    # One of the interpolation types in tiamat.constants
    interpolation: str = None
    # Wether to apply Gauss smothing, default is False:
    anti_aliasing: bool = False
    # (Maybe not needed) Std of Gauss filter, default is (s - 1) / 2:
    # anti_aliasing_sigma: float = None
    # Fill value for out-of-bounds request (padding)
    # Can be set to None for no padding
    fill_value: int | float = None
    history: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"ImageAccessor("
            f"x={self.x}, "
            f"y={self.y}, "
            f"z={self.z}, "
            f"c={self.c},\n"
            f"  scale={self.scale}, "
            f"spacing={self.spacing}, "
            f"coordinate_scale={self.coordinate_scale}, "
            f"coordinate_spacing={self.coordinate_spacing},\n"
            f"  interpolation={self.interpolation}, "
            f"anti_aliasing={self.anti_aliasing}, "
            f"fill_value={self.fill_value}"
            f")"
        )

    def __str__(self):
        return self.__repr__()


@dataclass
class ImageResult:
    """
    Stores an image with its corresponding metadata.
    """

    image: np.ndarray
    metadata: ImageMetadata | None = None
