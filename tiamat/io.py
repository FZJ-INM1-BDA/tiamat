"""
IO objects.
"""

from typing import Iterable
from dataclasses import dataclass, field
import numpy as np
from tiamat.metadata import ImageMetadata

# interpolation strategies for rescaling
INTERPOLATION_TYPE_NEAREST = "nearest"
INTERPOLATION_TYPE_LINEAR = "linear"
INTERPOLATION_TYPE_CUBIC = "cubic"
INTERPOLATION_TYPE_AREA = "area"
INTERPOLATION_TYPE_LANCZOS4 = "lanczos4"


@dataclass
class ImageAccessor:
    x: tuple[int | float | None, int | float | None] | int = None
    y: tuple[int | float | None, int | float | None] | int = None
    z: tuple[int | float | None, int | float | None] | int = None
    c: (tuple[int | float | None, int | float | None] | int) | dict[str: tuple[int | float | None, int | float | None] | int] = None
    # scale/spacing to retrieve
    scale: float = 1.0
    spacing: float = None
    # scale/spacing of coordinates
    coordinate_scale: float = 1.0
    coordinate_spacing: float = 1.0
    interpolation: int = None
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
    metadata: ImageMetadata = None
