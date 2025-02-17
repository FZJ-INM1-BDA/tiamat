"""
IO objects.
"""
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
    c: tuple[int | float | None, int | float | None] | int = None
    # scale/spacing to retrieve
    scale: float = 1.0
    spacing: float = None
    # scale/spacing of coordinates
    coordinate_scale: float = 1.0
    coordinate_spacing: float = 1.0
    metadata: ImageMetadata = None
    interpolation: int = None
    # Wether to apply Gauss smothing, default is False
    anti_aliasing: bool = False
    # (Maybe not needed) Std of Gauss filter, default is (s - 1) / 2:
    # anti_aliasing_sigma: float = None
    history: dict = field(default_factory=dict)


@dataclass
class ImageResult:
    image: np.ndarray
    accessor: ImageAccessor
    metadata: ImageMetadata = None
