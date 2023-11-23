"""
IO objects.
"""
from dataclasses import dataclass
import numpy as np
from tiamat.metadata import ImageMetadata

# interpolation strategies for rescaling
INTERPOLATION_TYPE_NEAREST = "nearest"
INTERPOLATION_TYPE_LINEAR = "linear"
INTERPOLATION_TYPE_CUBIC = "cubic"


@dataclass
class ImageAccessor:
    file_name: str
    x: tuple[int | float | None, int | float | None] | int = None
    y: tuple[int | float | None, int | float | None] | int = None
    z: tuple[int | float | None, int | float | None] | int = None
    c: tuple[int | float | None, int | float | None] | int = None
    scale: float = 1.0
    spacing: float = None
    metadata: ImageMetadata = None
    interpolation: int = None
    coordinate_spacing: float = None


@dataclass
class ImageResult:
    image: np.ndarray
    accessor: ImageAccessor
    metadata: ImageMetadata = None
