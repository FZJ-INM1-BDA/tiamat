"""
Protocol for readers.
"""

from typing import Protocol

import numpy as np

from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class ImageReader(Protocol):
    def read_image(self, accessor: ImageAccessor) -> np.ndarray: ...

    def read_metadata(self) -> ImageMetadata: ...

    @classmethod
    def check_file(cls, fname) -> bool | int | float: ...
