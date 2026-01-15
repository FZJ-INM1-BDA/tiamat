"""
Protocol for readers.
"""

from typing import Protocol

import numpy as np

from ..io import ImageAccessor
from ..metadata import ImageMetadata

CheckResult = bool | int | float


class ImageReader(Protocol):
    """
    Protocol for reading image data and metadata from a source.

    Expected call order in a pipeline:
        1. `read_metadata` → supplies metadata to the first transformer's `transform_metadata`
        2. `read_image` → supplies image data to the first transformer's `transform_image`
    """

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        """
        Read and return the image content.

        Args:
            accessor (ImageAccessor): Accessor defining which part of the image to read.

        Returns:
            np.ndarray: The resulting image data as a Numpy array.
        """
        raise NotImplementedError

    def read_metadata(self) -> ImageMetadata:
        """
        Read and return metadata associated with the image.

        Returns:
            ImageMetadata: The metadata of the image containing shape, spatial and channel dimensions, spacing,
            and other properties.
        """

        raise NotImplementedError

    @classmethod
    def check_file(cls, fname: str) -> bool | int | float:
        """
        Check if the reader is compatible with the given file.

        Args:
            fname (str): The file path or identifier.

        Returns:
             Truthy value if the file is supported. Can return a numeric
            score indicating priority among multiple readers.
        """
        raise NotImplementedError
