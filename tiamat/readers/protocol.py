"""
Protocol for readers.
"""

from typing import Protocol, Union
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata

CheckResult = Union[bool, int, float]

class ImageReader(Protocol):
    """
    Interface for image readers used to load image data and metadata.
    """
    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        """
        Read and return the image content.

        Args:
            accessor (ImageAccessor): Accessor defining which part of the image to read.

        Returns:
            ImageResult: The resulting image data.
        """
        ...

    def read_metadata(self) -> ImageMetadata:
        """
        Read and return metadata associated with the image.

        Returns:
            ImageMetadata: The metadata of the image.
        """
    ...

    @classmethod
    def check_file(cls, fname:str) -> CheckResult:
        """
        Check if the reader is compatible with the given file.

        Args:
            fname (str): The file path or identifier.

        Returns:
            CheckResult: True (0), or a priority (int/float) if compatible, or False/<0> if not.
        """
        ...
