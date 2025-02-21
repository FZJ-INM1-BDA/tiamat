"""
Protocol for readers.
"""
from typing import Protocol
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class ImageReader(Protocol):
    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        ...

    def read_metadata(self) -> ImageMetadata:
        ...

    def check_file(self) -> bool:
        ...
