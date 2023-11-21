"""
Protocol for readers.
"""
from typing import Protocol


class ImageReader(Protocol):
    def get_crop(self, x, y, scale, z=None, c=None):
        ...

    def get_metadata(self):
        ...
