"""
Protocol for transformers.
"""
from typing import Protocol


class Transformer(Protocol):
    def transform_coordinates(self, x, y, scale, metadata=None, z=None, c=None):
        ...

    def transform_image(self, image, metadata=None):
        ...
