"""
Protocol for transformers.
"""
from typing import Protocol
from ..io import ImageAccessor, ImageResult


class Transformer(Protocol):
    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        """
        Applies a transformation that affects accessing the image.

        Note: If this method modifies the incoming accessor,
        it has to return a __copy__ of the incoming object.
        The copy can be created using datalcasses.replace.
        """
        ...

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        """
        Transform an incoming image.
        """
        ...
