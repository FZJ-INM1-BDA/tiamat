"""
Protocol for transformers.
"""
from typing import Protocol
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class Transformer(Protocol):
    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        """
        Applies a transformation that affects accessing the image.

        Note: If this method modifies the incoming accessor,
        it has to return a __copy__ of the incoming object.
        The copy can be created using datalcasses.replace.
        """
        ...

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Applies a transformation to the metadata of an image.
        """
        ...

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        """
        Transform an incoming image.
        """
        ...
