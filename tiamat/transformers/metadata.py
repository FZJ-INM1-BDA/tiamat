"""
Transformers to specifically modify metadata.
"""

from abc import ABC
from typing import Callable
from .protocol import Transformer
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class _MetadataTransformer(ABC, Transformer):
    """
    Base class for metadata transformers to ensure they only modify metadata.
    """

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        # noop, no access transformation
        return accessor

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        # noop, no image transformation
        return image_result


class MetadataLambdaTransformer(_MetadataTransformer):
    """
    Modify metadata based on a given callable.
    """

    def __init__(
        self,
        metadata_lambda: Callable[
            [
                ImageMetadata,
            ],
            ImageMetadata,
        ],
    ) -> None:
        self.metadata_lambda = metadata_lambda

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Applied the transformation defined by the given lambda.
        """
        return self.metadata_lambda(metadata)
