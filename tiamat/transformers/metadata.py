"""
Transformers to specifically modify metadata.
"""

from abc import ABC
from typing import Callable
from dataclasses import fields
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


class MetadataKwargsTransformer(_MetadataTransformer):
    """
    Modify metadata based on given kwargs
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.metadata_kwargs = kwargs

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Applied the transformation defined by the given lambda.
        """
        from dataclasses import replace

        metadata = replace(metadata)

        field_names = {f.name for f in fields(metadata)}

        for key, value in self.metadata_kwargs.items():
            if key in field_names:
                setattr(metadata, key, value)
            else:
                raise AttributeError(f"{key} is not a valid metadata attribute")

        return metadata


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
