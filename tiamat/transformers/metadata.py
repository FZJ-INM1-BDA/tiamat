"""
Transformers to specifically modify metadata.
"""

from abc import ABC
from dataclasses import fields
from typing import Any, Callable

import numpy as np

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import Transformer


class _MetadataTransformer(ABC, Transformer):
    """
    Base class for transformers that only modify metadata.
    """

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        # noop, no access transformation
        return accessor

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        # noop, no image transformation
        return image


class MetadataKwargsTransformer(_MetadataTransformer):
    """
    Modify metadata based on given kwargs
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            **kwargs: Field-value pairs to set in the metadata.
        """
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
        from dataclasses import replace

        metadata = replace(metadata)

        return self.metadata_lambda(metadata)
