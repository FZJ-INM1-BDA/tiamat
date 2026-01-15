"""
Protocol for transformers.
"""

from typing import Protocol

import numpy as np

from ..io import ImageAccessor
from ..metadata import ImageMetadata


class Transformer(Protocol):
    """
    Protocol for image transformation steps in a pipeline.

    A transformer can modify metadata, access parameters, and image data
    in three stages, executed in the following order:

        1. transform_metadata
        2. transform_access
        3. transform_image
    """

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Adjusts image metadata to match the metadata describing
        the transformed image returned by `transform_image`.

        Parameters
        ----------
        metadata : ImageMetadata
            Metadata from the downstream transformer or pipeline reader.

        Returns
        -------
        ImageMetadata
            Updated metadata associated with the transformed image.
        """
        from dataclasses import replace

        metadata = replace(metadata)

        return metadata

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Modifies how the image should be accessed to read the correct image
        input required by `transform_image`.

        Parameters
        ----------
        accessor : ImageAccessor
            Access configuration from the upstream transformer or user.
        metadata : ImageMetadata
            Metadata output from this transformer's `transform_metadata`.

        Returns
        -------
        ImageAccessor
            Updated accessor to be passed downstream.
        """
        return accessor

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Modifies image pixel data based on its metadata and an accessor.

        Parameters
        ----------
        image : np.ndarray
            Image data from a downstream transformer or the pipeline reader
            to be transformed.
        metadata : ImageMetadata
            Associated metadata from the downstream transformer or reader.
        accessor : ImageAccessor
            Access configuration from the upstream transformer or user.

        Returns
        -------
        np.ndarray
            Transformed image data.
        """
        return image
