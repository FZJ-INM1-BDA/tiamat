"""
Normalization transformers.
"""

import numpy as np

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import Transformer


class MinMaxNormalizationTransformer(Transformer):
    """
    Normalize image intensities to [0, 1] using the metadata value range.
    """

    def __init__(self, target_dtype: type[np.floating] = np.float32) -> None:
        """
        Initialize the transformer.

        Args:
            target_dtype: Floating point dtype for the normalized image.
                Defaults to np.float32.
        """
        self.target_dtype = target_dtype

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        No changes to the accessor are needed for normalization.

        Args:
            accessor: ImageAccessor of the target image.
            metadata: Metadata of the target image.

        Returns:
            The unmodified ImageAccessor.
        """
        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Update metadata to reflect normalization.

        Args:
            metadata: Original ImageMetadata.

        Returns:
            Updated ImageMetadata with dtype set to target_dtype and value_range set to (0.0, 1.0).
        """
        from dataclasses import replace

        metadata = replace(metadata)

        metadata.dtype = self.target_dtype
        metadata.value_range = (0.0, 1.0)

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Apply min-max normalization to the image.

        Args:
            image: Image to transform.
            metadata: Metadata of the image.
            accessor: Accessor to the image.

        Returns:
            np.ndarray: Image normalized to the range [0, 1].
        """
        assert metadata.value_range is not None, "MinMaxNormalizationTransformer requires metadata.value_range."

        vmin, vmax = metadata.value_range

        image = (image.astype(self.target_dtype) - vmin) / (vmax - vmin)

        return image
