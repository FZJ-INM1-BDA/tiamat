"""
Protocol for transformers.
"""
from typing import Protocol

import numpy as np

from ..io import ImageAccessor
from ..metadata import ImageMetadata


class Transformer(Protocol):

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Applies a transformation that affects accessing the image.

        Note: If this method modifies the incoming accessor,
        it has to return a __copy__ of the incoming object.
        The copy can be created using datalcasses.replace.
        """
        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Applies a transformation to the metadata of an image.
        """
        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Transform an incoming image.
        """
        return image
