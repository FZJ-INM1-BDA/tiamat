"""
Protocol for readers.
"""

from typing import Protocol

import numpy as np

from ..io import ImageAccessor
from ..metadata import ImageMetadata


class ImageReader(Protocol):
    """
    Protocol for reading image data and metadata from a source.

    Expected call order in a pipeline:
        1. `read_metadata` → supplies metadata to the first transformer's `transform_metadata`
        2. `read_image` → supplies image data to the first transformer's `transform_image`
    """

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        """
        Read image data from the source according to the provided accessor.

        The accessor passed to `read_image` may have been modified by 
        upstream transformers via their `transform_access` method.

        Parameters
        ----------
        accessor : ImageAccessor
            Access configuration specifying region, scale, and retrieval options.

        Returns
        -------
        np.ndarray
            Image data as a NumPy array.
        """
        raise NotImplementedError

    def read_metadata(self) -> ImageMetadata:
        """
        Read metadata describing the image without loading the image data itself.

        Returns
        -------
        ImageMetadata
            Metadata containing shape, spatial and channel dimensions, spacing, 
            and other properties.
        """
        raise NotImplementedError

    @classmethod
    def check_file(cls, fname) -> bool | int | float:
        """
        Determine whether the given file is supported by this reader.

        Parameters
        ----------
        fname : str or Path
            Path to the file to check.

        Returns
        -------
        bool | int | float
            Truthy value if the file is supported. Can return a numeric 
            score indicating priority among multiple readers.
        """
        raise NotImplementedError
