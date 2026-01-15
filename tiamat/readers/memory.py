"""
Reader for in-memory arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from tiamat.cache import instance_cache

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import ImageReader


class MemoryReader(ImageReader):
    """
    Reader that serves images directly from an in-memory array-like object.

    Notes:
        - The constructor argument `image` can be a numpy array or any object exposing
           a numpy-compatible `__array__` interface.
        - Minimal metadata is inferred from the array, but can be overridden via `metadata_kwargs`.
    """

    def __init__(self, image: Any, **metadata_kwargs: Any) -> None:
        """
        Initialize the memory reader.

        Args:
            image: Array-like image data (e.g., numpy.ndarray).
            **metadata_kwargs: Overrides for `ImageMetadata` fields.
                Accepted keys include:
                    - image_type
                    - value_range
                    - spacing
                    - dimensions
        """

        self.image = image
        self._cached_image = None
        self.metadata_kwargs: dict[str, Any] = metadata_kwargs

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        """
        Apply access (crop/roi) and rescale based on the given accessor.

        Args:
            accessor: Access descriptor (slicing, scaling, spacing etc.).

        Returns:
            np.ndarray: The processed image.
        """

        from .processing import access_and_rescale_image

        # Read, crop, and rescale.
        image = access_and_rescale_image(image=self.image, metadata=self.read_metadata(), accessor=accessor)

        return image

    @instance_cache
    def read_metadata(self) -> ImageMetadata:
        """
        Infer minimal metadata from the in-memory image and apply overrides.

        Heuristics for dimensions:
            - Start with (Y, X).
            - If ndim == 3 and last axis is 3 or 4, assume RGB / RGBA.
            - Otherwise, treat extra axes as generic channels C.

        Returns:
            ImageMetadata: Object with inferred or overridden metadata.
        """
        from tiamat import metadata as md

        fallback_dimensions = [
            md.dimensions.Y,
            md.dimensions.X,
        ] + [md.dimensions.C for _ in range(len(self.image.shape) - 2)]

        return md.ImageMetadata(
            image_type=self.metadata_kwargs.get("image_type", md.IMAGE_TYPE_IMAGE),
            shape=self.image.shape,
            dtype=self.image.dtype,
            value_range=self.metadata_kwargs.get("value_range", (0, 255)),
            spacing=self.metadata_kwargs.get("spacing", None),
            dimensions=self.metadata_kwargs.get(
                "dimensions",
                fallback_dimensions,
            ),
        )

    @classmethod
    def check_file(cls, fname) -> bool | int | float:
        """
        Indicate support for array-like inputs.

        Args:
            fname: Object to test.

        Returns:
            True iff `fname` exposes a numpy-compatible array protocol.
        """

        return hasattr(fname, "__array__")


class ConstantImage:
    """
    Lazy array-like that returns a constant value for any requested slice.

    Only materializes the requested region in __getitem__, not the whole array.
    """

    def __init__(self, shape: Sequence[int], dtype: np.dtype = np.uint8, constant: int | float = 0) -> None:
        """
        Initialize a constant-valued image.

        Args:
            shape: Shape of the virtual image.
            dtype: NumPy dtype of the values.
            constant: Constant fill value.
        """
        self.shape: tuple[int, ...] = tuple(int(s) for s in shape)
        self.constant: int | float = constant
        self.dtype: np.dtype = np.dtype(dtype)

    def __getitem__(self, array_slice: slice | tuple[slice] | None):
        """
        Build a numpy array filled with the constant value for the requested slice.

        Args:
            array_slice: Standard numpy slicing (can include ellipsis etc.).

        Returns:
            np.ndarray with shape determined by the slice.
        """
        from tiamat.array import slice_to_interval

        array_intervals, _ = slice_to_interval(array_slice, self.shape)

        out_shape = []

        for i, ai in enumerate(array_intervals):
            dim_size = min(ai[1], self.shape[i]) - max(ai[0], 0)
            out_shape.append(dim_size)

        out_array = np.full(out_shape, self.constant, dtype=self.dtype)

        return out_array


class ConstantReader(ImageReader):
    """
    Reader that returns a constant-valued image with provided metadata.
    Useful as a synthetic source, e.g., for padding or tests.
    """

    def __init__(self, fill_value: int | float, metadata: ImageMetadata) -> None:
        """
        Initialize a constant reader.

        Args:
            fill_value: Value used to fill the image.
            metadata: ImageMetadata describing the constant image.
        """
        from dataclasses import replace

        self.fill_value: int | float = fill_value
        self.metadata = replace(metadata)
        self.metadata.file_path = None

        self.image = ConstantImage(metadata.shape, metadata.dtype, constant=fill_value)

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        """
        Build the requested region from the constant image and rescale if needed.

        Args:
            accessor: ImageAccessor describing requested region and scaling.

        Returns:
            np.ndarray: The processed constant image.
        """
        from .processing import access_and_rescale_image

        # Read, crop, and rescale.
        image = access_and_rescale_image(
            image=self.image, metadata=self.read_metadata(), accessor=accessor, image_scale=accessor.scale
        )

        return image

    @instance_cache
    def read_metadata(self) -> ImageMetadata:
        """
        Return the metadata of the constant image.

        Returns:
            ImageMetadata: Metadata associated with this constant image.
        """
        return self.metadata

    @classmethod
    def check_file(cls, fname: Any) -> bool | int | float:
        """
        For parity with MemoryReader: declare support for array-like inputs.

        Args:
            fname: Object to test.

        Returns:
            True if `fname` exposes a numpy-compatible array protocol.

        Note:
            In normal file-based selection this will evaluate to False (strings lack __array__),
            so ConstantReader won't be selected by accident in the file reader factory.
        """
        return hasattr(fname, "__array__")
