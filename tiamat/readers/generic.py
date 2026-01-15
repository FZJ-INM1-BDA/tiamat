"""
Reader for generic image formats.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tiamat.cache import instance_cache

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import ImageReader


class GenericReader(ImageReader):
    """
    Reader for generic image formats using `imageio`.

    This acts as a fallback reader for standard image files without a specialized format handler.
    It supports caching the image in memory to avoid repeated disk reads.
    """

    def __init__(
        self,
        fname: str,
        cache_image: bool = False,
        image_spacing: float | Sequence[float] | None = None,
    ) -> None:
        """
        Initialize a generic image reader.

        Args:
            fname: Path to the image file.
            cache_image: If True, keep the loaded image in memory to avoid repeated disk access.
            image_spacing: Physical pixel spacing as a single float (isotropic),
                a tuple of floats (anisotropic), or None if not specified.
        """
        self.fname = fname
        self.image_spacing = image_spacing
        self._cache_image = cache_image
        self._cached_image: np.ndarray | None = None

    def _read_image(self) -> np.ndarray:
        """
        Load the image from disk or return the cached version.

        Returns:
            np.ndarray: Raw image array as loaded by `imageio`.
        """
        from imageio.v3 import imread

        image = self._cached_image if self._cached_image is not None else imread(self.fname)
        if self._cache_image:
            self._cached_image = image

        return image

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        """
        Read, crop, and rescale the image according to the accessor.

        Args:
            accessor: ImageAccessor describing the requested region and scaling.

        Returns:
            np.ndarray: The processed image array.
        """
        from .processing import access_and_rescale_image

        # Read, crop, and rescale.
        image = self._read_image()
        image = access_and_rescale_image(image=image, metadata=self.read_metadata(), accessor=accessor)

        return image

    @instance_cache
    def read_metadata(self) -> ImageMetadata:
        """
        Read minimal metadata for a generic image.

        Since no external metadata is available, dimensions and spacing
        are inferred from the image array.

        Returns:
            ImageMetadata: Object with basic metadata such as shape, dtype,
            dimensions, spacing, and value range.
        """
        from tiamat import metadata as md
        from tiamat.readers.processing import get_value_range_from_dtype

        image = self._read_image()
        dims = [md.dimensions.Y, md.dimensions.X]

        if image.ndim == 3:
            if image.shape[2] == 3:
                dims.append(md.dimensions.RGB)
            elif image.shape[2] == 4:
                dims.append(md.dimensions.RGBA)
            else:
                dims.append(md.dimensions.C)
        else:
            dims.extend([md.dimensions.C for _ in range(max(len(image.shape) - 2, 0))])

        shape = image.shape
        dtype = image.dtype
        value_range = get_value_range_from_dtype(dtype)

        return md.ImageMetadata(
            image_type=md.IMAGE_TYPE_IMAGE,
            shape=shape,
            dtype=dtype,
            file_path=self.fname,
            value_range=value_range,
            spacing=self.image_spacing,
            dimensions=dims,
        )

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """
        Get the list of file extensions supported by `imageio`.

        Returns:
            list[str]: Supported file extensions.
        """
        import imageio

        extensions: list[str] = []
        for fmt in imageio.formats:
            extensions.extend(fmt.extensions)
        return extensions

    @classmethod
    def check_file(cls, fname: str) -> bool | int | float:
        """
        Check if the file extension is supported by this reader.

        Args:
            fname: Path to the file.

        Returns:
            True if supported, else False.
        """
        import os

        _, ext = os.path.splitext(fname)
        return ext in cls.supported_extensions()
