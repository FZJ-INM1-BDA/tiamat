"""
Color transformers.
"""

import numpy as np

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import Transformer


class LUTTransformer(Transformer):
    """
    Apply a look-up table (LUT) or color map to an image.
    """

    def __init__(self, color_map: str | np.ndarray | list | tuple) -> None:
        """
        Initialize a LUTTransformer.

        Args:
            color_map: Colormap to apply. Can be a string (matplotlib name), numpy array,
                list, or tuple.
        """
        self.color_map = color_map

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Leave accessor unchanged.

        Args:
            accessor: Accessor of the image.
            metadata: Metadata of the image.

        Returns:
            The unchanged accessor.
        """
        return accessor

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Apply the LUT or colormap to the image.

        Args:
            image: Input image.
            metadata: Metadata of the image, must provide `value_range`.
            accessor: Accessor of the image.

        Returns:
            Color-mapped image as ndarray.
        """
        assert metadata.value_range is not None, "LUTTransformer requires metadata.value_range."

        image = self._apply_color_map(image=image, value_range=metadata.value_range)

        return image

    def _apply_color_map(self, image: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
        """
        Internal helper to apply the color map.

        Args:
            image: Input image array.
            value_range: Tuple of (min, max) values for normalization.

        Returns:
            Color-mapped image as ndarray.
        """
        import numpy as np

        if isinstance(self.color_map, str):
            # Matplotlib colormap
            import matplotlib
            from matplotlib.colors import Normalize

            color_map = matplotlib.colormaps.get_cmap(self.color_map)
            vmin, vmax = value_range
            normalizer = Normalize(vmin=vmin, vmax=vmax)

            return color_map(normalizer(image))
        elif isinstance(self.color_map, (np.ndarray, (tuple, list))):
            # Color map provided as indexable array.
            color_map = np.asarray(self.color_map)
            return color_map[image]
        else:
            raise RuntimeError(f"Unknown type for color map: {type(self.color_map)}")

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Update metadata shape for 3-channel color output.

        Args:
            metadata: Original ImageMetadata.

        Returns:
            Updated ImageMetadata.
        """
        from dataclasses import replace

        metadata = replace(metadata)

        if isinstance(self.color_map, str):
            import matplotlib

            metadata.dtype = np.asarray(matplotlib.colormaps.get_cmap(self.color_map)(0)).dtype
            ldim = len(matplotlib.colormaps.get_cmap(self.color_map)(0))
            metadata.value_range = (0.0, 1.0)
        elif isinstance(self.color_map, (np.ndarray, (tuple, list))):
            metadata.dtype = np.asarray(self.color_map).dtype
            ldim = len(self.color_map[0])
            if isinstance(self.color_map, np.ndarray):
                print(self.color_map)
                metadata.value_range = (np.min(self.color_map).item(), np.max(self.color_map).max().item())
            else:
                metadata.value_range = (min(self.color_map), max(self.color_map))

        else:
            raise RuntimeError(f"Unknown type for color map: {type(self.color_map)}")
        metadata.shape = (*metadata.shape, ldim)

        return metadata


class GrayscaleTransformer(Transformer):
    """
    Convert RGB/RGBA images to grayscale.
    """

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Leave accessor unchanged.

        Args:
            accessor: Accessor of the image.
            metadata: Metadata of the image.

        Returns:
            The unchanged accessor.
        """
        return accessor

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Convert RGB/RGBA image to grayscale using OpenCV.

        Args:
            image: Image to transform.
            metadata: Metadata of the image.
            accessor: Accessor of the image.

        Returns:
            Grayscale image as ndarray.
        """
        import cv2

        from tiamat.metadata import dimensions

        # Only do this if the image contains RGB/RGBA channels
        if dimensions.RGB in metadata.dimensions or dimensions.RGBA in metadata.dimensions:
            # TODO: Check if COLOR_BGR2GRAY is correct or COLOR_RGB2GRAY should be used
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Remove RGB/RGBA dimensions from metadata.

        Args:
            metadata: Original ImageMetadata.

        Returns:
            Updated ImageMetadata without color channels.
        """
        from dataclasses import replace

        from tiamat.metadata import dimensions

        # remove all color dimensions
        metadata = replace(metadata)
        if dimensions.RGB in metadata.dimensions or dimensions.RGBA in metadata.dimensions:
            metadata.shape = metadata.shape[:-1]
        metadata.dimensions = [
            dimension for dimension in metadata.dimensions if dimension not in (dimensions.RGB, dimensions.RGBA)
        ]

        return metadata


class GrayscaleToRGBTransformer(Transformer):
    """
    Convert grayscale images to RGB.
    """

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Leave accessor unchanged.

        Args:
            metadata: Metadata of the Image.
            accessor: Accessor of the image.

        Returns:
            The unchanged accessor.
        """
        return accessor

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Convert grayscale image to RGB using OpenCV.

        Args:
            image: Image to transform.
            metadata:
            accessor:

        Returns:
            RGB ImageResult.
        """
        import cv2

        from tiamat.metadata import dimensions

        # Only do something if the image is not already RGB.
        if not (dimensions.RGB in metadata.dimensions or dimensions.RGBA in metadata.dimensions):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Update metadata shape and dimensions for RGB.

        Args:
            metadata: Original ImageMetadata.

        Returns:
            Updated ImageMetadata.
        """
        from dataclasses import replace

        from tiamat.metadata import dimensions

        metadata = replace(metadata)
        if not (dimensions.RGB in metadata.dimensions or dimensions.RGBA in metadata.dimensions):
            metadata.shape = (*metadata.shape, 3)
            metadata.dimensions = list(metadata.dimensions) + [
                dimensions.RGB,
            ]

        return metadata


class FloatToByteTransformer(Transformer):
    """
    Convert floating-point images to 8-bit unsigned byte images.
    """

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Leave accessor unchanged.

        Args:
            accessor: Accessor of the image.
            metadata: Metadata of the image.

        Returns:
            The unchanged accessor.
        """
        return accessor

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Convert float images to 8-bit uint images in range [0, 255].

        Args:
            image: Image to transform.
            metadata: Metadata of the image.
            accessor: Accessor of the image.

        Returns:
            Converted image as uint8.
        """
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255).astype(np.uint8)

        return image

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Set metadata value_range to [0,255] after conversion.

        Args:
            metadata: Original ImageMetadata.

        Returns:
            Updated metadata with value_range 0–255.
        """
        from dataclasses import replace

        metadata = replace(metadata)

        metadata.value_range = (0, 255)
        return metadata
