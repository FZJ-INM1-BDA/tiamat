"""
Color transformers.
"""

from .protocol import Transformer
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class LUTTransformer(Transformer):
    def __init__(self, color_map):
        self.color_map = color_map

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        return accessor

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        assert image_result.metadata, f"LUTTransformer requires metadata."
        assert (
            image_result.metadata.value_range is not None
        ), f"LUTTransformer requires metadata.value_range."

        image_result.image = self._apply_color_map(
            image=image_result.image, value_range=image_result.metadata.value_range
        )

        return image_result

    def _apply_color_map(self, image, value_range):
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
        from dataclasses import replace

        metadata = replace(metadata)

        metadata.shape = (*metadata.shape, 3)
        
        return metadata


class GrayscaleTransformer(Transformer):
    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        return accessor

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        import cv2

        # Only do something if the image is not already grayscale.
        if image_result.metadata.channel_dimension is not None:
            image_result.image = cv2.cvtColor(image_result.image, cv2.COLOR_BGR2GRAY)

        return image_result

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        from dataclasses import replace

        metadata = replace(metadata)

        ch_dim = metadata.channel_dimension
        if ch_dim is not None:
            metadata.shape = (*metadata.shape[:ch_dim], *metadata.shape[(ch_dim + 1):])
            metadata.channel_dimension = None
           
        return metadata


class GrayscaleToRGBTransformer(Transformer):
    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        accessor.metadata.channel_dimension = None

        return accessor

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        import cv2

        # Only do something if the image is not already RGB.
        if image_result.metadata.channel_dimension is None:
            image_result.image = cv2.cvtColor(image_result.image, cv2.COLOR_GRAY2RGB)

        return image_result

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        from dataclasses import replace

        metadata = replace(metadata)

        if metadata.channel_dimension is None:
            metadata.channel_dimension = len(metadata.shape)
            metadata.shape = (*metadata.shape, 3)
        
        return metadata


class FloatToByteTransformer(Transformer):
    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        return accessor

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        import numpy as np

        if np.issubdtype(image_result.image.dtype, np.floating):
            image_result.image = (image_result.image * 255).astype(np.uint8)

        return image_result

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        metadata.value_range = (0, 255)
        return metadata
