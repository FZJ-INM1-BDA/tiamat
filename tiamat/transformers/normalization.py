"""
Normalization transformers.
"""

import numpy as np

from .protocol import Transformer
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class MinMaxNormalizationTransformer(Transformer):
    def __init__(self, target_dtype=np.float32):
        self.target_dtype = target_dtype

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        metadata.dtype = self.target_dtype
        metadata.value_range = (0.0, 1.0)

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        assert metadata.value_range is not None, "LUTTransformer requires metadata.value_range."

        vmin, vmax = metadata.value_range
        image = (image.astype(self.target_dtype) - vmin) / (vmax - vmin)

        return image
