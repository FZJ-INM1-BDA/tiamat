"""
Normalization transformers.
"""
from .protocol import Transformer
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata
import numpy as np


class MinMaxNormalizationTransformer(Transformer):
    def __init__(self, target_dtype=np.float32):
        self.target_dtype = target_dtype

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        return metadata

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        assert image_result.metadata, f"LUTTransformer requires metadata."
        assert image_result.metadata.value_range is not None, f"LUTTransformer requires metadata.value_range."

        vmin, vmax = image_result.metadata.value_range
        image_result.image = (image_result.image.astype(self.target_dtype) - vmin) / (vmax - vmin)

        return image_result

