"""
Normalization transformers.
"""
from .protocol import Transformer
from ..io import ImageAccessor, ImageResult


class MinMaxNormalizationTransformer(Transformer):
    def __init__(self, target_dtype=float):
        self.target_dtype = target_dtype

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        return accessor

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        assert image_result.metadata, f"LUTTransformer requires metadata."
        assert image_result.metadata.value_range is not None, f"LUTTransformer requires metadata.value_range."

        vmin, vmax = image_result.metadata.value_range
        image_result.image = (image_result.image.astype(self.target_dtype) - vmin) / (vmax - vmin)

        return image_result

