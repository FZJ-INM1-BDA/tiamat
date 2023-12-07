"""
Reader for list of images interpreted as pyramid.
"""
from collections.abc import Iterable
from functools import cache, cached_property

from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata
from .protocol import ImageReader


class Pyramid(ImageReader):
    def __init__(
        self,
        pyramid: Iterable,
        metadata: dict,
    ):
        self.pyramid = list(pyramid)
        self.metadata = metadata

        self._check_pyramid()
        self._check_metadata()

    def _check_pyramid(self):
        assert isinstance(self.pyramid, list)
        assert all(
            [a.shape > b.shape for a, b in zip(self.pyramid[:-1], self.pyramid[1:])]
        )

    def _check_metadata(self):
        assert isinstance(self.metadata, dict)
        assert "resolution_x" in self.metadata
        assert "resolution_y" in self.metadata
        # assert "value_range" in self.metadata

    # TODO: discuss
    def level(self, level: int) -> ImageReader:
        return self.pyramid[level]

    @cache
    def read_metadata(self) -> ImageMetadata:
        from tiamat import metadata as md

        return md.ImageMetadata(
            image_type=md.IMAGE_TYPE_IMAGE,
            shape=self.shape,
            dtype=self.dtype,
            value_range=self.value_range,
            spacing=self.image_spacing,
            channel_interpretation=md.CHANNEL_INTERPRETATION_COLOR,
            additional_metadata=self.metadata,
        )

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        from ._processing import access_and_rescale_image

        # Read, crop, and rescale.
        target_scale = accessor.scale
        available_scale, _ = self._find_scale(target_scale)

        image = access_and_rescale_image(
            image=self.level(0), accessor=accessor, image_scale=available_scale
        )

        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    @cache
    def _find_scale(self, target_scale: float) -> tuple[float, int]:
        scales = [(scale, None) for scale in self.scales if scale >= target_scale]
        return scales[-1]

    @cached_property
    def scales(self) -> list[float]:
        # TODO: what if x and y are not scaled the same?
        return [layer.shape[0] / self.pyramid[0].shape[0] for layer in self.pyramid]

    @cached_property
    def shape(self) -> tuple:
        return self.level(0).shape

    @cached_property
    def dtype(self):
        return self.level(0).dtype

    @property
    def image_spacing(self) -> tuple[float, float]:
        return self.metadata["resolution_x"], self.metadata["resolution_y"]

    @cached_property
    def value_range(self) -> tuple[float | int, float | int]:
        import numpy as np

        # TODO: discuss
        if "value_range" in self.metadata:
            return self.metadata["value_range"]

        dtype = self.dtype
        if np.issubdtype(dtype, np.integer):
            dtype_info = np.iinfo(dtype)
        elif np.issubdtype(dtype, np.floating):
            dtype_info = np.finfo(dtype)
        else:
            raise TypeError(f"Cannot determin value range for dtype {dtype}")

        return dtype_info.min, dtype_info.max
