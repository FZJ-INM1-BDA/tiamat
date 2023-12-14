"""
Reader for BigTiff.
"""
from functools import cached_property, cache
from .protocol import ImageReader
import pytiff

from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class BigTiffReader(ImageReader):
    def __init__(self, fname):
        self.fname = fname

    @cache
    def read_metadata(self) -> ImageMetadata:
        from tiamat import metadata as md

        return md.ImageMetadata(image_type=md.IMAGE_TYPE_IMAGE,
                                shape=self.shape,
                                dtype=self.file_handle.dtype,
                                file_path=self.fname,
                                value_range=self.value_range,
                                spacing=self.image_spacing,
                                channel_interpretation=md.CHANNEL_INTERPRETATION_COLOR,
                                additional_metadata=self.tags)

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        from ._processing import access_and_rescale_image

        # Read, crop, and rescale.
        target_scale = accessor.scale
        available_scale, available_scale_index = self._find_scale(target_scale)
        current_page = self.file_handle.current_page
        self.file_handle.set_page(available_scale_index)
        image = access_and_rescale_image(image=self.file_handle, accessor=accessor, image_scale=available_scale)
        self.file_handle.set_page(current_page)

        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    @cache
    def _find_scale(self, target_scale: float) -> tuple[float, int]:
        tiled_scales = [(scale, index) for index, scale in enumerate(self.scales) if scale >= target_scale and self.is_page_tiled(index)]
        if tiled_scales:
            return tiled_scales[-1]
        untiled_scales = [(scale, index) for index, scale in enumerate(self.scales) if scale >= target_scale and not self.is_page_tiled(index)]
        return untiled_scales[0]

    @cached_property
    def file_handle(self) -> pytiff.Tiff:
        return pytiff.Tiff(self.fname)

    @cached_property
    def num_pages(self) -> int:
        return self.file_handle.number_of_pages

    @cached_property
    def page_sizes(self) -> list[tuple[int, int]]:
        page_sizes = []
        for page in range(self.num_pages):
            self.file_handle.set_page(page)
            page_sizes.append(self.file_handle.shape)
        return page_sizes

    @cached_property
    def scales(self) -> list[float]:
        import math
        scales = [shape[0] / float(self.shape[0]) for shape in self.page_sizes]
        return [1 / 2 ** round(math.log(1 / scale, 2)) for scale in scales]

    @cached_property
    def shape(self) -> tuple:
        return sorted(self.page_sizes, key=lambda shape: -shape[0])[0]

    @cached_property
    def dtype(self):
        return self.file_handle.dtype

    @cache
    def is_page_tiled(self, page_index: int) -> bool:
        current_page = self.file_handle.current_page
        # At the moment, we read metadata from page zero by default.
        # We might want to change this in the future.
        self.file_handle.set_page(page_index)
        is_tiled = self.file_handle.is_tiled()
        self.file_handle.set_page(current_page)
        return is_tiled

    @property
    def image_spacing(self) -> tuple[float, float]:
        from pytiff import tags
        resolution_unit = self.tags[tags.resolution_unit]
        # https://www.awaresystems.be/imaging/tiff/tifftags/resolutionunit.html
        # 0 - no unit
        # 1 - inch
        # 3 - cm
        assert resolution_unit in (3, ), f"Unsupported resolution unit: {resolution_unit}"
        # centimeter
        resolution_unit_micron = 10000
        #
        return self.tags[tags.x_resolution] / resolution_unit_micron, self.tags[tags.y_resolution] / resolution_unit_micron

    @cached_property
    def tags(self) -> dict:
        current_page = self.file_handle.current_page
        self.file_handle.set_page(0)
        tags = self.file_handle.read_tags()
        self.file_handle.set_page(current_page)
        return tags

    @cached_property
    def is_tiled(self) -> bool:
        current_page = self.file_handle.current_page
        is_tiled = self.file_handle.is_tiled()
        self.file_handle.set_page(current_page)
        return is_tiled

    @cached_property
    def value_range(self) -> tuple[float | int, float | int]:
        import numpy as np
        dtype = self.dtype
        if np.issubdtype(dtype, np.integer):
            dtype_info = np.iinfo(dtype)
        elif np.issubdtype(dtype, np.floating):
            dtype_info = np.finfo(dtype)
        else:
            raise TypeError(f"Cannot determin value range for dtype {dtype}")

        return dtype_info.min, dtype_info.max

