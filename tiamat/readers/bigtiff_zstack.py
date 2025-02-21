"""
Reader for BigTiff.
"""
import glob
import os.path
from functools import cached_property, cache

from .protocol import ImageReader

from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata

from .bigtiff import BigTiffReader


def _find_slices(fname: str) -> tuple[str, ...]:
    head, _ = os.path.splitext(fname)
    pattern = f"{head}_Slice[0-9][0-9].tif*"
    slices = sorted(glob.glob(pattern))
    return tuple(slices)


class ZstackBigTiffReader(ImageReader):
    def __init__(self, fname):
        self.fname = fname

    @cache
    def _get_handle_for_slice(self, slice_fname):
        return BigTiffReader(fname=slice_fname)

    @cache
    def _get_ordered_slice_handles(self):
        return [self._get_handle_for_slice(fname) for fname in self.slices]

    @cached_property
    def prototype_slice_handle(self):
        return self._get_ordered_slice_handles()[0]

    @cache
    def read_metadata(self) -> ImageMetadata:
        metadata = self.prototype_slice_handle.read_metadata()

        # Expand metadata for zstack by simply expanding the shape
        shape = metadata.shape
        shape = tuple([*shape, self.num_slices])
        metadata.shape = shape

        return metadata

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        import numpy as np

        # Read and stack all images. For efficiency, create empty array first, then write remaining data into arrays.
        slice_handles = self._get_ordered_slice_handles()
        first_result = slice_handles[0].read_image(accessor=accessor)
        shape = tuple([*first_result.image.shape, self.num_slices])
        dtype = first_result.image.dtype
        image = np.zeros(shape=shape, dtype=dtype)
        image[..., 0] = first_result.image
        # write remaining images
        for i, handle in enumerate(slice_handles[1:], 1):
            image[..., i] = handle.read_image(accessor=accessor).image
        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    @cached_property
    def file_handle(self) -> BigTiffReader:
        return self.prototype_slice_handle

    @cached_property
    def num_pages(self) -> int:
        return self.prototype_slice_handle.file_handle.number_of_pages

    @cached_property
    def page_sizes(self) -> list[tuple[int, int]]:
        return self.prototype_slice_handle.page_sizes

    @cached_property
    def scales(self) -> list[float]:
        return self.prototype_slice_handle.page_sizes

    @cached_property
    def shape(self) -> tuple:
        return tuple([*self.prototype_slice_handle.shape, self.num_slices])

    @cached_property
    def dtype(self):
        return self.prototype_slice_handle.dtype

    @cache
    def is_page_tiled(self, page_index: int) -> bool:
        return self.prototype_slice_handle.is_page_tiled(page_index)

    @property
    def image_spacing(self) -> tuple[float, float]:
        return self.prototype_slice_handle.image_spacing

    @cached_property
    def tags(self) -> dict:
        return self.prototype_slice_handle.tags

    @cached_property
    def is_tiled(self) -> bool:
        return self.prototype_slice_handle.is_tiled

    @cached_property
    def value_range(self) -> tuple[float | int, float | int]:
        return self.prototype_slice_handle.value_range

    @cached_property
    def slices(self) -> tuple[str, ...]:
        return _find_slices(fname=self.fname)

    @cached_property
    def num_slices(self) -> int:
        return len(self.slices)

