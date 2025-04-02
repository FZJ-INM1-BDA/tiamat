"""
Reader for Stacks.
"""

import glob
import os.path
from functools import cached_property, cache

from tiamat.readers.protocol import ImageReader

from tiamat.io import ImageAccessor, ImageResult
from tiamat.metadata import ImageMetadata


def _find_slices(fnames: str) -> tuple[str, ...]:
    # TODO: Implement
    return fnames


class StackReader(ImageReader):
    def __init__(self, fnames, reader_factory):
        # TODO: Allow Glob or Regex as fnames
        self.fnames = fnames
        self.reader_factory = reader_factory

    @cache
    def _get_handle_for_slice(self, slice_fname, **reader_kwargs):
        return self.reader_factory(slice_fname, **reader_kwargs)

    @cache
    def _get_ordered_slice_handles(self):
        return [self._get_handle_for_slice(fname) for fname in self.fnames]

    @cached_property
    def prototype_slice_handle(self):
        return self._get_ordered_slice_handles()[0]

    @cache
    def read_metadata(self) -> ImageMetadata:
        metadata = self.prototype_slice_handle.read_metadata()

        # Expand metadata for stack by simply expanding the shape
        shape = metadata.shape
        shape = tuple([self.num_slices, *shape])
        metadata.shape = shape

        return metadata

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        import numpy as np

        # Read and stack all images. For efficiency, create empty array first, then write remaining data into arrays.
        slice_handles = self._get_ordered_slice_handles()
        first_result = slice_handles[0].read_image(accessor=accessor)
        shape = tuple([self.num_slices, *first_result.image.shape])
        dtype = first_result.image.dtype
        image = np.zeros(shape=shape, dtype=dtype)
        image[0] = first_result.image
        # write remaining images
        for i, handle in enumerate(slice_handles[1:], 1):
            image[i] = handle.read_image(accessor=accessor).image
        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    @cached_property
    def file_handle(self) -> ImageReader:
        return self.prototype_slice_handle
    
    @cached_property
    def scales(self) -> list[float]:
        # TODO: Implement
        raise NotImplementedError

    @cached_property
    def shape(self) -> tuple:
        return tuple([*self.prototype_slice_handle.shape, self.num_slices])

    @cached_property
    def dtype(self):
        return self.prototype_slice_handle.dtype

    @property
    def image_spacing(self) -> tuple[float, float]:
        return self.prototype_slice_handle.image_spacing

    @cached_property
    def value_range(self) -> tuple[float | int, float | int]:
        return self.prototype_slice_handle.value_range

    @cached_property
    def slices(self) -> tuple[str, ...]:
        return _find_slices(fnames=self.fnames)

    @cached_property
    def num_slices(self) -> int:
        return len(self.slices)

    @classmethod
    def check_file(cls, fname: str) -> bool | int | float:
        # TODO: Implement
        raise NotImplementedError
