from typing import Iterable, Tuple
from functools import cached_property

from .pipeline import Pipeline
from .metadata import ImageMetadata, dimensions
from .io import ImageAccessor


def slice_to_interval(array_slice, shape):
    if array_slice is None:
        # expand None slice
        array_slice = tuple([slice(None) for _ in range(len(shape))])
    elif not isinstance(array_slice, tuple):
        # expand single integer slice
        array_slice = (array_slice,)

    if len(array_slice) > len(shape):
        raise IndexError(
            f"Encountered invalid slice with {len(array_slice)} dimensions, but array has dimension {len(shape)}"
        )
    
    # expand ellipsis (array[...])
    array_slice = list(array_slice)
    for i, sl in enumerate(array_slice):
        if sl is Ellipsis:
            diff = len(shape) - len(array_slice) + 1
            array_slice.remove(sl)
            for _ in range(diff):
                array_slice.insert(i, slice(None))
            break

    # handle special indices
    array_intervals = []
    squeeze_dims = []
    for i, sl in enumerate(array_slice):
        # integer/float indices
        if isinstance(sl, int):
            if sl < 0:
                # negative indices
                sl = slice(shape[i] + sl, shape[i] + sl + 1)
            else:
                sl = slice(sl, sl + 1)
            squeeze_dims.append(i)
        start, stop = sl.start, sl.stop
        if not start:
            start = 0
        if not stop:
            stop = shape[i]
        # negative indices
        if start and start < 0:
            start = shape[i] + start
        if stop and stop < 0:
            stop = shape[i] + stop
        array_intervals.append((start, stop))
    
    return array_intervals


class Array(object):
    """
    Array interface for any tiamat pipeline, providing compatibility with code that uses numpy-style arrays.
    By creating together, we bind together:
    1. a pipeline
    2. A file
    3. A scale

    Things to keep in mind:
    - When accessing an array, all operations (e.g., slice, shape) opereate at the given scale.
    """

    def __init__(
        self,
        file_name,
        pipeline: Pipeline,
        scale: float | int | Iterable[float | int],
        reader_kwargs: dict | None = None,
    ) -> None:
        self.file_name = file_name
        self.pipeline = pipeline
        self.scale = scale
        self.reader_kwargs = reader_kwargs or {}

    @classmethod
    def create_arrays_for_scales(
        cls,
        file_name,
        pipeline: Pipeline,
        reader_kwargs: dict | None = None,
    ) -> Tuple:
        reader_kwargs = reader_kwargs or {}
        metadata = pipeline.read_metadata(file_name=file_name, **reader_kwargs)
        scales = metadata.scales or [
            1.0,
        ]
        if not isinstance(scales, Iterable):
            scales = [
                scales,
            ]
        return tuple(
            cls(file_name=file_name, pipeline=pipeline, scale=scale, **reader_kwargs)
            for scale in scales
        )

    @cached_property
    def metadata(self) -> ImageMetadata:
        return self.pipeline.read_metadata(
            file_name=self.file_name, **self.reader_kwargs
        )

    @property
    def shape(self):
        """Shape of the image."""
        import numpy as np

        return tuple(
            np.ceil(np.array(self.metadata.shape) * self.scale).astype(int).tolist()
        )

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Size of the array."""
        import math

        return math.prod(self.shape)

    @property
    def dtype(self):
        """Dtype of the image."""
        return self.metadata.dtype

    def __getitem__(self, array_slice: slice | Tuple[slice] | None):

        array_intervals = slice_to_interval(array_slice, self.shape)

        # matching from slice to dimension names. We assume fixed (z, y, x) indexing
        dim_names = self.metadata.dimensions

        accessor_kwargs = {
            dim: ai for ai, dim in zip(array_intervals, dim_names)
        }

        # consolidate channel access into a named dictionary 
        spatial_accessor_kwargs = {key: value for key, value in accessor_kwargs.items() if key in dimensions.SPATIAL_DIMENSIONS}
        channel_accessor_kwargs = {key: value for key, value in accessor_kwargs.items() if key not in spatial_accessor_kwargs}

        accessor = ImageAccessor(
            **spatial_accessor_kwargs,
            c=channel_accessor_kwargs,
            scale=self.scale,
            coordinate_scale=self.scale,
        )
        result = self.pipeline(
            file_name=self.file_name, accessor=accessor, **self.reader_kwargs
        )
        image = result.image

        squeeze_dims = [dim for dim in squeeze_dims if image.shape[dim] == 1]
        if squeeze_dims:
            image = image.squeeze(axis=tuple(squeeze_dims))
        return image
