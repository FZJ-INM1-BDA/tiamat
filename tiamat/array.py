"""
Array interface for any tiamat pipeline, providing compatibility with code that uses numpy-style arrays.
By creating together, we bind together:
1. a pipeline
2. A file
3. A scale

Things to keep in mind:
- When accessing an array, all operations (e.g., slice, shape) opereate at the given scale. 
"""

from typing import Iterable, Tuple
from functools import cached_property

from numpy import array, isin, ndim
from .pipeline import Pipeline
from .metadata import ImageMetadata
from .io import ImageAccessor
from tiamat import metadata


class Array(object):
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
        return self.metadata.shape

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    @property
    def dtype(self):
        """Dtype of the image."""
        return self.metadata.dtype

    def __getitem__(self, array_slice: slice | Tuple[slice] | None):
        # expand None slice
        if array_slice is None:
            array_slice = tuple([slice(None) for _ in range(self.ndim)])
        # expand single integer slice
        elif not isinstance(array_slice, tuple):
            array_slice = tuple(
                [
                    array_slice,
                ],
            )
        if len(array_slice) > self.ndim:
            raise IndexError(
                f"Encountered invalid slice with {len(array_slice)} dimensions, but array has dimension {self.ndim}"
            )
        # expand ellipsis (array[...])
        array_slice = list(array_slice)
        for i, sl in enumerate(array_slice):
            if sl is Ellipsis:
                diff = self.ndim - len(array_slice) + 1
                array_slice.remove(sl)
                for _ in range(diff):
                    array_slice.insert(i, slice(None))
                break

        # matching from slice to dimension names. Mind that we swap x and y
        dim_names = ["y", "x", "z", "c"]
        if (
            self.metadata.channel_interpretation
            in (metadata.CHANNEL_INTERPRETATION_COLOR,)
            and self.ndim < 4
        ):
            dim_names.remove("z")

        # handle special indeices
        array_slice_cleaned = []
        squeeze_dims = []
        for i, sl in enumerate(array_slice):
            # integer/float indices
            if isinstance(sl, int):
                if sl < 0:
                    # negative indices
                    sl = slice(self.shape[i] + sl, self.shape[i] + sl + 1)
                else:
                    sl = slice(sl, sl + 1)
                squeeze_dims.append(i)
            start, stop = sl.start, sl.stop
            if not start:
                start = 0
            if not stop:
                stop = self.shape[i]
            # negative indices
            if start and start < 0:
                start = self.shape[i] + start
            if stop and stop < 0:
                stop = self.shape[i] + stop
            array_slice_cleaned.append(slice(start, stop))
        array_slice = array_slice_cleaned
        del array_slice_cleaned

        accessor_kwargs = {
            dim: (sl.start, sl.stop) for sl, dim in zip(array_slice, dim_names)
        }
        accessor = ImageAccessor(
            **accessor_kwargs,
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
