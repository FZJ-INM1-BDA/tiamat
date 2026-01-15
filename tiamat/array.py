from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property

from .io import ImageAccessor
from .metadata import ImageMetadata, dimensions
from .pipeline import Pipeline


def slice_to_interval(
    array_slice: int | slice | tuple[int | slice | Ellipsis, ...] | None,
    shape: tuple[int, ...],
) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Convert a given slice or index into intervals usable for image access.

    Args:
        array_slice: Slice, integer, ellipsis or tuple of them. Can also be None to expand to full slice.
        shape: The shape of the array to index into.

    Returns:
        A tuple containing:
        - A list of (start, stop) intervals for each dimension.
        - A list of dimensions to be squeezed after slicing.

    Raises:
        IndexError: If the number of slice dimensions exceeds the array's dimensions.
    """
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
        if isinstance(sl, int):
            if sl < 0:
                sl = slice(shape[i] + sl, shape[i] + sl + 1)
            else:
                sl = slice(sl, sl + 1)
            squeeze_dims.append(i)
        start, stop = sl.start, sl.stop
        if not start:
            start = 0
        if not stop:
            stop = shape[i]
        if start and start < 0:
            start = shape[i] + start
        if stop and stop < 0:
            stop = shape[i] + stop
        array_intervals.append((start, stop))

    return array_intervals, squeeze_dims


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
        shape_round_mode: str = "round",
    ) -> None:
        """
        Initialize an Array instance.

        Args:
            file_name: Path to the image file.
            pipeline: Pipeline used to read/process the image.
            scale: Scale factor(s) to apply to the image.
            reader_kwargs: Additional kwargs to pass to the pipeline when reading.
            shape_round_mode: Determines how shape values are rounded.
                              Valid options: "round", "ceil", "floor".
        """
        self.file_name = file_name
        self.pipeline = pipeline
        self.scale = scale
        self.reader_kwargs = reader_kwargs or {}
        self.shape_round_mode = shape_round_mode

    @classmethod
    def create_arrays_for_scales(
        cls,
        file_name: str,
        pipeline: Pipeline,
        reader_kwargs: dict | None = None,
    ) -> tuple:
        """
        Factory method to create `Array` instances for all scales of an image.

        Args:
            file_name: Path to the image file.
            pipeline: Pipeline used to read/process the image.
            reader_kwargs: Additional kwargs for the pipeline.

        Returns:
            Tuple of Array instances for each available scale.
        """
        reader_kwargs = reader_kwargs or {}
        metadata = pipeline.read_metadata(file_name=file_name, **reader_kwargs)
        scales = metadata.scales or [
            1.0,
        ]
        if not isinstance(scales, Iterable):
            scales = [
                scales,
            ]
        return tuple(cls(file_name=file_name, pipeline=pipeline, scale=scale, **reader_kwargs) for scale in scales)

    @cached_property
    def metadata(self) -> ImageMetadata:
        """Read and cache image metadata."""
        return self.pipeline.read_metadata(file_name=self.file_name, **self.reader_kwargs)

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """Shape of the image."""
        import numpy as np

        shape_round_functions = {
            "round": np.round,
            "ceil": np.ceil,
            "floor": np.floor,
        }
        shape_fn = shape_round_functions.get(self.shape_round_mode)
        if shape_fn is None:
            valid = ",".join(list(shape_round_functions.keys()))
            raise RuntimeError(f"Invalid shape_round_mode {self.shape_round_mode}. Valid: {valid}")

        return tuple(shape_fn(np.array(self.metadata.shape) * self.scale).astype(int).tolist())

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements in the array."""
        import math

        return math.prod(self.shape)

    @property
    def dtype(self) -> str:
        """ "Dtype of the image."""
        return self.metadata.dtype

    def __getitem__(self, array_slice: slice | tuple[slice] | None):
        """
        Retrieve image data using numpy-like slicing.

        Args:
            array_slice: Slice object(s), integer index, ellipsis, or None.

        Returns:
            The image data as returned by the pipeline.
        """

        array_intervals, squeeze_dims = slice_to_interval(array_slice, self.shape)

        # matching from slice to dimension names. We assume fixed (z, y, x) indexing
        dim_names = self.metadata.dimensions

        accessor_kwargs = {dim: ai for ai, dim in zip(array_intervals, dim_names)}

        # consolidate channel access into a named dictionary
        spatial_accessor_kwargs = {
            key: value for key, value in accessor_kwargs.items() if key in dimensions.SPATIAL_DIMENSIONS
        }
        channel_accessor_kwargs = {
            key: value for key, value in accessor_kwargs.items() if key not in spatial_accessor_kwargs
        }
        if len(channel_accessor_kwargs) == 0:
            channel_accessor_kwargs = None

        accessor = ImageAccessor(
            **spatial_accessor_kwargs,
            c=channel_accessor_kwargs,
            scale=self.scale,
            coordinate_scale=self.scale,
        )
        result = self.pipeline(file_name=self.file_name, accessor=accessor, **self.reader_kwargs)
        image = result.image

        squeeze_dims = [dim for dim in squeeze_dims if image.shape[dim] == 1]
        if squeeze_dims:
            image = image.squeeze(axis=tuple(squeeze_dims))
        return image
