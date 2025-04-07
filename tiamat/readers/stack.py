"""
Reader for Stacks.
"""

from functools import cached_property, cache, partial
from typing import Any, Callable, Dict, Iterable, List

from tiamat.readers.protocol import ImageReader
from tiamat.readers.factory import get_reader

from tiamat.io import ImageAccessor, ImageResult
from tiamat.metadata import ImageMetadata


def _find_slices(fnames: str) -> List[str]:
    import glob

    return sorted(glob.glob(fnames))


class ImageStackReader(ImageReader):
    
    def __init__(
            self,
            fnames: str | Iterable[str],
            reader_factory: Callable[[str], ImageReader] | Iterable[Callable[[str], ImageReader]] = None,
            slice_spacing: float = None,
            **reader_kwargs
        ):
        """_summary_

        Parameters
        ----------
        fnames : str | Iterable[str]
            Can be a Unix shell compatible file pattern or an iterable object providing filenames
        reader_factory : Callable[[str], ImageReader] | Iterable[Callable[[str], ImageReader]], optional
            A reader factory function or specific image reader. By default search for registered readers
        slice_spacing : float, optional
            Slice spacing if volume spacing is not isotropic. By default assume same z spacing as for x, y
        reader_kwargs:
            Arguments passed to each reader from reader_factory
        """
        self.fnames = fnames
        self.reader_factory = reader_factory or get_reader
        self.slice_spacing = slice_spacing
        self.reader_kwargs = reader_kwargs

    @cache
    def _get_ordered_slice_handles(self):
        if hasattr(self.reader_factory, '__iter__'):
            return [factory(fname) for fname, factory in zip(self.slices, self.reader_factory)]
        else:
            return [self.reader_factory(fname) for fname in self.slices]

    @cached_property
    def prototype_slice_handle(self):
        return self._get_ordered_slice_handles()[0]

    def _access_slices(self, accessor):
        import math
        from tiamat.readers.processing import prepare_coordinate, expand_to_length

        image_scale = expand_to_length(accessor.scale, 3)[-1]  # x, y, z
        z_slice = prepare_coordinate(accessor.z)

        print(image_scale, z_slice)

        min_ix = math.floor(z_slice[0] / self.slice_spacing)
        max_ix = math.ceil(z_slice[1] / self.slice_spacing)
        step = 1 / image_scale

        start_ix = min_ix + math.ceil(min(self.num_slices, step) / 2) - 1

        selected_ix = []
        ix = start_ix
        while ix < max_ix:
            selected_ix.append(min(ix, self.num_slices - 1))
            ix = int(ix + step)

        print(selected_ix)

        slice_handles = [self._get_ordered_slice_handles()[i] for i in selected_ix]

        return slice_handles

    @cache
    def read_metadata(self) -> ImageMetadata:
        from dataclasses import replace
        from tiamat.readers.processing import expand_to_length

        metadata = replace(self.prototype_slice_handle.read_metadata())

        # Expand metadata for stack by simply expanding the shape
        metadata.shape = tuple([self.num_slices, *metadata.shape])

        # Set spacing
        if self.slice_spacing is None:
            spacing_2d = expand_to_length(metadata.spacing, 2)
            assert spacing_2d[0] == spacing_2d[1], "StackReader assumes isotropic image spacing if slice_spacing is not provided"
            metadata.spacing = (*spacing_2d, spacing_2d[0])
        else:
            metadata.spacing = (*expand_to_length(metadata.spacing, 2), self.slice_spacing)

        if metadata.channel_dimension is not None:
            metadata.channel_dimension = metadata.channel_dimension + 1

        return metadata

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        from dataclasses import replace
        import numpy as np

        # Use only slice handles for the requested scale
        slice_handles = self._access_slices(accessor)

        # Remove z axis for 2D access from metadata
        accessor = replace(accessor)
        metadata = replace(accessor.metadata)
        accessor.z = None

        metadata.shape = metadata.shape[1:]
        if metadata.channel_dimension is not None:
            metadata.channel_dimension = metadata.channel_dimension - 1

        first_result = slice_handles[0].read_image(accessor=accessor)
        # For efficiency, create empty array first, then write remaining data into arrays.
        image = np.zeros(
            shape=([len(slice_handles), *first_result.image.shape]),
            dtype=first_result.image.dtype
        )

        # Reuse first result
        image[0] = first_result.image
        # Read and stack all remaining images.
        for i, handle in enumerate(slice_handles[1:], 1):
            image[i] = handle.read_image(accessor=accessor).image

        return ImageResult(image=image, accessor=accessor, metadata=metadata)

    @cached_property
    def file_handle(self) -> ImageReader:
        return self.prototype_slice_handle

    @cached_property
    def slices(self) -> List[str]:
        if hasattr(self.fnames, '__iter__') and not isinstance(self.fnames, str):
            return [fname for fname in self.fnames]
        else:
            return _find_slices(fnames=self.fnames)

    @cached_property
    def num_slices(self) -> int:
        return len(self.slices)

    @classmethod
    def check_file(cls, fname: str | List[str]) -> bool | int | float:
        # StackReader requires initialization before being able to check the files
        # TODO: Maybe check if fname refers to a list of files. Check if any readers
        # exists for this filetype and return this one
        return False

    @classmethod
    def from_json(cls, args: Dict[str, Any], reader_post_creation_hook=None):
        from tiamat.serialization import get_reader_from_config

        reader_factory = args.get("reader_factory")

        if isinstance(reader_factory, list):
            reader = tuple(get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook) for r in reader_factory)
        else:
            reader = get_reader_from_config(reader_factory, reader_post_creation_hook=reader_post_creation_hook)

        if reader_post_creation_hook is None:
            return partial(
                cls,
                reader_factory=reader,
                slice_spacing=float(args.get("slice_spacing")),
            )
        else:
            return partial(
                reader_post_creation_hook,
                cls,
                reader_factory=reader,
                slice_spacing=float(args.get("slice_spacing")),
            )
