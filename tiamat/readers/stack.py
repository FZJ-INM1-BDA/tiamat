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


class StackReader(ImageReader):
    
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

    @cache
    def read_metadata(self) -> ImageMetadata:
        from dataclasses import replace

        metadata = replace(self.prototype_slice_handle.read_metadata())

        # Expand metadata for stack by simply expanding the shape
        metadata.shape = tuple([self.num_slices, *metadata.shape])
        metadata.channel_dimension = metadata.channel_dimension + 1

        return metadata

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        from dataclasses import replace
        import numpy as np

        # Store for slice access
        z_access = accessor.z

        # Revert added axis for 2D access
        accessor = replace(accessor)
        accessor.z = None
        accessor.metadata.shape = accessor.metadata.shape[1:]
        accessor.metadata.channel_dimension = accessor.metadata.channel_dimension - 1
        
        # TODO: Read only requested images based on spacing and the scale of the accessor
        slice_handles = self._get_ordered_slice_handles()[:]

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
        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    @cached_property
    def file_handle(self) -> ImageReader:
        return self.prototype_slice_handle
    
    @cached_property
    def scales(self) -> list[float]:
        return self.prototype_slice_handle.scales

    @cached_property
    def shape(self) -> tuple:
        return (self.num_slices, *self.prototype_slice_handle.shape)

    @cached_property
    def dtype(self):
        return self.prototype_slice_handle.dtype

    @property
    def image_spacing(self) -> tuple[float, float]:
        from tiamat.readers.processing import expand_to_length

        # x, y, z
        if self.slice_spacing is None:
            spacing_2d = expand_to_length(self.prototype_slice_handle.image_spacing, 2)
            assert spacing_2d[0] == spacing_2d[1], "StackReader assumes isotropic image spacing if slice_spacing is not provided"
            return (*spacing_2d, spacing_2d[0])
        else:
            return (*expand_to_length(self.prototype_slice_handle.image_spacing, 2), self.slice_spacing)

    @cached_property
    def value_range(self) -> tuple[float | int, float | int]:
        return self.prototype_slice_handle.value_range

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
    def from_json(cls, args: Dict[str, Any], reader_factory=None):
        from tiamat.serialization import get_reader_from_config

        if reader_factory is None:
            reader = args.get("reader_factory")

            if isinstance(reader, list):
                reader = [get_reader_from_config(r) for r in reader]
            else:
                reader = get_reader_from_config(reader)

            return partial(
                cls,
                reader_factory=reader,
                slice_spacing=args.get("request_margin")
            )
        else:
            return partial(
                cls,
                reader_factory=reader_factory,
                slice_spacing=args.get("request_margin")
            )
