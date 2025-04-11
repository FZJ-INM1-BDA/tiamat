"""
Reader for Stacks.
"""

from functools import cached_property, cache, partial
from typing import Any, Callable, Dict, Iterable, List

from tiamat.readers.protocol import ImageReader
from tiamat.readers.factory import get_reader

from tiamat.io import ImageAccessor, ImageResult
from tiamat.metadata import ImageMetadata


def find_slices(fnames: str) -> List[str]:
    import glob

    return sorted(glob.glob(fnames))


def get_reader_identifier(fname, identifier):
    import re

    match = re.search(identifier, fname)

    return match.group(1)


def _select_slice_ix(accessor, num_slices):
    import math
    from tiamat.readers.processing import prepare_coordinate, expand_to_length
    from tiamat.transformers.coordinates import resolve_coordinate_slice

    spatial_dims = accessor.metadata.spatial_dimensions

    assert len(spatial_dims) == 3, "Only able to perform stack slicing for 3D images"

    image_scale = expand_to_length(accessor.scale, 3)[-1]  # x, y, z
    z_slice = prepare_coordinate(accessor.z)
    z_shape = accessor.metadata.shape[spatial_dims[0]]
    z_from, z_to = resolve_coordinate_slice(z_slice, z_shape)

    # Set spacing
    # spacing = ImageStackReader.fill_spacing(self.slice_spacing, accessor.metadata.spacing)
    # slice_spacing = spacing[-1]
    slice_spacing = 1.0  # Ignore spacing for now

    min_ix = math.floor(z_from / slice_spacing)
    max_ix = math.ceil(z_to / slice_spacing)
    step = 1 / image_scale

    start_ix = min_ix + math.ceil(min(num_slices, step) / 2) - 1

    selected_ix = []
    ix = start_ix
    while ix < max_ix:
        selected_ix.append(min(ix, num_slices - 1))
        ix = int(ix + step)

    return selected_ix


class ImageStackReader(ImageReader):
    
    def __init__(
            self,
            fnames: str | Iterable[str],
            reader_identifier: str = None,
            reader_factory: Callable[[str], ImageReader] | Iterable[Callable[[str], ImageReader]] | Dict[str, Callable[[str], ImageReader]] = None,
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

        if isinstance(self.reader_factory, dict):
            assert reader_identifier is not None
        self.reader_identifier = reader_identifier

        self.slice_spacing = slice_spacing
        self.reader_kwargs = reader_kwargs

    @cached_property
    def ordered_slice_handles(self):
        if isinstance(self.reader_factory, dict):
            reader_list = []
            for k, factory in self.reader_factory.items():
                # Find all files that match k
                file_matches = [fname for fname in self.slices if get_reader_identifier(fname, self.reader_identifier) == k]
                reader_list.append(factory(file_matches))
            return reader_list
        
        elif isinstance(self.reader_factory, list):
            return [factory(fname) for fname, factory in zip(self.slices, self.reader_factory)]
        
        else:
            return [self.reader_factory(fname) for fname in self.slices]


    @cached_property
    def prototype_slice_handle(self):
        return self.ordered_slice_handles[0]

    @staticmethod
    def fill_spacing(slice_spacing, spacing):
        from tiamat.readers.processing import expand_to_length

        if slice_spacing is None:
            spacing_2d = expand_to_length(spacing, 2)
            assert spacing_2d[0] == spacing_2d[1], "StackReader assumes isotropic image spacing if slice_spacing is not provided"
            return (*spacing_2d, spacing_2d[0])
        else:
            return (*expand_to_length(spacing, 2), slice_spacing)

    @cache
    def read_metadata(self) -> ImageMetadata:
        from dataclasses import replace

        metadata = replace(self.prototype_slice_handle.read_metadata())

        metadata.file_path = self.fnames

        # Expand metadata for stack by simply expanding the shape
        metadata.shape = tuple([self.num_slices, *metadata.shape])

        # Set spacing
        metadata.spacing = ImageStackReader.fill_spacing(self.slice_spacing, metadata.spacing)

        if metadata.channel_dimension is not None:
            metadata.channel_dimension = metadata.channel_dimension + 1

        return metadata

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        from dataclasses import replace
        import numpy as np

        # Use only slice handles for the requested scale
        slice_handles = [self.ordered_slice_handles[i] for i in _select_slice_ix(accessor, self.num_slices)]

        # Remove z axis for 2D access from metadata
        tmp_accessor = replace(accessor, z = None)

        metadata = replace(tmp_accessor.metadata)
        metadata.shape = metadata.shape[1:]
        if metadata.channel_dimension is not None:
            metadata.channel_dimension = metadata.channel_dimension - 1
        tmp_accessor.metadata = metadata

        first_result = slice_handles[0].read_image(accessor=tmp_accessor)
        # For efficiency, create empty array first, then write remaining data into arrays.
        image = np.zeros(
            shape=([len(slice_handles), *first_result.image.shape]),
            dtype=first_result.image.dtype
        )

        # Reuse first result
        image[0] = first_result.image
        # Read and stack all remaining images.
        for i, handle in enumerate(slice_handles[1:], 1):
            image[i] = handle.read_image(accessor=tmp_accessor).image

        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    @cached_property
    def file_handle(self) -> ImageReader:
        return self.prototype_slice_handle

    @cached_property
    def slices(self) -> List[str]:
        if hasattr(self.fnames, '__iter__') and not isinstance(self.fnames, str):
            return [fname for fname in self.fnames]
        else:
            return find_slices(fnames=self.fnames)

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
        elif isinstance(reader_factory, dict):
            reader = dict((k, get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook)) for k, r in reader_factory.items())
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


class VolumeStackReader(ImageReader):
    
    def __init__(
            self,
            fnames: str | Iterable[str],
            reader_identifier: str = None,
            reader_factory: Callable[[str], ImageReader] | Iterable[Callable[[str], ImageReader]] | Dict[str, Callable[[str], ImageReader]] = None,
            **reader_kwargs
        ):

        self.fnames = fnames
        self.reader_factory = reader_factory or get_reader

        if isinstance(self.reader_factory, dict):
            assert reader_identifier is not None
        self.reader_identifier = reader_identifier

        self.reader_kwargs = reader_kwargs

    @cached_property
    def slices(self) -> List[str]:
        if hasattr(self.fnames, '__iter__') and not isinstance(self.fnames, str):
            return [fname for fname in self.fnames]
        else:
            return find_slices(fnames=self.fnames)

    @cached_property
    def num_slices(self) -> int:
        return len(self.slices)

    @cached_property
    def ordered_subvolume_handles(self):
        if isinstance(self.reader_factory, dict):
            reader_list = []
            for k, factory in self.reader_factory.items():
                # Find all files that match k
                file_matches = tuple(fname for fname in self.slices if get_reader_identifier(fname, self.reader_identifier) == k)
                reader_list.append(factory(file_matches))
            return reader_list
        
        elif isinstance(self.reader_factory, list):
            return [factory(fname) for fname, factory in zip(self.slices, self.reader_factory)]
        
        else:
            return [self.reader_factory(fname) for fname in self.slices]

    @cached_property
    def prototype_subvolume_handle(self):
        return self.ordered_subvolume_handles[0]
    
    @cached_property
    def subvolume_shapes(self):
        return [handle.read_metadata().shape for handle in self.ordered_subvolume_handles]

    @cached_property
    def shape(self):
        metadata = self.ordered_subvolume_handles[0].read_metadata()
        z_dim = metadata.spatial_dimensions[0]

        shape = list(self.subvolume_shapes[0])
        for s in self.subvolume_shapes[1:]:
            shape[z_dim] = shape[z_dim] + s[z_dim]

        return tuple(shape)

    @cache
    def read_metadata(self) -> ImageMetadata:
        from dataclasses import replace

        metadata = replace(self.ordered_subvolume_handles[0].read_metadata())

        metadata.file_path = self.fnames

        metadata.shape = self.shape

        return metadata

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        from dataclasses import replace
        import numpy as np
        
        from tiamat.readers.processing import prepare_coordinate, expand_to_length
        from tiamat.transformers.coordinates import resolve_coordinate_slice
        
        z_dim = accessor.metadata.spatial_dimensions[0]

        image_z_size = self.shape[z_dim]
        image_scale = expand_to_length(accessor.scale, 3)[-1]  # x, y, z

        z_slice = prepare_coordinate(accessor.z)
        z_from, z_to = resolve_coordinate_slice(z_slice, image_z_size)

        # TODO: We might also filter subvolumes based on the requested scale
        # When scale is larger than shape of a subvolume we might skip some
        # selected_ix = _select_slice_ix(accessor, image_scale)

        # Build volume stack
        cur_z = 0
        volume_stack = []
        for handle, shape in zip(self.ordered_subvolume_handles, self.subvolume_shapes):
            z_size = shape[z_dim]

            if cur_z < z_to and cur_z + z_size > z_from:
                # Use only slice handles with z slice overlap
                tmp_accessor = replace(accessor, z=(z_from - cur_z, z_to - cur_z))
                volume_stack.append(handle.read_image(accessor=tmp_accessor).image)

            cur_z += z_size

        # TODO: This might be slow. Replace with fixed array initialization and write to this
        image = np.stack(volume_stack, axis=z_dim)

        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

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
        elif isinstance(reader_factory, dict):
            reader = dict((k, get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook)) for k, r in reader_factory.items())
        else:
            reader = get_reader_from_config(reader_factory, reader_post_creation_hook=reader_post_creation_hook)

        if reader_post_creation_hook is None:
            return partial(
                cls,
                reader_factory=reader,
                reader_identifier=args.get("reader_identifier"),
            )
        else:
            return partial(
                reader_post_creation_hook,
                cls,
                reader_factory=reader,
                reader_identifier=args.get("reader_identifier"),
            )
