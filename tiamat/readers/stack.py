"""
Reader for Stacks.
"""

from functools import cached_property, partial
from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from tiamat.cache import instance_cache
from tiamat.io import ImageAccessor, ImageResult
from tiamat.metadata import ImageMetadata
from tiamat.readers.factory import get_reader
from tiamat.readers.protocol import ImageReader


def find_slices(fnames: str) -> List[str]:
    import glob

    return sorted(glob.glob(fnames))


def get_reader_identifier(fname, identifier):
    import re

    match = re.search(identifier, fname)

    return match.group(1)


def _select_slice_ix(accessor, num_slices):
    import math

    from tiamat.readers.processing import expand_to_length, prepare_coordinate
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

    # Calculate minimum and maximum slice index to use
    min_ix = math.floor(z_from / slice_spacing)
    max_ix = math.ceil(z_to / slice_spacing)

    # Define step between slice indices to pick
    step = 1 / image_scale

    size = max_ix - min_ix

    # Start index is the center of the size or step (depending on which is smaller)
    start_ix = min_ix + math.ceil(min(size, step) / 2) - 1

    # Select indices starting with start_ix and step size
    selected_ix = []
    ix = start_ix
    while (ix - start_ix) < size:
        selected_ix.append(min(ix, min(max_ix, num_slices) - 1))
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

    def _prepare_slices(self, slice_ix: Iterable[int]):
        selected_slices = [self.slices[i] for i in slice_ix]

        if isinstance(self.reader_factory, dict):
            reader_list = []
            for k, factory in self.reader_factory.items():
                # Find all files that match k
                file_matches = [fname for fname in selected_slices if get_reader_identifier(fname, self.reader_identifier) == k]
                reader_list.append((factory, file_matches))
            return reader_list

        elif hasattr(self.reader_factory, '__iter__'):
            reader_list = [r for r in self.reader_factory]
            selected_readers = [reader_list[i] for i in slice_ix]
            return [(factory, fname) for factory, fname in zip(selected_readers, selected_slices)]

        else:
            return [(self.reader_factory, fname) for fname in selected_slices]

    def access_slices(self, slice_ix: Iterable[int]):
        selected_handles = self._prepare_slices(slice_ix)
        return [reader(file) for reader, file in selected_handles]

    @property
    def ordered_slice_handles(self):
        return self.access_slices(range(len(self.slices)))

    @property
    def prototype_slice_handle(self):
        selected_handles = self._prepare_slices(range(len(self.slices)))
        if len(selected_handles) > 0:
            reader, file = selected_handles[0]
            return reader(file)
        else:
            return None

    @staticmethod
    def fill_spacing(slice_spacing, spacing):
        from tiamat.readers.processing import expand_to_length

        if slice_spacing is None:
            spacing_2d = expand_to_length(spacing, 2)
            assert spacing_2d[0] == spacing_2d[1], "StackReader assumes isotropic image spacing if slice_spacing is not provided"
            return (*spacing_2d, spacing_2d[0])
        else:
            return (*expand_to_length(spacing, 2), slice_spacing)

    @instance_cache
    def read_metadata(self) -> ImageMetadata:
        from dataclasses import replace

        metadata = replace(self.prototype_slice_handle.read_metadata())

        metadata.file_path = self.fnames

        # Expand metadata for stack by simply expanding the shape
        metadata.shape = tuple([self.num_slices, *metadata.shape])

        # Set spacing
        metadata.spacing = ImageStackReader.fill_spacing(self.slice_spacing, metadata.spacing)

        # Set scales
        metadata.scales = self.scales

        if metadata.channel_dimension is not None:
            metadata.channel_dimension = metadata.channel_dimension + 1

        return metadata

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        from dataclasses import replace

        # Use only slice handles for the requested scale
        try:
            selected_slice_ix = _select_slice_ix(accessor, self.num_slices)
            # slice_handles = [self.ordered_slice_handles[i] for i in selected_slice_ix]
            slice_handles = self.access_slices(selected_slice_ix)
        except:
            raise

        if len(slice_handles) == 0:
            raise Exception("Requested empty stack")

        # Remove z axis for 2D access from metadata
        tmp_accessor = replace(accessor, z = None)
        # if scale is tuple, remove z scale
        if isinstance(tmp_accessor.scale, (tuple, list)):
            assert len(tmp_accessor.scale) == 3
            tmp_accessor.scale = tmp_accessor.scale[:2]

        metadata = replace(tmp_accessor.metadata)
        metadata.shape = metadata.shape[1:]
        if metadata.channel_dimension is not None:
            metadata.channel_dimension = metadata.channel_dimension - 1
        tmp_accessor.metadata = metadata

        # print("StackReader", tmp_accessor)
        first_result = slice_handles[0].read_image(accessor=tmp_accessor)
        # For efficiency, create empty array first, then write remaining data into arrays.
        image = np.full(
            shape=([len(slice_handles), *first_result.image.shape]),
            fill_value=accessor.fill_value,
            dtype=first_result.image.dtype,
        )

        # Reuse first result
        image[0] = first_result.image
        # Read and stack all remaining images.
        for i, handle in enumerate(slice_handles[1:], 1):
            image[i] = handle.read_image(accessor=tmp_accessor).image

        return ImageResult(image=image, accessor=accessor, metadata=accessor.metadata)

    @property
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

        if isinstance(reader_factory, dict):
            if "class" in reader_factory.keys():
                # Single reader
                reader = get_reader_from_config(reader_factory, reader_post_creation_hook=reader_post_creation_hook)
            else:
                # Stack of readers
                reader = dict((k, get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook)) for k, r in reader_factory.items())
        elif hasattr(reader_factory, '__iter__'):
            reader = tuple(get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook) for r in reader_factory)
        elif reader_factory is None:
            reader = get_reader_from_config(reader_factory, reader_post_creation_hook=reader_post_creation_hook)
        else:
            raise Exception(f"Can't parse reader {reader}")

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

    @cached_property
    def scales(self):
        """Return scales of the image stack.
        To improve io, the scale along the stacked axis is set to 1.0
        """

        # for now, assert scale is same for all slices
        metadata_first_slice = self.ordered_slice_handles[0].read_metadata()

        if not hasattr(metadata_first_slice, "scales"):
            return None
        if metadata_first_slice.scales is None:
            return None

        # print("metadata_first_slice", metadata_first_slice)

        # TODO: make shure position of z is correct
        # TODO: future: zyxt(c)
        scales = metadata_first_slice.scales
        return [(*s[:2], 1.0, *s[2:]) for s in scales]


class VolumeStackReader(ImageReader):

    def __init__(
        self,
        fnames: str | Iterable[str],
        flag_const_shape: bool = False,
        reader_identifier: str = None,
        reader_factory: (
            Callable[[str], ImageReader]
            | Iterable[Callable[[str], ImageReader]]
            | Dict[str, Callable[[str], ImageReader]]
        ) = None,
        reader_kwargs=None,
    ):

        self.fnames = fnames
        self.flag_const_shape = flag_const_shape
        self.reader_factory = reader_factory or get_reader

        if isinstance(self.reader_factory, dict):
            assert reader_identifier is not None
        self.reader_identifier = reader_identifier

        self.reader_kwargs = reader_kwargs or {}

    @cached_property
    def slices(self) -> List[str]:
        if hasattr(self.fnames, '__iter__') and not isinstance(self.fnames, str):
            return [fname for fname in self.fnames]
        else:
            return find_slices(fnames=self.fnames)

    @cached_property
    def num_slices(self) -> int:
        return len(self.slices)

    @property
    def ordered_subvolume_handles(self):
        if isinstance(self.reader_factory, dict):
            reader_list = []
            for k in sorted(self.reader_factory.keys()):
                factory = self.reader_factory[k]
                # Find all files that match k
                file_matches = tuple(fname for fname in self.slices if get_reader_identifier(fname, self.reader_identifier) == k)
                if len(file_matches) > 0:
                    reader_list.append(factory(file_matches))
            return reader_list

        elif hasattr(self.reader_factory, '__iter__'):
            return [factory(fname, **self.reader_kwargs) for fname, factory in zip(self.slices, self.reader_factory)]

        else:
            return [self.reader_factory(fname, **self.reader_kwargs) for fname in self.slices]

    @property
    def prototype_subvolume_handle(self):
        return self.ordered_subvolume_handles[0]

    @cached_property
    def subvolume_shapes(self):

        if self.flag_const_shape:
            metadata = self.ordered_subvolume_handles[0].read_metadata()
            logger.debug("VolumeStackReader subvolume_shapes: %s", metadata.shape)
            return [(metadata.shape) for _ in range(self.num_slices)]

        return [handle.read_metadata().shape for handle in self.ordered_subvolume_handles]

    @cached_property
    def shape(self):
        metadata = self.ordered_subvolume_handles[0].read_metadata()
        z_dim = metadata.spatial_dimensions[0]

        shape = list(self.subvolume_shapes[0])
        for s in self.subvolume_shapes[1:]:
            shape[z_dim] = shape[z_dim] + s[z_dim]

        return tuple(shape)

    @instance_cache
    def read_metadata(self) -> ImageMetadata:
        from dataclasses import replace

        metadata = replace(self.ordered_subvolume_handles[0].read_metadata())

        metadata.file_path = self.fnames

        metadata.shape = self.shape

        return metadata

    def read_image(self, accessor: ImageAccessor) -> ImageResult:
        import math
        from dataclasses import replace

        from tiamat.readers.processing import (
            _prepare_coordinates,
            expand_to_length,
            prepare_coordinate,
        )
        from tiamat.transformers.coordinates import resolve_coordinate_slice

        z_dim, y_dim, x_dim = accessor.metadata.spatial_dimensions
        ch_dim = accessor.metadata.channel_dimension

        image_z_size = self.shape[z_dim]
        image_scales = expand_to_length(accessor.scale, 3)  # x, y, z

        # TODO: Probably also account for coordinate_scale
        coordinate_scales = expand_to_length(accessor.coordinate_scale, 3)  # x, y, z

        z_slice = prepare_coordinate(accessor.z)
        z_from, z_to = resolve_coordinate_slice(z_slice, image_z_size)

        out_image = None

        # TODO: We might want to support padding in the future
        def insert_array(array, offset):
            nonlocal out_image

            if out_image is None:
                out_shape = list(array.shape)
                out_shape[z_dim] = math.floor((z_to - z_from) * image_scales[-1])

                out_image = np.zeros(
                    shape=out_shape,
                    dtype=array.dtype,
                )

            z_size = min(array.shape[z_dim], out_image.shape[z_dim] - offset)

            index_to = [slice(None)] * len(out_image.shape)
            index_to[z_dim] = slice(offset, offset + z_size)

            index_from = [slice(None)] * len(out_image.shape)
            index_from[z_dim] = slice(0, z_size)

            out_image[tuple(index_to)] = array[tuple(index_from)]

        # Build volume stack
        cur_z_offset = 0
        for handle, shape in zip(self.ordered_subvolume_handles, self.subvolume_shapes):
            z_size = shape[z_dim]

            # Use only slice handles with z slice overlap
            if cur_z_offset + z_size > z_from and cur_z_offset < z_to:

                # From, to slice for image access
                from_ix = max(z_from - cur_z_offset, 0)
                to_ix = min(z_to - cur_z_offset, z_size)

                # Access subvolume and read from it
                tmp_accessor = replace(accessor, z=(from_ix, to_ix))

                # print("VolumeStackReader", accessor)
                tmp_image = handle.read_image(accessor=tmp_accessor).image

                # Position in the output array to place the image
                scaled_z_offset = math.floor(max(cur_z_offset - z_from, 0) * image_scales[-1])

                # Insert the result in the array at specific offfset
                insert_array(tmp_image, scaled_z_offset)

                # Output image might cover more than z_size
                cur_z_offset += z_size # tmp_image.shape[z_dim] / image_scales[-1]
            else:
                cur_z_offset += z_size

        return ImageResult(image=out_image, accessor=accessor, metadata=accessor.metadata)

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

        if isinstance(reader_factory, dict):
            if "class" in reader_factory.keys():
                # Single reader
                reader = get_reader_from_config(reader_factory, reader_post_creation_hook=reader_post_creation_hook)
            else:
                # Stack of readers
                reader = dict((k, get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook)) for k, r in reader_factory.items())
        elif hasattr(reader_factory, '__iter__'):
            reader = tuple(get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook) for r in reader_factory)
        elif reader_factory is None:
            reader = get_reader_from_config(reader_factory, reader_post_creation_hook=reader_post_creation_hook)
        else:
            raise Exception(f"Can't parse reader {reader}")

        if reader_post_creation_hook is None:
            return partial(
                cls,
                flag_const_shape=args.get("flag_const_shape"),
                reader_factory=reader,
                reader_identifier=args.get("reader_identifier"),
            )
        else:
            return partial(
                reader_post_creation_hook,
                cls,
                flag_const_shape=args.get("flag_const_shape"),
                reader_factory=reader,
                reader_identifier=args.get("reader_identifier"),
            )
