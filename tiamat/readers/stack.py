"""
Reader for image stacks and volume stacks.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import cached_property, partial
from typing import Any

import numpy as np

from tiamat.cache import instance_cache, instance_cached_property
from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata, dimensions
from tiamat.readers.factory import get_reader
from tiamat.readers.protocol import ImageReader


def find_slices(fnames: str) -> list[str]:
    """Find and sort slice files matching a glob pattern.

    Args:
        fnames (str): Unix shell-compatible file pattern.


    Returns:
        list[str]: Sorted list of file paths.
    """
    import glob

    return sorted(glob.glob(fnames))


def compile_identifier(identifier: str | None) -> re.Pattern | None:  #
    """
    Compile a regex pattern for extracting identifiers from file names.

    Args:
        identifier (str | None): Regex pattern with a capture group, or None.

    Returns:
        re.Pattern | None: Compiled regex pattern, or None if input is None.
    """
    if identifier is None:
        return None
    else:
        # Parse the regex and see if there are any groups present
        if "(" not in identifier:
            # Add group if no group is present
            identifier = f".*({identifier}).*"
        return re.compile(identifier)


def get_reader_identifier(fname: str, identifier: re.Pattern) -> str:
    """
    Extract a reader identifier from a filename using a regex.

    Args:
        fname (str): Filename to match.
        identifier (re.Pattern): Compiled regex pattern with a capture group.

    Returns:
        str: Extracted identifier.

    Raises:
        ValueError: If no match is found.
    """
    match = identifier.search(fname)
    if not match:
        raise ValueError(f'No match for identifier "{identifier}" in filename "{fname}"')

    return match.group(1)


def select_slice_ix(
    accessor: ImageAccessor, metadata: ImageMetadata, num_slices: int, slice_spacing: float = 1.0
) -> list[int]:
    """
    Select slice indices for stack access given spacing and scale.

    Args:
        accessor (ImageAccessor): Accessor specifying slicing and scaling.
        metadata (ImageMetadata): Metadata of the image stack.
        num_slices (int): Number of available slices.
        slice_spacing (float, optional): Spacing between slices. Defaults to 1.0.

    Returns:
        list[int]: Selected slice indices.

    Raises:
        AssertionError: If metadata does not represent a 3D image.
    """
    from tiamat.readers.processing import expand_to_length
    from tiamat.transformers.coordinates import resolve_coordinate_slice

    spatial_dims = metadata.spatial_dimensions

    assert len(spatial_dims) == 3, "Only able to perform stack slicing for 3D images"

    z_scale = expand_to_length(accessor.scale, 3)[-1]  # x, y, z
    z_shape = metadata.shape[spatial_dims[0]]
    z_from, z_to = resolve_coordinate_slice(accessor.z, z_shape)

    # Calculate minimum and maximum slice index to use
    min_ix = z_from / slice_spacing
    max_ix = z_to / slice_spacing

    # Define step between slice indices to pick
    step = 1 / z_scale
    size = (z_to - z_from) / slice_spacing

    # Start index is the center of the size or step (depending on which is smaller)
    start_ix = min_ix + min(size, step) / 2

    # Select indices starting with start_ix and step size
    selected_ix = []
    ix = start_ix
    while True:
        selected_ix.append(int(min(ix, min(max_ix, num_slices - 1))))
        ix = ix + step
        # Break condition accounting for rounding errors
        if (ix - start_ix) + min(size, step) / 2 >= size:
            break

    return selected_ix


class ImageStackReader(ImageReader):
    """Reader for stacks of 2D slices forming a 3D volume."""

    def __init__(
        self,
        fnames: str | Iterable[str],
        reader_identifier: str = None,
        reader_factory: (
            Callable[[str], ImageReader] | list[Callable[[str], ImageReader]] | dict[str, Callable[[str], ImageReader]]
        ) = None,
        slice_spacing: float = None,
        stack_dimension: str = dimensions.Z,
        missing_section_interpolation: str | None = None,
        missing_section_fill_value: int | float = 0,
        **reader_kwargs,
    ) -> None:
        """
        Initialize an ImageStackReader.

        Args:
            fnames (str | Iterable[str]): File pattern or iterable of filenames.
            reader_identifier (str | None, optional): Regex identifier for slice grouping.
            reader_factory (Callable | list | dict, optional): Factory for slice readers.
            slice_spacing (float | None, optional): Slice spacing. Defaults to isotropic spacing.
            stack_dimension (str, optional): Dimension along which slices are stacked. Defaults to Z.
            missing_section_interpolation (str | None, optional): Strategy for missing slices.
            missing_section_fill_value (int | float, optional): Fill value for missing slices.
            **reader_kwargs: Extra kwargs passed to reader factory.
        """
        self.fnames = fnames
        self.reader_factory = reader_factory or get_reader

        if isinstance(self.reader_factory, dict):
            assert reader_identifier is not None
        self.compiled_identifier = compile_identifier(reader_identifier)

        self.slice_spacing = slice_spacing
        self.stack_dimension = stack_dimension
        self.missing_section_interpolation = missing_section_interpolation
        self.missing_section_fill_value = missing_section_fill_value
        self.reader_kwargs = reader_kwargs

    @staticmethod
    def fill_spacing(slice_spacing: float | None, spacing: Iterable[float]) -> tuple[float, float, float]:
        """
        Fill spacing for 3D stack.


        Args:
        slice_spacing (float | None): Explicit slice spacing.
        spacing (Iterable[float]): In-plane spacing.


        Returns:
        tuple[float, float, float]: Full 3D spacing.
        """
        from tiamat.readers.processing import expand_to_length

        if slice_spacing is None:
            spacing_2d = expand_to_length(spacing, 2)
            assert (
                spacing_2d[0] == spacing_2d[1]
            ), "StackReader assumes isotropic image spacing if slice_spacing is not provided"
            return (*spacing_2d, spacing_2d[0])
        else:
            return (*expand_to_length(spacing, 2), slice_spacing)

    def selected_slice_handles(self, slice_ix: Iterable[int]) -> list[ImageReader]:
        """
        select slice readers for given indices.

        Args:
        slice_ix (Iterable[int]): Slice indices.

        Returns:
        list[ImageReader]: Readers for selected slices.
        """
        from tiamat.readers.memory import ConstantReader

        selected_slices = [self.slices[i] for i in slice_ix]

        reader_list = []
        if isinstance(self.reader_factory, dict):
            # Loop over all selected files and assign corresponding reader via the identifier

            for fname in selected_slices:
                if fname is None:
                    # Missing section
                    reader_list.append(ConstantReader(self.missing_section_fill_value, self.prototype_metadata))
                else:
                    # Existing section needs corresponding reader
                    identifier = get_reader_identifier(fname, self.compiled_identifier)
                    factory = self.reader_factory[identifier]
                    reader_list.append(factory(fname))

        elif isinstance(self.reader_factory, (tuple, list)):
            # Map each slice to reader at same slice index
            selected_readers = [self.reader_factory[i] for i in slice_ix]
            for factory, fname in zip(selected_readers, selected_slices):
                reader_list.append(factory(fname))

        else:
            # Use same reader for all selected slices
            for fname in selected_slices:
                if fname is None:
                    reader_list.append(ConstantReader(self.missing_section_fill_value, self.prototype_metadata))
                else:
                    reader_list.append(self.reader_factory(fname))

        return reader_list

    @property
    def ordered_slice_handles(self):
        """All slice handles in order."""
        return self.selected_slice_handles(range(len(self.slices)))

    @property
    def prototype_slice_handle(self) -> ImageReader:
        """Prototype slice handle for metadata inspection."""
        selected_handles = self.selected_slice_handles([0])

        return selected_handles[0]

    @instance_cached_property
    def prototype_metadata(self) -> ImageMetadata:
        """Metadata from a prototype slice."""
        metadata = self.prototype_slice_handle.read_metadata()

        return metadata

    @instance_cache
    def read_metadata(self) -> ImageMetadata:
        """Read metadata for the stack."""
        from dataclasses import replace

        # Load metadata from an arbitrary prototype slice
        metadata = replace(self.prototype_metadata)

        # Set file path
        metadata.file_path = self.fnames

        # Expand metadata for stack by simply expanding the shape
        metadata.shape = tuple([self.num_slices, *metadata.shape])

        # Set spacing
        metadata.spacing = ImageStackReader.fill_spacing(self.slice_spacing, metadata.spacing)

        # Set scales
        metadata.scales = self.scales

        metadata.dimensions = [
            self.stack_dimension,
        ] + list(metadata.dimensions)

        metadata.additional_metadata["stack_dimension"] = self.stack_dimension

        return metadata

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        """Read stacked image for given accessor."""
        from dataclasses import replace

        metadata = self.read_metadata()

        # Request only subset of slice handles neded for the requested scale
        selected_slice_ix = select_slice_ix(accessor, metadata, self.num_slices)
        slice_handles = self.selected_slice_handles(selected_slice_ix)

        if len(slice_handles) == 0:
            raise Exception("Requested empty stack")

        # Remove z axis for 2D access from metadata
        tmp_accessor = replace(accessor, z=None)
        # if scale is tuple, remove z scale
        if isinstance(tmp_accessor.scale, (tuple, list)):
            if self.stack_dimension == dimensions.X:
                tmp_accessor.scale = tmp_accessor.scale[1:]
            elif self.stack_dimension == dimensions.Y:
                tmp_accessor.scale = (tmp_accessor.scale[0], *tmp_accessor.scale[2:])
            elif self.stack_dimension == dimensions.Z:
                tmp_accessor.scale = tmp_accessor.scale[:2]

        # TODO: Check if unnecessary and remove
        # metadata = replace(tmp_accessor.metadata)
        # metadata.shape = metadata.shape[1:]
        # dimension_list = list(metadata.dimensions)
        # dimension_list.remove(self.stack_dimension)

        # # for each scale, remove the z scale
        # if metadata.scales is not None:
        #     scales = list(metadata.scales)
        #     for i, s in enumerate(scales):
        #         if self.stack_dimension == dimensions.X:
        #             scales[i] = s[1:]
        #         elif self.stack_dimension == dimensions.Y:
        #             scales[i] = (s[0], *s[2:])
        #         elif self.stack_dimension == dimensions.Z:
        #             scales[i] = s[:2]
        #         else:
        #             raise ValueError(f"Unknown stack dimension {self.stack_dimension}")
        #     metadata.scales = tuple(scales)

        # metadata.dimensions = dimension_list
        # tmp_accessor.metadata = metadata

        first_result = slice_handles[0].read_image(accessor=tmp_accessor)
        # For efficiency, create empty array first, then write remaining data into arrays.
        image = np.empty(
            shape=([len(slice_handles), *first_result.shape]),
            dtype=first_result.dtype,
        )

        # Reuse first result
        image[0] = first_result
        # Read and stack all remaining images.
        for i, handle in enumerate(slice_handles[1:], 1):
            image[i] = handle.read_image(accessor=tmp_accessor)

        return image

    @property
    def file_handle(self) -> ImageReader:
        """Representative file handle."""
        return self.prototype_slice_handle

    @cached_property
    def slices(self) -> list[str | None]:
        """Return list of slices, optionally with interpolation for missing sections."""
        if hasattr(self.fnames, "__iter__") and not isinstance(self.fnames, str):
            available_slices = list(self.fnames)
        else:
            available_slices = find_slices(fnames=self.fnames)

        if self.compiled_identifier is None:
            if isinstance(self.reader_factory, dict):
                raise Exception("Using a dictionary as reader_factory requires a reader_identifier to be provided")
            if self.missing_section_interpolation is None:
                return available_slices
            else:
                raise Exception(
                    f"{self.missing_section_interpolation} missing_section_interpolation"
                    "requires reader_identifier to be provided"
                )
        else:
            # Sort available slices by their reader_identifier
            if isinstance(self.reader_factory, dict):
                available_slices = [
                    f
                    for f in available_slices
                    if get_reader_identifier(f, self.compiled_identifier) in self.reader_factory.keys()
                ]

            available_keys = [int(get_reader_identifier(f, self.compiled_identifier)) for f in available_slices]
            sorted_ix = np.argsort(available_keys)

            if self.missing_section_interpolation is None:
                # Only return available slices ordered by their identifier
                ordered_slices = [available_slices[k] for k in sorted_ix]
            else:
                # Perform interpolation of gaps between ordered slices
                ordered_slices = [available_slices[sorted_ix[0]]]

                for i in range(len(available_keys) - 1):
                    gap = int(available_keys[sorted_ix[i + 1]] - available_keys[sorted_ix[i]])
                    missing = max(gap - 1, 0)

                    if missing >= 1:
                        if self.missing_section_interpolation.lower() == "nearest":
                            # Nearest neighbor interplation of missing slices
                            ordered_slices += [available_slices[sorted_ix[i]]] * math.ceil(missing / 2)
                            ordered_slices += [available_slices[sorted_ix[i + 1]]] * math.floor(missing / 2)
                        elif self.missing_section_interpolation.lower() == "constant":
                            # Fill gaps with None (will be filled with fill value later)
                            ordered_slices += [None] * missing
                        else:
                            raise AttributeError(
                                f"Unknown missing_section_interpolation: {self.missing_section_interpolation}"
                            )

                    ordered_slices.append(available_slices[sorted_ix[i + 1]])

            return ordered_slices

    @cached_property
    def num_slices(self) -> int:
        """Number of slices."""

        return len(self.slices)

    @classmethod
    def check_file(cls, fname: str | list[str]) -> bool | int | float:
        # StackReader requires initialization before being able to check the files
        # TODO: Maybe check if fname refers to a list of files. Check if any readers
        # exists for this filetype and return this one
        return False

    @classmethod
    def from_json(cls, args: dict[str, Any], reader_post_creation_hook=None) -> Callable:
        """Construct reader from JSON config."""
        from tiamat.serialization import get_reader_from_config

        reader_factory = args.get("reader_factory")

        if isinstance(reader_factory, dict):
            if "class" in reader_factory.keys():
                # Single reader
                reader = get_reader_from_config(reader_factory, reader_post_creation_hook=reader_post_creation_hook)
            else:
                # Stack of readers
                reader = dict(
                    (k, get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook))
                    for k, r in reader_factory.items()
                )
        elif hasattr(reader_factory, "__iter__"):
            reader = tuple(
                get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook) for r in reader_factory
            )
        elif reader_factory is None:
            reader = get_reader_from_config(reader_factory, reader_post_creation_hook=reader_post_creation_hook)
        else:
            raise Exception(f"Can't parse reader {reader}")

        if reader_post_creation_hook is None:
            return partial(
                cls,
                reader_factory=reader,
                slice_spacing=float(args.get("slice_spacing")),
                reader_identifier=args.get("reader_identifier"),
                stack_dimension=args.get("stack_dimension", dimensions.Z),
                missing_section_interpolation=args.get("missing_section_interpolation"),
                missing_section_fill_value=args.get("missing_section_fill_value", 0),
            )
        else:
            return partial(
                reader_post_creation_hook,
                cls,
                reader_factory=reader,
                slice_spacing=float(args.get("slice_spacing")),
                reader_identifier=args.get("reader_identifier"),
                stack_dimension=args.get("stack_dimension", dimensions.Z),
                missing_section_interpolation=args.get("missing_section_interpolation"),
                missing_section_fill_value=args.get("missing_section_fill_value", 0),
            )

    @cached_property
    def scales(self) -> list[tuple[float, float, float]] | None:
        """Return scales of the image stack.
        To improve io, the scale along the stacked axis is set to 1.0
        """

        # for now, assert scale is same for all slices
        metadata_first_slice = self.prototype_slice_handle.read_metadata()

        if not hasattr(metadata_first_slice, "scales"):
            return None
        if metadata_first_slice.scales is None:
            return None

        spacing_3d = ImageStackReader.fill_spacing(self.slice_spacing, metadata_first_slice.spacing)
        spacing = min(spacing_3d[0], spacing_3d[1])
        slice_spacing = spacing_3d[2]

        # TODO: make shure position of z is correct
        scales = metadata_first_slice.scales

        min_scale = 1.0 / self.num_slices

        if isinstance(scales[0], Iterable):
            scales = [(*s[:2], max(min((slice_spacing * s[0]) / spacing, 1.0), min_scale), *s[2:]) for s in scales]
        elif isinstance(scales[0], (int, float)):
            scales = [(s, s, max(min((slice_spacing * s) / spacing, 1.0), min_scale)) for s in scales]

        return scales


class VolumeStackReader(ImageReader):
    """
    Reader for stacks of sub-volumes forming a larger 3D volume.


    This reader concatenates multiple 3D sub-volumes along the Z axis (or, more
    generally, along the first spatial dimension reported by the prototype
    metadata), producing a single larger volume. Each sub-volume is read via a
    provided reader factory and stitched together on-the-fly during reads.


    Notes:
    - If a mapping (``dict``) of reader factories is provided, files are
    grouped by an identifier extracted from their filenames using
    ``reader_identifier`` (a regex with a capture group).
    - If multiple files match the same identifier key, the corresponding
    factory is called with a ``tuple[str, ...]`` (see code comments).
    """

    def __init__(
        self,
        fnames: str | Iterable[str],
        flag_const_shape: bool = False,
        reader_identifier: str | None = None,
        reader_factory: (
            Callable[[str], ImageReader]
            | Iterable[Callable[[str], ImageReader]]
            | dict[str, Callable[[str], ImageReader]]
        ) = None,
        reader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a VolumeStackReader.

        Args:
            fnames (str | Iterable[str]): File pattern or iterable of filenames.
            flag_const_shape (bool, optional): Whether all sub-volumes have the same shape. Defaults to False.
            reader_identifier (str | None, optional): Regex for identifying files in a dict of factories.
            reader_factory (Callable | Iterable | Dict, optional): Factory or factories to create sub-volume readers.
            reader_kwargs (dict[str, Any], optional): Additional arguments passed to the factories.
        """

        self.fnames = fnames
        self.flag_const_shape = flag_const_shape
        self.reader_factory = reader_factory or get_reader

        if isinstance(self.reader_factory, dict):
            assert reader_identifier is not None
        self.compiled_identifier = compile_identifier(reader_identifier)

        self.reader_kwargs = reader_kwargs or {}

    @cached_property
    def slices(self) -> list[str]:
        """
        Return an ordered list of sub-volume file specifications.

        If ``fnames`` is an iterable (and not a string), it is returned as a
        list. Otherwise, the glob pattern in ``fnames`` is expanded and sorted.

        Returns:
            list[str]: List of file paths (or path specs) for sub-volumes.
        """
        if hasattr(self.fnames, "__iter__") and not isinstance(self.fnames, str):
            return [fname for fname in self.fnames]
        else:
            return find_slices(fnames=self.fnames)

    @cached_property
    def num_slices(self) -> int:
        """
        Number of sub-volumes.

        Returns:
            int: Number of entries in :pyattr:`slices`.
        """
        return len(self.slices)

    @cached_property
    def _file_names_per_reader_identifier(self):
        """
        Map files to their reader identifiers.

        Returns:
            dict[str, list[str]]: Mapping of identifier -> list of matching file paths.
        """
        file_matches = defaultdict(list)
        for fname in self.slices:
            file_matches[get_reader_identifier(fname, self.compiled_identifier)].append(fname)
        return file_matches

    @property
    def ordered_subvolume_handles(self) -> list[ImageReader]:
        """
        Create ImageReader handles for all sub-volumes, in order.

        Behavior depends on `reader_factory` type:
        - dict[str, Callable]: For each sorted key, find matching files.
          Single or multiple files are passed to the factory.
        - Iterable[Callable]: Zip slices and factories.
        - Callable: Use the same factory for all slices.

        Returns:
            list[ImageReader]: List of instantiated sub-volume readers.

        Notes:
        - If a key maps to multiple files, the factory must accept a
        sequence of paths.
        # NOTE: This is implicit in the original code path.
        """
        if isinstance(self.reader_factory, dict):
            reader_list = []
            for k in sorted(self.reader_factory.keys()):
                factory = self.reader_factory[k]
                # Find all files that match k
                file_matches = tuple(self._file_names_per_reader_identifier[k])
                if len(file_matches) > 1:
                    reader_list.append(factory(file_matches))
                if len(file_matches) == 1:
                    reader_list.append(factory(file_matches[0]))
            return reader_list

        elif hasattr(self.reader_factory, "__iter__"):
            return [factory(fname, **self.reader_kwargs) for fname, factory in zip(self.slices, self.reader_factory)]

        else:
            return [self.reader_factory(fname, **self.reader_kwargs) for fname in self.slices]

    @property
    def prototype_subvolume_handle(self) -> ImageReader:
        """
        Return the first sub-volume handle as a prototype for metadata.

        Returns:
            ImageReader: The first instantiated sub-volume reader.
        """
        return self.ordered_subvolume_handles[0]

    @cached_property
    def subvolume_shapes(self) -> list[tuple[int, ...]]:
        """
        Return the shapes of all sub-volumes.

        Returns:
            list[tuple[int, ...]]: A list of shape tuples for each sub-volume.
        """

        subvolume_handles = self.ordered_subvolume_handles

        if self.flag_const_shape:
            metadata = subvolume_handles[0].read_metadata()
            # logger.debug("VolumeStackReader subvolume_shapes: %s", metadata.shape)
            return [(metadata.shape) for _ in range(len(subvolume_handles))]

        return [handle.read_metadata().shape for handle in subvolume_handles]

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """
        Compute the composite volume shape produced by concatenating all
        sub-volumes along the Z dimension (first entry of spatial dims).

        Returns:
            tuple[int, ...]: Shape of the stitched volume.
        """
        metadata = self.ordered_subvolume_handles[0].read_metadata()
        z_dim = metadata.spatial_dimensions[0]

        shape = list(self.subvolume_shapes[0])
        for s in self.subvolume_shapes[1:]:
            shape[z_dim] = shape[z_dim] + s[z_dim]

        return tuple(shape)

    @instance_cache
    def read_metadata(self) -> ImageMetadata:
        """
        Read and compose metadata for the stacked volume.

        Returns:
            ImageMetadata: Metadata of the stitched volume.
        """
        from dataclasses import replace

        metadata = replace(self.ordered_subvolume_handles[0].read_metadata())

        metadata.file_path = self.fnames

        metadata.shape = self.shape

        return metadata

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        """
        Read the stitched volume given an accessor.

        The method determines the intersection of the requested Z-slice with
        each sub-volume, reads those ranges, and inserts them into an output
        buffer. The output shape along Z is determined by the requested range
        and scale factor.

        Args:
            accessor (ImageAccessor): Accessor specifying coordinates (including Z-range) and scale.

        Returns:
            np.ndarray: Stitched image volume.

        Raises:
            ValueError: If no sub-volume handles are available.
        """
        import math
        from dataclasses import replace

        from tiamat.readers.processing import (
            expand_to_length,
            prepare_coordinate,
        )
        from tiamat.transformers.coordinates import resolve_coordinate_slice

        metadata = self.read_metadata()
        z_dim, y_dim, x_dim = metadata.spatial_dimensions

        image_z_size = self.shape[z_dim]
        image_scales = expand_to_length(accessor.scale, 3)  # x, y, z

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
                tmp_image = handle.read_image(accessor=tmp_accessor)

                # Position in the output array to place the image
                scaled_z_offset = math.floor(max(cur_z_offset - z_from, 0) * image_scales[-1])

                # Insert the result in the array at specific offfset
                insert_array(tmp_image, scaled_z_offset)

                # Output image might cover more than z_size
                cur_z_offset += z_size  # tmp_image.shape[z_dim] / image_scales[-1]
            else:
                cur_z_offset += z_size

        return out_image

    @classmethod
    def check_file(cls, fname: str | list[str]) -> bool | int | float:
        # StackReader requires initialization before being able to check the files
        # TODO: Maybe check if fname refers to a list of files. Check if any readers
        # exists for this filetype and return this one
        return False

    @classmethod
    def from_json(cls, args: dict[str, Any], reader_post_creation_hook=None) -> Callable:
        """
        Construct a VolumeStackReader (or wrapper) from a JSON-like config.

        Args:
            args (dict): Configuration dictionary.
            reader_post_creation_hook (Callable | None): Optional wrapper hook.

        Returns:
            Callable: Partial that instantiates the reader or invokes the hook.

        Raises:
            ValueError: If the ``reader_factory`` cannot be parsed.
        """

        from tiamat.serialization import get_reader_from_config

        reader_factory = args.get("reader_factory")

        if isinstance(reader_factory, dict):
            if "class" in reader_factory.keys():
                # Single reader
                reader = get_reader_from_config(reader_factory, reader_post_creation_hook=reader_post_creation_hook)
            else:
                # Stack of readers
                reader = dict(
                    (k, get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook))
                    for k, r in reader_factory.items()
                )
        elif hasattr(reader_factory, "__iter__"):
            reader = tuple(
                get_reader_from_config(r, reader_post_creation_hook=reader_post_creation_hook) for r in reader_factory
            )
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
