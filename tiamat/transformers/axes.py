"""
Transforms manipulating individual axes of images
"""

from dataclasses import replace
from typing import Any

import numpy as np

from tiamat.transformers.coordinates import resolve_coordinate_slice

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import Transformer


class ImageToVolumeTransformer(Transformer):
    """
    Transformer that adds a new Z-axis to 2D images, expanding (y, x) to (z, y, x).
    """

    def __init__(self, z_spacing: float = None) -> None:
        """
        Initialize an ImageToVolumeTransformer.

        Args:
            z_spacing: Spacing for the new Z-axis. Defaults to 0.0.
        """
        self.z_spacing = z_spacing

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        No changes to accessor; only metadata and image shape are updated.

        Args:
            accessor: ImageAccessor to pass through.
            metadata: Metadata of the image.

        Returns:
            The unchanged accessor.
        """
        # dimensional metadata has to be removed
        accessor = replace(accessor)
        accessor.z = None

        if accessor.scale is not None and hasattr(accessor.scale, "__len__"):
            accessor.scale = accessor.scale[:2]

        if accessor.spacing is not None and hasattr(accessor.spacing, "__len__"):
            accessor.spacing = accessor.spacing[:2]

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Expand 2D metadata to 3D by inserting a Z-axis.

        Args:
            metadata: Original ImageMetadata.

        Returns:
            Updated ImageMetadata with new Z-axis.
        """
        from dataclasses import replace

        from tiamat.metadata import dimensions
        from tiamat.readers.processing import expand_to_length

        metadata = replace(metadata)

        # this transformer always expands y, x to z,y,x
        # we search for the position of y, then prepend a 1-dimension
        y_index = metadata.dimensions.index(dimensions.Y)
        new_axis = max(y_index - 1, 0)

        # Insert new spatial axis
        new_shape = list(metadata.shape)
        new_shape.insert(new_axis, 1)
        metadata.shape = tuple(new_shape)

        # Insert the axis to the dimensions
        new_dimensions = list(metadata.dimensions)
        new_dimensions.insert(new_axis, dimensions.Z)
        metadata.dimensions = new_dimensions

        metadata.additional_metadata["stack_dimension"] = dimensions.Z

        # Do not provide downsampled versions in z direction
        metadata.scales = [(*expand_to_length(s, 2), 1.0) for s in metadata.scales]

        if self.z_spacing is None:
            if hasattr(metadata.spacing, "__len__"):
                raise Exception(f"Need provide z_spacing for non-uniform spacing {metadata.spacing}")
        else:
            if hasattr(metadata.spacing, "__len__"):
                metadata.spacing = (*metadata.spacing, self.z_spacing)
            else:
                metadata.spacing = (*expand_to_length(metadata.spacing, 2), self.z_spacing)

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Expand 2D image to 3D by adding a singleton Z dimension.

        Args:
            image: Original Image.
            metadata: The images metadata.
            accessor: The accessor of the image.

        Returns:
            Updated ImageResult with new Z-axis.
        """
        from tiamat.metadata import dimensions

        # this transformer always expands y, x to z,y,x
        # we search for the position of y, then prepend a 1-dimension
        y_index = metadata.dimensions.index(dimensions.Y)
        new_axis = max(y_index - 1, 0)
        new_shape = list(image.shape)
        new_shape.insert(new_axis, 1)

        image = image.reshape(new_shape)

        return image

    @classmethod
    def from_json(cls, args: dict[str, Any]) -> "ImageToVolumeTransformer":
        """
        Create an ImageToVolumeTransformer from JSON arguments.

        Args:
            args: Dictionary with optional key 'z_spacing'.

        Returns:
            ImageToVolumeTransformer instance.
        """
        return cls(
            z_spacing=float(args.get("z_spacing", None)),
        )


class ReorderCoordinatesTransformer(Transformer):
    """
    Transformer that reorders spatial axes of images and metadata.
    """

    def __init__(self, axes: tuple[str, ...] = ("x", "y", "z")) -> None:
        """
        Initialize a ReorderCoordinatesTransformer.

        Args:
            axes: Desired axis order, either 2D ('x', 'y') or 3D ('x', 'y', 'z').
        """
        from tiamat.metadata.dimensions import META_DIMENSIONS, SPATIAL_DIMENSIONS

        assert len(axes) == 2 or len(axes) == 3

        self.reorder_axes = axes

        if len(axes) == 2:
            in_axes = SPATIAL_DIMENSIONS[1:]  # ('y', 'x')
            meta_axes = META_DIMENSIONS[:-1]  # ('x', 'y')
        else:
            in_axes = SPATIAL_DIMENSIONS  # ('z', 'y', 'x')
            meta_axes = META_DIMENSIONS  # ('x', 'y', 'z')

        # Determine forward and backward indices for z, y, x ordered
        self.from_indices = tuple(in_axes.index(a) for a in self.reorder_axes)
        self.to_indices = tuple(self.reorder_axes.index(a) for a in in_axes)

        # Invert for x, y, z ordered values (scale, spacing, ...)
        # Assume that the given axes have z, y, x ordering
        self.from_indices_meta = tuple(meta_axes.index(a) for a in self.reorder_axes[::-1])
        self.to_indices_meta = tuple(self.reorder_axes[::-1].index(a) for a in meta_axes)

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Reorder coordinates in accessor according to the specified axes.

        Args:
            accessor: ImageAccessor to transform.
            metadata: The metadata.

        Returns:
            Updated ImageAccessor with reordered coordinates and scales.
        """
        from dataclasses import replace

        from tiamat.readers.processing import expand_to_length

        accessor = replace(accessor)

        scale = expand_to_length(accessor.scale, 3)

        if len(self.reorder_axes) == 2:
            coord_slices = [accessor.y, accessor.x]
            accessor.y = coord_slices[self.to_indices[0]]
            accessor.x = coord_slices[self.to_indices[1]]

            accessor.scale = (
                scale[self.to_indices_meta[0]],
                scale[self.to_indices_meta[1]],
            )
        else:
            coord_slices = [accessor.z, accessor.y, accessor.x]
            accessor.z = coord_slices[self.to_indices[0]]
            accessor.y = coord_slices[self.to_indices[1]]
            accessor.x = coord_slices[self.to_indices[2]]

            accessor.scale = (
                scale[self.to_indices_meta[0]],
                scale[self.to_indices_meta[1]],
                scale[self.to_indices_meta[2]],
            )

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Reorder metadata shape, spacing, and scales according to axes.

        Args:
            metadata: Original ImageMetadata.

        Returns:
            Updated ImageMetadata.
        """
        from dataclasses import replace

        from tiamat.metadata.dimensions import SPATIAL_DIMENSIONS
        from tiamat.readers.processing import expand_to_length

        metadata = replace(metadata)

        # Adjust shape of image
        sp_dims = metadata.spatial_dimensions
        assert len(sp_dims) == len(self.reorder_axes)

        new_shape = list(metadata.shape)
        for i, ix in enumerate(self.from_indices):
            new_shape[sp_dims[i]] = metadata.shape[sp_dims[ix]]
        metadata.shape = tuple(new_shape)

        # Adjust stack dimension if present
        if "stack_dimension" in metadata.additional_metadata.keys():
            stack_dimension = metadata.additional_metadata["stack_dimension"]
            stack_ix = self.reorder_axes.index(stack_dimension)
            if len(self.reorder_axes) == 2:
                metadata.additional_metadata["stack_dimension"] = SPATIAL_DIMENSIONS[1:][stack_ix]
            else:
                metadata.additional_metadata["stack_dimension"] = SPATIAL_DIMENSIONS[stack_ix]

        # Adjust spacing and scale
        spacing = list(expand_to_length(metadata.spacing, len(sp_dims)))
        scales = [list(expand_to_length(s, len(sp_dims))) for s in metadata.scales]

        new_spacing = spacing.copy()
        new_scales = [s.copy() for s in scales]
        for i, ix in enumerate(self.from_indices_meta):
            new_spacing[i] = spacing[ix]

            for j in range(len(scales)):
                new_scales[j][i] = scales[j][ix]

        metadata.spacing = tuple(new_spacing)
        metadata.scales = [tuple(s) for s in new_scales]

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Reorder axes of the image array.

        Args:
            image: Image to transform.
            metadata: Metadata of the image.
            accessor: Accessor of the image.

        Returns:
            Updated Image with axes reordered.
        """
        sp_dims = metadata.spatial_dimensions
        assert len(sp_dims) == len(self.reorder_axes)

        image = np.moveaxis(
            image,
            [sp_dims[i] for i in self.from_indices],
            sp_dims,
        )

        return image

    @classmethod
    def from_json(cls, args: dict[str, Any]) -> "ReorderCoordinatesTransformer":
        """
        Create a ReorderCoordinatesTransformer from JSON arguments.

        Args:
            args: Dictionary with optional key 'axes'.

        Returns:
            ReorderCoordinatesTransformer instance.
        """
        return cls(
            axes=tuple(args.get("axes", ("x", "y", "z"))),
        )


class MirrorTransformer(Transformer):
    """
    Transformer that mirrors images along specified axes.
    """

    def __init__(self, mirror_x: bool = False, mirror_y: bool = False, mirror_z: bool = False):
        """
        Initialize a MirrorTransformer.

        Args:
            mirror_x: Whether to mirror along the X axis.
            mirror_y: Whether to mirror along the Y axis.
            mirror_z: Whether to mirror along the Z axis.
        """
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
        self.mirror_z = mirror_z

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Mirror accessor coordinates along specified axes.

        Args:
            accessor: ImageAccessor to transform.
            metadata: Metadata from the image.

        Returns:
            Updated ImageAccessor with mirrored coordinates.
        """
        from dataclasses import replace

        from tiamat.metadata import dimensions

        accessor = replace(accessor)

        if self.mirror_x:
            x_ix = metadata.dimensions.index(dimensions.X)
            x_size = metadata.shape[x_ix]
            x_from, x_to = resolve_coordinate_slice(accessor.x, x_size)
            accessor.x = (x_size - x_to, x_size - x_from)

        if self.mirror_y:
            y_ix = metadata.dimensions.index(dimensions.Y)
            y_size = metadata.shape[y_ix]
            y_from, y_to = resolve_coordinate_slice(accessor.y, y_size)
            accessor.y = (y_size - y_to, y_size - y_from)

        if len(metadata.spatial_dimensions) > 2 and self.mirror_z:
            z_ix = metadata.dimensions.index(dimensions.Z)
            z_size = metadata.shape[z_ix]
            z_from, z_to = resolve_coordinate_slice(accessor.z, z_size)
            accessor.z = (z_size - z_to, z_size - z_from)

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Mirroring does not affect metadata.

        Args:
           metadata: Metadata of the image.

        Returns:
           The unchanged metadata.
        """
        from dataclasses import replace

        metadata = replace(metadata)

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Mirror the image along the specified axes.

        Args:
            image: ImageResult to transform.
            metadata: Metadata of the image.
            accessor: Accessor to the image.

        Returns:
            Mirrored Image.
        """
        s_dim = metadata.spatial_dimensions
        if len(s_dim) == 2:
            flip_axes = self.mirror_y * [s_dim[0]] + self.mirror_x * [s_dim[1]]
        else:
            flip_axes = self.mirror_z * [s_dim[0]] + self.mirror_y * [s_dim[1]] + self.mirror_x * [s_dim[2]]

        image = np.flip(image, axis=flip_axes)

        return image

    @classmethod
    def from_json(cls, args: dict[str, Any]) -> "MirrorTransformer":
        """
        Create a MirrorTransformer from JSON arguments.

        Args:
            args: Dictionary with optional keys 'mirror_x', 'mirror_y', 'mirror_z'.

        Returns:
            MirrorTransformer instance.
        """
        return cls(
            mirror_x=bool(args.get("mirror_x", False)),
            mirror_y=bool(args.get("mirror_y", False)),
            mirror_z=bool(args.get("mirror_z", False)),
        )
