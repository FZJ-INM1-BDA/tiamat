"""
Transforms manipulating individual axes of images
"""

from typing import Any, Dict

import numpy as np

from .protocol import Transformer
from ..io import ImageResult, ImageAccessor
from ..metadata import ImageMetadata
from tiamat.transformers.coordinates import resolve_coordinate_slice


class ImageToVolumeTransformer(Transformer):

    def __init__(self, z_spacing=None):
        self.z_spacing = z_spacing

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        from dataclasses import replace
        from tiamat.metadata import dimensions

        metadata = replace(metadata)

        # this transformer always expands y, x to z,y,x
        # we search for the position of y, then prepend a 1-dimension
        y_index = metadata.dimensions.index(dimensions.Y)
        new_axis = max(y_index - 1, 0)
        new_shape = list(metadata.shape)
        new_shape.insert(new_axis, 1)
        metadata.shape = new_shape

        if self.z_spacing is None:
            if hasattr(metadata.spacing, '__len__'):
                raise Exception(f"Need provide z_spacing for non-uniform spacing {metadata.spacing}")
        else:
            if hasattr(metadata.spacing, '__len__'):
                metadata.spacing = (self.z_spacing, *metadata.spacing)
            else:
                from tiamat.readers.processing import expand_to_length

                metadata.spacing = (self.z_spacing, *expand_to_length(metadata.spacing, 2))

        return metadata

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        metadata = image_result.metadata
        assert metadata is not None, "ImageToVolumeTransformer requires metadata."
        
        # this transformer always expands y, x to z,y,x
        # we search for the position of y, then prepend a 1-dimension
        y_index = metadata.dimensions.index(dimensions.Y)
        new_axis = max(y_index - 1, 0)
        new_shape = list(metadata.shape)
        image_result.image = image_result.image.reshape(new_shape)

        return image_result

    @classmethod
    def from_json(cls, args: Dict[str, Any]):
        return cls(
            z_spacing=float(args.get("z_spacing", None)),
        )


class ReorderCoordinatesTransformer(Transformer):

    def __init__(self, axes=('x', 'y', 'z')):
        assert len(axes) == 2 or len(axes) == 3

        self.reorder_axes = axes

        if len(axes) == 2:
            in_axes = ('y', 'x')
        else:
            in_axes = ('z', 'y', 'x')

        self.from_indices = tuple(in_axes.index(a) for a in self.reorder_axes)
        self.to_indices = tuple(self.reorder_axes.index(a) for a in in_axes)

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        from dataclasses import replace

        accessor = replace(accessor)

        if len(self.reorder_axes) == 2:
            coord_slices = [accessor.y, accessor.x]
            accessor.y = coord_slices[self.to_indices[0]]
            accessor.x = coord_slices[self.to_indices[1]]
        else:
            coord_slices = [accessor.z, accessor.y, accessor.x]
            accessor.z = coord_slices[self.to_indices[0]]
            accessor.y = coord_slices[self.to_indices[1]]
            accessor.x = coord_slices[self.to_indices[2]]

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        from dataclasses import replace
        
        metadata = replace(metadata)

        sp_dims = metadata.spatial_dimensions

        assert len(sp_dims) == len(self.reorder_axes)

        new_shape = list(metadata.shape)
        for i, ix in enumerate(self.from_indices):
            new_shape[sp_dims[i]] = metadata.shape[sp_dims[ix]]
        metadata.shape = tuple(new_shape)

        if hasattr(metadata.spacing, '__len__'):
            assert len(metadata.spacing) == len(sp_dims)
            metadata.spacing = tuple(metadata.spacing[ix] for ix in self.from_indices[::-1])

        return metadata

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        metadata = image_result.metadata
        assert metadata is not None, "MirrorTransformer requires metadata."

        sp_dims = metadata.spatial_dimensions
        assert len(sp_dims) == len(self.reorder_axes)

        image_result.image = np.moveaxis(
            image_result.image,
            [sp_dims[i] for i in self.from_indices],
            sp_dims,
        )

        return image_result
    
    @classmethod
    def from_json(cls, args: Dict[str, Any]):
        return cls(
            axes=tuple(args.get("axes", ('x', 'y', 'z'))),
        )


class MirrorTransformer(Transformer):

    def __init__(self, mirror_x=False, mirror_y=False, mirror_z=False):
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y
        self.mirror_z = mirror_z

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        from dataclasses import replace

        spatial_shape = accessor.metadata.spatial_shape

        accessor = replace(accessor)

        if self.mirror_x:
            x_from, x_to = resolve_coordinate_slice(accessor.x, spatial_shape[-1])
            accessor.x = (spatial_shape[-1] - x_to, spatial_shape[-1] - x_from)

        if self.mirror_y:
            y_from, y_to = resolve_coordinate_slice(accessor.y, spatial_shape[-2])
            accessor.y = (spatial_shape[-2] - y_to, spatial_shape[-2] - y_from)

        if len(spatial_shape) > 2 and self.mirror_z:
            z_from, z_to = resolve_coordinate_slice(accessor.z, spatial_shape[-3])
            accessor.z = (spatial_shape[-3] - z_to, spatial_shape[-3] - z_from)

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        return metadata

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        metadata = image_result.metadata
        assert metadata is not None, "MirrorTransformer requires metadata."

        s_dim = metadata.spatial_dimensions
        if len(s_dim) == 2:
            flip_axes = self.mirror_y * [s_dim[0]] + self.mirror_x * [s_dim[1]]
        else :
            flip_axes = self.mirror_z * [s_dim[0]] + self.mirror_y * [s_dim[1]] + self.mirror_x * [s_dim[2]]

        image_result.image = np.flip(image_result.image, axis=flip_axes)

        return image_result

    @classmethod
    def from_json(cls, args: Dict[str, Any]):
        return cls(
            mirror_x=bool(args.get("mirror_x", False)),
            mirror_y=bool(args.get("mirror_y", False)),
            mirror_z=bool(args.get("mirror_z", False)),
        )
