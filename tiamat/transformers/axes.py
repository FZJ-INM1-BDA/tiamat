"""
Transforms manipulating individual axes of images
"""

from typing import Any, Dict
from .protocol import Transformer
from ..io import ImageResult, ImageAccessor
from ..metadata import ImageMetadata
from tiamat.transformers.coordinates import resolve_coordinate_slice


class TransposeTransformer(Transformer):

    def __init__(self, axes=(0, 1, 2)):
        import numpy as np

        # Assume exactly 3 dimensions in (z, y, x) order for now.
        # TODO: Should be made dynamic with an update of ImageAccessor supporting variable dimension order and channel axis
        assert len(axes) == 3, "Values for all three axes (z, y, x) need to be provided"
        assert set(axes).difference(set((0, 1, 2))) == set(), "Need to provide all values in (0, 1, 2)"

        self.reorder_axes = axes
        self.indices_axes = tuple(np.argsort(axes))

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        from dataclasses import replace

        coord_slices = [accessor.z, accessor.y, accessor.x]

        accessor = replace(accessor)
        accessor.z = coord_slices[self.indices_axes[0]]
        accessor.y = coord_slices[self.indices_axes[1]]
        accessor.x = coord_slices[self.indices_axes[2]]

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        # TODO: Shape should be changed in image metadata as well!
        return metadata

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        import numpy as np

        metadata = image_result.accessor.metadata
        assert metadata is not None, "CPNTransformer requires metadata."

        # TODO: Only support 2D for now, remove [1:] later if 3D supported
        image_result.image = np.moveaxis(
            image_result.image,
            [metadata.spatial_dimensions[i] for i in self.reorder_axes[1:]],
            metadata.spatial_dimensions,
        )
        # image_result.image = np.transpose(image_result.image, axes=self.reorder_axes[1:])

        return image_result
    
    @classmethod
    def from_json(cls, args: Dict[str, Any]):
        return cls(
            axes=args.get("axes", (0, 1, 2)),
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
        import numpy as np

        metadata = image_result.accessor.metadata
        assert metadata is not None, "CPNTransformer requires metadata."

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
