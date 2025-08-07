"""
Transformers that change the output view
"""

from typing import Tuple

from tiamat.metadata import dimensions
from .protocol import Transformer
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class BoundingBoxTransformer(Transformer):
        
    def __init__(
        self,
        bounds_x: Tuple[int | float | None, int | float | None] | int = None,
        bounds_y: Tuple[int | float | None, int | float | None] | int = None,
        bounds_z: Tuple[int | float | None, int | float | None] | int = None,
    ):
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.bounds_z = bounds_z

    @staticmethod
    def crop_coordinate(coord_slice, bounds_slice, image_dimension, coord_scale=1.):
        import math
        from tiamat.transformers.coordinates import get_coordinate_bounds
        from tiamat.readers.processing import prepare_coordinate

        # Here image_scale=coord_scale scales bounds to slice scale
        bounds_slice = prepare_coordinate(bounds_slice, image_scale=coord_scale)
        bounds_from, bounds_to = get_coordinate_bounds(bounds_slice, math.ceil(image_dimension * coord_scale))

        coord_slice = prepare_coordinate(coord_slice)
        coord_from, coord_to = get_coordinate_bounds(coord_slice, bounds_to)

        out_from, out_to = max(coord_from + bounds_from, bounds_from), min(coord_to + bounds_from, bounds_to)
        res_from = max(-coord_from, 0)
        res_to = max(coord_to + bounds_from - bounds_to, 0)

        return (out_from, out_to), (res_from, res_to)

    @staticmethod
    def get_coordinate_shape(coord, image_dimension, coord_scale=1., image_scale=1.):
        import math
        from tiamat.readers.processing import prepare_coordinate
        from tiamat.transformers.coordinates import resolve_coordinate_slice

        coord_slice = prepare_coordinate(coord, image_scale, coord_scale)
        coord_from, coord_to = resolve_coordinate_slice(coord_slice, math.ceil(image_scale * image_dimension))

        coord_shape = coord_to - coord_from

        return coord_shape

    def bounds_spatial_shape(self, spatial_shape):
        bounds = (self.bounds_y, self.bounds_x)
        if len(spatial_shape) > 2:
            bounds = (self.bounds_z, *bounds)
        shape = tuple(
            BoundingBoxTransformer.get_coordinate_shape(bounds[i], spatial_shape[i])
            for i in range(len(spatial_shape))
        )
        return shape

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        from dataclasses import replace

        metadata = accessor.metadata
        assert metadata is not None, "BoundingBoxTransformer requires metadata."
        assert metadata.shape is not None, "BoundingBoxTransformer requires metadata.shape."

        accessor = replace(accessor)

        spatial_shape = self.bounds_spatial_shape(metadata.spatial_shape)

        accessor.x, residual_x = BoundingBoxTransformer.crop_coordinate(
            accessor.x,
            self.bounds_x,
            spatial_shape[-1],
            coord_scale=accessor.coordinate_scale,
        )
        accessor.y, residual_y = BoundingBoxTransformer.crop_coordinate(
            accessor.y,
            self.bounds_y,
            spatial_shape[-2],
            coord_scale=accessor.coordinate_scale,
        )
        residuals = [residual_x, residual_y]
        if len(spatial_shape) > 2:
            accessor.z, residual_z = BoundingBoxTransformer.crop_coordinate(
                accessor.z,
                self.bounds_z,
                spatial_shape[-3],
                coord_scale=accessor.coordinate_scale,
            )
            residuals.append(residual_z)

        accessor.history[id(self)] = residuals[::-1]

        return accessor


    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        from dataclasses import replace

        spatial_shape = list(self.bounds_spatial_shape(metadata.spatial_shape))

        # create the new shape by copying the new spatial shape, but keeping the channels
        shape = list(metadata.shape)
        for i, dimension in enumerate(metadata.spatial_dimensions): 
            shape[dimension] = spatial_shape[i]

        metadata = replace(metadata, shape=shape)

        return metadata


    def transform_image(self, image_result: ImageResult, accessor: ImageAccessor) -> ImageResult:
        import numpy as np

        residuals = accessor.history[id(self)]

        # Revert cropping from crop_coordinate with fill value padding
        if accessor.fill_value is not None:
            if any(any(p > 0 for p in pad) for pad in residuals):
                padding = [(0, 0) for _ in range(len(image_result.metadata.dimensions))]
                for i, dimension in enumerate(image_result.metadata.spatial_dimensions):
                    padding[dimension] = residuals[i]
                image_result.image = np.pad(image_result.image, padding, constant_values=accessor.fill_value)

        return image_result
