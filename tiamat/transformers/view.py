"""
Transformers that change the output view
"""

import numpy as np

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import Transformer


class BoundingBoxTransformer(Transformer):
    """
    Crop images to a defined bounding box along x, y, and optionally z axes.

    The bounding box is applied to the accessor, metadata, and image.
    Missing regions are optionally padded with the accessor's `fill_value`.
    """

    def __init__(
        self,
        bounds_x: tuple[int | float | None, int | float | None] | int = None,
        bounds_y: tuple[int | float | None, int | float | None] | int = None,
        bounds_z: tuple[int | float | None, int | float | None] | int = None,
    ):
        """
        Initialize a bounding box transformer.

        Args:
            bounds_x: Bounding box along the x-axis (tuple or single int).
            bounds_y: Bounding box along the y-axis (tuple or single int).
            bounds_z: Bounding box along the z-axis (tuple or single int), if present.
        """
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.bounds_z = bounds_z

    @staticmethod
    def crop_coordinate(coord_slice, bounds_slice, image_dimension, coord_scale: float = 1.0):
        """
        Compute the cropped coordinate range and residual padding.

        Args:
            coord_slice: Original coordinate slice of the accessor.
            bounds_slice: Bounding box slice to crop into.
            image_dimension: Dimension length of the image.
            coord_scale: Scaling factor for coordinates.

        Returns:
            A tuple of:
                - (out_from, out_to): Cropped coordinate interval.
                - (res_from, res_to): Residual padding needed for reconstruction.
        """
        import math

        from tiamat.readers.processing import prepare_coordinate
        from tiamat.transformers.coordinates import get_coordinate_bounds

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
    def get_coordinate_shape(coord, image_dimension, coord_scale: float = 1.0, image_scale: float = 1.0) -> int:
        """
        Compute the length of a coordinate slice.

        Args:
            coord: Coordinate slice or index.
            image_dimension: Dimension length of the image.
            coord_scale: Scale of the coordinate system.
            image_scale: Scale of the image.

        Returns:
            Length of the coordinate slice.
        """
        import math

        from tiamat.readers.processing import prepare_coordinate
        from tiamat.transformers.coordinates import resolve_coordinate_slice

        coord_slice = prepare_coordinate(coord, image_scale, coord_scale)
        coord_from, coord_to = resolve_coordinate_slice(coord_slice, math.ceil(image_scale * image_dimension))

        coord_shape = coord_to - coord_from

        return coord_shape

    def bounds_spatial_shape(self, spatial_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Compute the new spatial shape after applying the bounding box.

        Args:
            spatial_shape: Original spatial shape of the image.

        Returns:
            Cropped spatial shape.
        """
        bounds = (self.bounds_y, self.bounds_x)
        if len(spatial_shape) > 2:
            bounds = (self.bounds_z, *bounds)
        shape = tuple(
            BoundingBoxTransformer.get_coordinate_shape(bounds[i], spatial_shape[i]) for i in range(len(spatial_shape))
        )
        return shape

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Apply bounding box cropping to the accessor.

        Args:
            accessor: The input accessor.
            metadata: The metadata.

        Returns:
            Updated accessor cropped to the bounding box.
        """
        from dataclasses import replace

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
        """
        Update metadata shape to reflect bounding box cropping.

        Args:
            metadata: The input metadata.

        Returns:
            Updated metadata with new shape.
        """
        from dataclasses import replace

        spatial_shape = list(self.bounds_spatial_shape(metadata.spatial_shape))

        # create the new shape by copying the new spatial shape, but keeping the channels
        shape = list(metadata.shape)
        for i, dimension in enumerate(metadata.spatial_dimensions):
            shape[dimension] = spatial_shape[i]
        shape = tuple(shape)

        metadata = replace(metadata, shape=shape)

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """Crop image and apply padding if necessary."""
        residuals = accessor.history[id(self)]

        # Revert cropping from crop_coordinate with fill value padding
        if accessor.fill_value is not None:
            if any(any(p > 0 for p in pad) for pad in residuals):
                padding = [(0, 0) for _ in range(len(metadata.dimensions))]
                for i, dimension in enumerate(metadata.spatial_dimensions):
                    padding[dimension] = residuals[i]
                image = np.pad(image, padding, constant_values=accessor.fill_value)

        return image
