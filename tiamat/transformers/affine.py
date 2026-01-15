"""
Affine transformers.
"""

import logging
from itertools import product, repeat
from typing import Any

import numpy as np

from tiamat.constants import OPENCV_INTERPOLATION_CODES

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import Transformer

logger = logging.getLogger(__name__)


class AffineTransformer(Transformer):
    """
    Transformer that applies affine transformations to images and metadata.

    Supports affine transformations in 2D using a (3, 3) or (2, 3) matrix.
    - `transform_access` modifies the accessor to request the correct input region.
    - `transform_metadata` updates metadata dimensions after the transform.
    - `transform_image` resamples the image using OpenCV warpAffine.
    """

    # TODO: define a unit of affine matrix, e.g. microns, mm, ...
    # TODO: allow to use center-pixel / corner-pixel aligned affine
    def __init__(
        self,
        affine_matrix: np.ndarray | list[list[float]],
        request_margin: int = 2,
        fill_value: int | float | None = None,
    ):
        """
        Initialize an AffineTransformer.

        Args:
            affine_matrix: 2x3 or 3x3 affine matrix (list or numpy array).
            request_margin: Pixel margin around requested image to avoid resampling artifacts.
            fill_value: Value used for filling empty regions after transformation.
        """

        self.affine_matrix = np.array(affine_matrix)

        # If affine is of shape (2, 3), extent to its (3, 3) form
        if self.affine_matrix.shape == (2, 3):
            self.affine_matrix = np.vstack((self.affine_matrix, [0.0, 0.0, 1.0]))

        # Pixel margin around requested image to avoid resampling artifacts
        self.request_margin = request_margin

        self.fill_value = fill_value

    def _make_corner_px_affine(self, affine: np.ndarray) -> np.ndarray:
        """
        Convert affine matrix to corner pixel aligned form.

        Args:
            affine: The affine matrix.

        Returns:
            Corner pixel aligned affine matrix.
        """
        input_offset = np.eye(3)
        input_offset[:2, -1] = (0.5, 0.5)

        target_offset = np.eye(3)
        target_offset[:2, -1] = (-0.5, -0.5)

        return target_offset @ affine @ input_offset

    def _transform_point(self, x: int | float, y: int | float, affine: np.ndarray) -> tuple[int | float, int | float]:
        """
        Transform a point using an affine matrix.

        Args:
            x: X coordinate.
            y: Y coordinate.
            affine: Affine matrix.

        Returns:
            Transformed (x, y) coordinates.
        """
        return affine[:2, :2] @ (x, y) + affine[:2, -1]

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Transform the accessor so that the correct input region is read before affine warping.

        Args:
            accessor: The ImageAccessor to transform.
            metadata: The metadata.

        Returns:
            A new ImageAccessor with adjusted coordinates and history.

        Raises:
            AssertionError: If metadata is missing.
        """
        import math
        from dataclasses import replace

        from tiamat.readers.processing import _prepare_coordinates
        from tiamat.transformers.coordinates import resolve_coordinate_slice

        target_spacing = metadata.spacing
        target_spacing = np.array(target_spacing) if target_spacing is not None else np.array((1.0, 1.0))
        if target_spacing.size == 1:
            target_spacing = np.array([target_spacing, target_spacing])
        target_spacing = target_spacing[:2]  # TODO: general solution for 3D

        # Invert affine to compute source coordinates
        affine = np.linalg.inv(self.affine_matrix)
        affine[:2, -1] = affine[:2, -1] / target_spacing  # Convert translation to pixel coordinates

        coordinate_scale = np.array(accessor.coordinate_scale)
        if coordinate_scale.size == 1:
            coordinate_scale = np.array([coordinate_scale, coordinate_scale])

        prepared_coordinates = _prepare_coordinates(x=accessor.x, y=accessor.y, coordinate_scale=coordinate_scale)
        x, y = prepared_coordinates["x"], prepared_coordinates["y"]

        spatial_dims = metadata.spatial_dimensions
        x_from, x_to = resolve_coordinate_slice(x, metadata.shape[spatial_dims[-1]])
        y_from, y_to = resolve_coordinate_slice(y, metadata.shape[spatial_dims[-2]])

        # Transform all four corners of the frame
        # Assume corner pixel coordinate system
        x1, y1 = self._transform_point(x_from, y_from, affine)
        x2, y2 = self._transform_point(x_to, y_from, affine)
        x3, y3 = self._transform_point(x_to, y_to, affine)
        x4, y4 = self._transform_point(x_from, y_to, affine)

        scale = np.array(accessor.scale)
        if scale.size == 1:
            scale = np.array([scale, scale])

        # Scale the margin by self.request_margin to obtain physical extent
        scaled_margin = self.request_margin / scale

        # Bounding box of transformed coordinates
        x_from_t = min(x1, x2, x3, x4) - scaled_margin[0]
        y_from_t = min(y1, y2, y3, y4) - scaled_margin[1]
        x_to_t = max(x1, x2, x3, x4) + scaled_margin[0]
        y_to_t = max(y1, y2, y3, y4) + scaled_margin[1]

        # Offset due to rounding
        offset_x_input = math.floor(x_from_t) - x_from_t
        offset_y_input = math.floor(y_from_t) - y_from_t

        # Replace accessor with new requested input
        accessor = replace(accessor)
        # TODO: Reconsider (math.floor(x_from_t), math.ceil(x_to_t) + 1)
        x_from_input, x_to_input = (math.floor(x_from_t), math.ceil(x_to_t))
        accessor.x = x_from_input, x_to_input
        y_from_input, y_to_input = (math.floor(y_from_t), math.ceil(y_to_t))
        accessor.y = y_from_input, y_to_input
        if self.fill_value is not None:
            accessor.fill_value = self.fill_value
        elif accessor.fill_value is None:
            accessor.fill_value = 0

        # Affine transformation is always applied at full coordinate scale
        accessor.coordinate_scale = 1.0

        # Up to here everything is physical coordinates, but in transform_image we need pixel coordinates
        # We need to scale the coordinates to obtain pixel coordinates
        accessor.history[id(self)] = (
            x_from_input,
            y_from_input,
            x_from,
            x_to,
            offset_x_input,
            y_from,
            y_to,
            offset_y_input,
        )

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Transform metadata spatial shape according to the affine transformation.

        Args:
            metadata: Original metadata.

        Returns:
            Updated ImageMetadata with transformed shape.
        """
        from dataclasses import replace

        shape_tuple = metadata.spatial_shape[-2:]

        # converts shape to extents
        # e.g. shape of 10, 20
        # extents = ((0, 10), (0, 20))
        extents = list(zip(repeat(0), shape_tuple[::-1]))
        extent_coords = list(product(*extents))

        metadata = replace(metadata)

        transformed_coords = (self.affine_matrix @ np.vstack((np.array(extent_coords).T, [1, 1, 1, 1])))[:2, :].T

        out_shape = (
            round(np.max(transformed_coords[:, 1]).item() - np.min(transformed_coords[:, 1]).item()),
            round(np.max(transformed_coords[:, 0]).item() - np.min(transformed_coords[:, 0]).item()),
        )

        metadata.spatial_shape = (*metadata.spatial_shape[:-2], *out_shape)

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Apply affine transformation to the image using OpenCV.

        Args:
            image: The Image.
            metadata: The metadata of the image.
            accessor: the accessor of the image.

        Returns:
            Transformed Image.

        Raises:
            Exception: If transform_access has not been called first.
        """
        import cv2

        from tiamat.readers.processing import (
            get_interpolation_for_accessor,
            rescale_shape,
        )

        target_scale = accessor.scale
        target_scale = np.array(target_scale) if target_scale is not None else np.array((1,))
        if target_scale.size == 1:
            target_scale = np.array([target_scale, target_scale])
        target_scale = target_scale[:2]  # TODO: general solution for 3D

        target_spacing = metadata.spacing
        target_spacing = np.array(target_spacing) if target_spacing is not None else np.array((1.0, 1.0))
        if target_spacing.size == 1:
            target_spacing = np.array([target_spacing, target_spacing])
        target_spacing = target_spacing[:2]  # TODO: general solution for 3D

        # Restore extent from requested frame
        try:
            x_from_input, y_from_input, x_from, x_to, offset_x_input, y_from, y_to, offset_y_input = accessor.history[
                id(self)
            ]
        except KeyError:
            raise Exception("transform_access has to be called once before transform_image")

        target_shape = rescale_shape(((y_to - y_from), (x_to - x_from)), target_scale)[::-1]

        # We have to take into account that our input image is not the actual origin of the image.
        # Also, the target image we aim to compute is not at the origin.
        # Subtract and add 0.5 to make the transformation corner aligned (center is default in CV2)
        # To get the result we want, we do the following:
        # 1. Specify an affine matrix shifting towards the origin of the input image
        # 2. Apply our actual affine matrix.
        # 3. Specify an affine matrix shifting towards the origin of the target image.

        px_affine = self.affine_matrix.copy()

        # scale translation to pixel coordinates
        px_affine[:2, -1] = px_affine[:2, -1] / target_spacing

        # Transform affine to pixel coordinates and make it corner pixel aligned
        px_affine[:2, -1] = px_affine[:2, -1] * target_scale
        px_affine = self._make_corner_px_affine(px_affine)

        # Step 1: Shift towards input.
        input_origin_affine = np.eye(3)
        input_origin_affine[:2, -1] = (x_from_input + offset_x_input, y_from_input + offset_y_input)
        input_origin_affine[:2, -1] = input_origin_affine[:2, -1] * target_scale

        # Step 3: Shift towards target.
        target_origin_affine = np.eye(3)
        target_origin_affine[:2, -1] = (-x_from, -y_from)
        target_origin_affine[:2, -1] = target_origin_affine[:2, -1] * target_scale
        # Step 1., 2., and 3.
        affine = target_origin_affine @ px_affine @ input_origin_affine
        interpolation = get_interpolation_for_accessor(accessor=accessor, metadata=metadata)

        if self.fill_value is None:
            if accessor.fill_value is None:
                fill_value = 0
            else:
                fill_value = accessor.fill_value
        else:
            fill_value = self.fill_value

        def _apply_affine(image):
            return cv2.warpAffine(
                src=image,
                M=affine[:2],
                dsize=target_shape,
                flags=OPENCV_INTERPOLATION_CODES[interpolation],
                borderValue=fill_value,
            )

        # Apply to image or loop over stack of images if 3 spatial dims
        spatial_dimensions = metadata.spatial_dimensions
        if len(spatial_dimensions) > 2:
            result_imgs = []
            # Loop over first spatial dimension
            for i in range(image.shape[spatial_dimensions[0]]):
                result_imgs.append(_apply_affine(image[i]))
            result_image = np.stack(result_imgs, axis=0)
            image = result_image
        else:
            image = _apply_affine(image)

        return image

    @classmethod
    def from_json(cls, args: dict[str, Any]) -> "AffineTransformer":
        """
        Create an AffineTransformer from a JSON-style dictionary.

        Args:
            args: Dictionary with keys:
                  - `affine_matrix`: 2x3 or 3x3 matrix
                  - optional `request_margin`: int
                  - optional `fill_value`: float

        Returns:
            AffineTransformer instance.
        """
        return cls(
            affine_matrix=np.array(args["affine_matrix"]),
            request_margin=args.get("request_margin", 2),
            fill_value=args.get("fill_value"),
        )
