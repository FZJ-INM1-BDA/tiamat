"""
Affine transformers.
"""
<<<<<<< HEAD
from typing import Tuple
=======
from dataclasses import asdict
from itertools import repeat, product

>>>>>>> 20e4d6d (feat: add affine pipeline metadata transform)
from .protocol import Transformer
from ..io import ImageResult, ImageAccessor
from ..metadata import ImageMetadata

import numpy as np


class AffineTransformer(Transformer):
    def __init__(self, affine_matrix):
        self.affine_matrix = affine_matrix

    def _resolve_coordinate(self, coordinate, image_dimension):

        if coordinate is None:
            return image_dimension
        elif isinstance(coordinate, (np.integer, int)):
            return coordinate
        elif isinstance(coordinate, (np.floating, float)):
            raise RuntimeError(f"AffineTransformer encountered fractional value {coordinate}, which is not supported at the moment.")
        else:
            return tuple(self._resolve_coordinate(coordinate_i, image_dimension) for coordinate_i in coordinate)

    def _transform_point(self, x: int, y: int, affine: np.ndarray) -> Tuple[int, int]:
        return np.ceil(affine[:2, :2] @ (x, y) + affine[:2, -1]).astype(int)

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        from dataclasses import replace
        from tiamat.readers._processing import _prepare_coordinates

        assert accessor.metadata is not None, f"AffineTransformer requires metadata."

        #TODO: Take care of spacing and/or scale.
        #TODO: Handle 3D.

        # Invert affine to find which coordinates we need to read
        affine = np.linalg.inv(self.affine_matrix)

        # Read coordinates from access and cache them for transform path
        x, y, *_ = _prepare_coordinates(x=accessor.x, y=accessor.y, z=accessor.z, c=accessor.c)
        self.x_from, self.x_to = self._resolve_coordinate(x, accessor.metadata.shape[1])
        self.y_from, self.y_to = self._resolve_coordinate(y, accessor.metadata.shape[0])

        # Transform all four corners of the affine to determine min, max coordinates
        x1, y1 = self._transform_point(self.x_from, self.y_from, affine)
        x2, y2 = self._transform_point(self.x_to, self.y_from, affine)
        x3, y3 = self._transform_point(self.x_to, self.y_to, affine)
        x4, y4 = self._transform_point(self.x_from, self.y_to, affine)

        # Request outer bounds of the transformed view
        x_from_t = min(x1, x2, x3, x4)
        y_from_t = min(y1, y2, y3, y4)
        x_to_t = max(x1, x2, x3, x4)
        y_to_t = max(y1, y2, y3, y4)

        # Replace accessor with new request
        accessor = replace(accessor)
        accessor.x = (x_from_t, x_to_t)
        accessor.y = (y_from_t, y_to_t)

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        import numpy as np
        from dataclasses import replace

        metadata_dict = asdict(metadata)
        shape_tuple = metadata_dict.pop("shape")

        # converts shape to extends
        # e.g. shape of 10, 20
        # extents = ((0, 10), (0, 20))
        extents = list(zip(repeat(0), shape_tuple))
        extent_coords = list(product(*extents))
        
        transformed_coords = (self.affine_matrix @ np.vstack((np.array(extent_coords).T, [1,1,1,1])))[:2, :].T
        
        xmax = np.max(transformed_coords[:,0])
        ymax = np.max(transformed_coords[:,1])
        
        return replace(metadata, shape=(xmax, ymax))

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        import cv2
        import numpy as np
        from ..readers.processing import get_interpolation_for_accessor, OPENCV_INTERPOLATION_CODES, _prepare_coordinates

        # We need to find the maximum rectangle, oriented along the coordinate axes, that fits inside the parallelogram
        target_size = (self.x_to - self.x_from, self.y_to - self.y_from)

        accessor = image_result.accessor
        (x_from_input, x_to_input), (y_from_input, y_to_input), *_ = _prepare_coordinates(x=accessor.x, y=accessor.y, z=accessor.z, c=accessor.c)

        # We have to take into account that our input image is not the actual origin of the image.
        # Also, the target image we aim to compute is not at the origin.
        # To get the result we want, we do the following:
        # 1. Specify an affine matrix shifting towards the origin of the input image
        # 2. Apply our actual affine matrix.
        # 3. Specify an affine matrix shifting towards the origin of the target image.

        # Step 1: Shift towards input.
        input_origin_affine = np.eye(3)
        input_origin_affine[:2, -1] = (x_from_input, y_from_input)

        # Step 3: Shift towards target. Determine these coordinates by inverting the matrix we used on the way here.
        # Invert the affine transformation we applied in the forward pass to determine the image size
        target_origin_affine = np.eye(3)
        target_origin_affine[:2, -1] = (-self.x_from, -self.y_from)

        # Step 1., 2., and 3.
        affine = target_origin_affine @ self.affine_matrix @ input_origin_affine
        interpolation = get_interpolation_for_accessor(accessor=image_result.accessor)
        image_result.image = cv2.warpAffine(src=image_result.image, M=affine[:2], dsize=target_size, flags=OPENCV_INTERPOLATION_CODES[interpolation])

        return image_result
