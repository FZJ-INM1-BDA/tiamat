"""
Affine transformers.
"""
from .protocol import Transformer
from ..io import ImageResult, ImageAccessor
from ..metadata import ImageMetadata


class AffineTransformer(Transformer):
    def __init__(self, affine_matrix):
        self.affine_matrix = affine_matrix

    def _resolve_coordinate(self, coordinate, image_dimension):
        import numpy as np

        if coordinate is None:
            return image_dimension
        elif isinstance(coordinate, (np.integer, int)):
            return coordinate
        elif isinstance(coordinate, (np.floating, float)):
            raise RuntimeError(f"AffineTransformer encountered fractional value {coordinate}, which is not supported at the moment.")
        else:
            return tuple(self._resolve_coordinate(coordinate_i, image_dimension) for coordinate_i in coordinate)

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        import numpy as np
        from dataclasses import replace

        assert accessor.metadata is not None, f"AffineTransformer requires metadata."
        # TODO: Take care of spacing and/or scale.
        # TODO. Handle 3D.

        # invert affine to find which coordinates we need to read
        affine = np.linalg.inv(self.affine_matrix)
        (x_from, y_from), (x_to, y_to) = self._warp_coordinates(accessor=accessor, affine=affine)

        accessor = replace(accessor)
        accessor.x = (x_from, x_to)
        accessor.y = (y_from, y_to)

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        return metadata

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        import cv2
        import numpy as np
        from ..readers.processing import get_interpolation_for_accessor, OPENCV_INTERPOLATION_CODES, _prepare_coordinates

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
        (x_from, y_from), (x_to, y_to) = self._warp_coordinates(accessor=accessor, affine=self.affine_matrix)
        target_size = (x_to - x_from, y_to - y_from)
        target_origin_affine = np.eye(3)
        target_origin_affine[:2, -1] = (-x_from, -y_from)

        # Step 1., 2., and 3.
        affine = target_origin_affine @ self.affine_matrix @ input_origin_affine
        interpolation = get_interpolation_for_accessor(accessor=image_result.accessor)
        image_result.image = cv2.warpAffine(src=image_result.image, M=affine[:2], dsize=target_size, flags=OPENCV_INTERPOLATION_CODES[interpolation])

        return image_result

    def _warp_coordinates(self, accessor, affine):
        import numpy as np
        from ..readers.processing import _prepare_coordinates

        x, y, *_ = _prepare_coordinates(x=accessor.x, y=accessor.y, z=accessor.z, c=accessor.c)
        x_from, x_to = self._resolve_coordinate(x, accessor.metadata.shape[1])
        y_from, y_to = self._resolve_coordinate(y, accessor.metadata.shape[0])

        # Transform points
        x_from_t, y_from_t = np.ceil(affine[:2, :2] @ (x_from, y_from) + affine[:2, -1]).astype(int)
        x_to_t, y_to_t = np.ceil(affine[:2, :2] @ (x_to, y_to) + affine[:2, -1]).astype(int)

        return (x_from_t, y_from_t), (x_to_t, y_to_t)
