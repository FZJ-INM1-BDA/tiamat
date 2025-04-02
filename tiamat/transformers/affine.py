"""
Affine transformers.
"""
from typing import Any, Dict, Tuple, List
from dataclasses import asdict
from itertools import repeat, product

from .protocol import Transformer
from ..io import ImageResult, ImageAccessor
from ..metadata import ImageMetadata

import numpy as np


class AffineTransformer(Transformer):
    def __init__(
            self,
            affine_matrix: np.array | List[List[float]],
            request_margin: int = 2,
        ):
        self.affine_matrix = np.array(affine_matrix)

        # Pixel margin around requested image to avoid resampling artifacts
        self.request_margin = request_margin

    def _make_corner_px_affine(self, affine):
        # Make input affine matrix corner pixel aligned by shifted half a pixel value and reverse
        input_offset = np.eye(3)
        input_offset[:2, -1] = (0.5, 0.5)

        target_offset = np.eye(3)
        target_offset[:2, -1] = (-0.5, -0.5)

        return target_offset @ affine @ input_offset

    def _transform_point(self, x: int | float, y: int | float, affine: np.ndarray) -> Tuple[int | float, int | float]:
        return affine[:2, :2] @ (x, y) + affine[:2, -1]

    def transform_access(self, accessor: ImageAccessor) -> ImageAccessor:
        import math
        from dataclasses import replace
        from tiamat.readers.processing import _prepare_coordinates
        from tiamat.transformers.coordinates import resolve_coordinate_slice

        assert accessor.metadata is not None, f"AffineTransformer requires metadata."

        # TODO: Handle 3D.

        # Invert affine to find which coordinates we need to read
        affine = np.linalg.inv(self.affine_matrix)

        # Read coordinates for requested frame
        x, y, *_ = _prepare_coordinates(x=accessor.x, y=accessor.y, z=accessor.z, c=accessor.c)
        # TODO: Account for spacing and coordinate scale
        x_from, x_to = resolve_coordinate_slice(x, accessor.metadata.shape[1])
        y_from, y_to = resolve_coordinate_slice(y, accessor.metadata.shape[0])

        # Transform all four corners of the requested frame by the affine to determine min, max coordinates
        x1, y1 = self._transform_point(x_from, y_from, affine)
        x2, y2 = self._transform_point(x_to, y_from, affine)
        x3, y3 = self._transform_point(x_to, y_to, affine)
        x4, y4 = self._transform_point(x_from, y_to, affine)

        # Scale the margin by self.request_margin to obtain physical extent
        scaled_margin = self.request_margin / accessor.scale

        # Request outer bounds of the transformed view
        x_from_t = min(x1, x2, x3, x4) - scaled_margin
        y_from_t = min(y1, y2, y3, y4) - scaled_margin
        x_to_t = max(x1, x2, x3, x4) + scaled_margin
        y_to_t = max(y1, y2, y3, y4) + scaled_margin

        # Calculate offsets of the requested frame due to integer rounding
        offset_x_input = math.floor(x_from_t) - x_from_t
        offset_y_input = math.floor(y_from_t) - y_from_t

        # Replace accessor with new requested input
        accessor = replace(accessor)
        # TODO: Reconsider (math.floor(x_from_t), math.ceil(x_to_t) + 1)
        accessor.x = (math.floor(x_from_t), math.ceil(x_to_t))
        accessor.y = (math.floor(y_from_t), math.ceil(y_to_t))
        accessor.fill_value = 0 if accessor.fill_value is None else accessor.fill_value

        # Up to here everything is physical coordinates, but in transform_image we need pixel coordinates
        # We need to scale the coordinates to obtain pixel coordinates
        accessor.history[id(self)] = (x_from, x_to, offset_x_input, y_from, y_to, offset_y_input)

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        import numpy as np
        from dataclasses import replace

        metadata_dict = asdict(metadata)
        shape_tuple = metadata_dict.pop("shape")

        # converts shape to extends
        # e.g. shape of 10, 20
        # extents = ((0, 10), (0, 20))
        extents = list(zip(repeat(0), shape_tuple[::-1]))
        extent_coords = list(product(*extents))
        
        # shape is ONLY affected by rotation component of 3x3 matrix
        transformed_coords = (self.affine_matrix[:2, :2] @ np.array(extent_coords).T).T
        
        xmax = np.max(transformed_coords[:,0])
        ymax = np.max(transformed_coords[:,1])
        
        new_metadata = replace(metadata, shape=(ymax, xmax))

        transformed_coords = (self.affine_matrix @ np.vstack((np.array(extent_coords).T, [1,1,1,1])))[:2, :].T
        new_metadata.extents = transformed_coords.tolist()

        return new_metadata

    def transform_image(self, image_result: ImageResult) -> ImageResult:
        import math
        import cv2
        import numpy as np
        from ..readers.processing import get_interpolation_for_accessor, OPENCV_INTERPOLATION_CODES, _prepare_coordinates

        target_scale = image_result.accessor.scale

        # Restore extent from requested frame
        try:
            x_from, x_to, offset_x_input, y_from, y_to, offset_y_input = image_result.accessor.history[id(self)]
        except KeyError:
            raise Exception("transform_access has to be called once before transform_image")
        target_size = (
            int(target_scale * (x_to - x_from)),
            int(target_scale * (y_to - y_from)),
        )

        accessor = image_result.accessor
        (x_from_input, x_to_input), (y_from_input, y_to_input), *_ = _prepare_coordinates(x=accessor.x, y=accessor.y, z=accessor.z, c=accessor.c)

        # We have to take into account that our input image is not the actual origin of the image.
        # Also, the target image we aim to compute is not at the origin.
        # Subtract and add 0.5 to make the transformation corner aligned (center is default in CV2)
        # To get the result we want, we do the following:
        # 1. Specify an affine matrix shifting towards the origin of the input image
        # 2. Apply our actual affine matrix.
        # 3. Specify an affine matrix shifting towards the origin of the target image.

        # Transform affine to pixel coordinates and make it corner pixel aligned
        px_affine = self.affine_matrix.copy()
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
        interpolation = get_interpolation_for_accessor(accessor=image_result.accessor)

        def _apply_affine(image):
            return cv2.warpAffine(
                src=image,
                M=affine[:2],
                dsize=target_size,
                flags=OPENCV_INTERPOLATION_CODES[interpolation],
                borderValue=accessor.fill_value,
            )

        # Apply to image or loop over stack of images if 3 spatial dims
        spatial_dimensions = accessor.metadata.spatial_dimensions
        if len(spatial_dimensions) > 2:
            result_imgs = []
            # Loop over first spatial dimension
            for i in range(image_result.image.shape[spatial_dimensions[0]]):
                result_imgs.append(_apply_affine(image_result.image[i]))
            result_image = np.stack(result_imgs, axis=0)
            image_result.image = result_image
        else:
            # CV2
            image_result.image = _apply_affine(image_result.image)

        return image_result

    @classmethod
    def from_json(cls, args: Dict[str, Any]):
        return cls(
            affine_matrix=np.array(args["affine_matrix"]),
            request_margin=args.get("request_margin", 2),
        )
