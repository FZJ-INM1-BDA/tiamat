"""
Processing required by readers.
"""
# Try to import OpenCV and use scikit-image as a fallback
import warnings

import numpy as np

try:
    # noinspection PyUnresolvedReferences
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    warnings.warn("image: Module cv2 is not available, using scikit-image as a fallback")
    CV2_AVAILABLE = False

from ..io import ImageAccessor, INTERPOLATION_TYPE_NEAREST, INTERPOLATION_TYPE_LINEAR, INTERPOLATION_TYPE_CUBIC, INTERPOLATION_TYPE_AREA, INTERPOLATION_TYPE_LANCZOS4

OPENCV_INTERPOLATION_CODES = {
    INTERPOLATION_TYPE_NEAREST: cv2.INTER_NEAREST,
    INTERPOLATION_TYPE_LINEAR: cv2.INTER_LINEAR,
    INTERPOLATION_TYPE_CUBIC: cv2.INTER_CUBIC,
    INTERPOLATION_TYPE_AREA: cv2.INTER_AREA,
    INTERPOLATION_TYPE_LANCZOS4: cv2.INTER_LANCZOS4
}


def _expand_to_image_shape(value, image_shape):
    if not isinstance(value, (list, tuple, np.ndarray)):
        return [value for _ in image_shape]
    return value


def rescale(
        image: np.ndarray,
        scale: float | tuple[float, ...],
        interpolation: str = INTERPOLATION_TYPE_CUBIC,
        anti_aliasing: bool = False,
    ) -> np.ndarray:
    import numpy as np

    scale = _expand_to_image_shape(scale, image.shape[:2])
    assert all(s > 0 for s in scale), f"Scale must be greater than zero, got {scale}."

    if np.allclose(scale, 1.0):
        # Nothing to do
        return image

    # Note: Rescale always rounds up. This is a design decision, that we might want to revisit.
    target_shape = np.ceil(np.array([dim * s for dim, s in zip(image.shape[:2], scale)], dtype=float)).astype(int)
    return resize(img=image, shape=target_shape, interpolation=interpolation, anti_aliasing=anti_aliasing)


def resize(
        img,
        shape,
        interpolation=INTERPOLATION_TYPE_CUBIC,
        anti_aliasing: bool = False,
    ):
    """Resize the image to specified shape using the given interpolation.
    If anti-alias is defined, a gauss filter will smooth the image before downsizing.
    If interpolation is NEAREST, anti-aliasing is turned of.

    Args:
        img (array-like): image to resize.
        shape (tuple): shape of the resized image.
        interpolation (str): interpolation strategy.
    """
    import cv2

    # check valid size of image
    if min(img.shape) == 0:
        warnings.warn("Not possible to resize image of shape {}".format(img.shape))
        return img
    
    # Scaling factors per dimension
    factors = np.divide(img.shape[:2], shape)

    # take care of rgb images and 2dim shapes
    if len(img.shape) == 3 and len(shape) == 2:
        shape = (shape[0], shape[1], img.shape[2])

    if anti_aliasing and np.any(factors > 1):
        sigma = np.maximum(0, (factors - 1) / 2)
        ksize = np.ceil(4. * sigma, dtype=int, casting='unsafe')
        ksize = ksize + (1 - ksize % 2)

        print(anti_aliasing, ksize, sigma, factors)

        arr = cv2.GaussianBlur(img, ksize[::-1], sigmaX=sigma[1], sigmaY=sigma[0])
    else:
        arr = img
        
    res = cv2.resize(src=arr, dsize=(shape[1], shape[0]), interpolation=OPENCV_INTERPOLATION_CODES[interpolation])

    # Resize tends to loose dimensions with size one, so we need to add them back
    if len(res.shape) < len(arr.shape) and arr.shape[-1] == 1:
        res = res.reshape(*res.shape, 1)

    return res.astype(img.dtype)


def _prepare_coordinates(x, y, z, c, image_scale=(1.0, 1.0), coordinate_scale=(1.0, 1.0)):
    prepared = []
    for i, coord in enumerate((x, y, z, c)):
        c_scale = coordinate_scale[i] if i < len(image_scale) else 1.0
        i_scale = image_scale[i] if i < len(coordinate_scale) else 1.0
        factor = i_scale / c_scale

        if isinstance(coord, int):
            # A single element in the given dimension
            coord = np.ceil(coord * factor).astype(int)
            prepared_coord = (coord, coord + 1)
        elif coord is None:
            # All elements in the given dimension
            prepared_coord = (0, None)
        else:
            # tuple of values
            prepared_coord = tuple(np.ceil(c * factor).astype(int) if c is not None else None for c in coord)
        prepared.append(prepared_coord)
    return prepared


def _zero_clip(values):
    return [max(value, 0) if value is not None else None for value in values]


def access_image(image: np.ndarray, accessor: ImageAccessor, image_scale: tuple[float, ...]) -> np.ndarray:
    coordinate_scale = _expand_to_image_shape(accessor.coordinate_scale, image.shape)
    x, y, z, c = _prepare_coordinates(x=accessor.x,
                                      y=accessor.y,
                                      z=accessor.z,
                                      c=accessor.c,
                                      image_scale=image_scale,
                                      coordinate_scale=coordinate_scale)

    x, y, z, c = _prepare_coordinates(x, y, z, c)

    max_y, max_x = image.shape[:2]

    def _pad_left(coordinate):
        if coordinate is None:
            return 0
        return max(-coordinate, 0)

    def _pad_right(coordinate, coordinate_max):
        if coordinate is None:
            return 0
        return max(coordinate - coordinate_max, 0)

    def _pad(coordinate, coordinate_max):
        return max(_pad_left(coordinate), _pad_right(coordinate, coordinate_max))

    def _clip(coordinate, min_coordinate, max_coordinate):
        if coordinate is None:
            return coordinate
        return min(max(coordinate, min_coordinate), max_coordinate - 1)

    x_from, x_to = x
    y_from, y_to = y

    if x_to is not None and (x_to < 0 or x_from >= image.shape[1]) \
       or y_to is not None and (y_to < 0 or y_from >= image.shape[0]):
        # the image will be empty, just return an empty array
        shape = (y_to - y_from, x_to - x_from, *image.shape[2:])
        return np.zeros(shape=shape, dtype=image.dtype)

    pad_x_left, pad_x_right = _pad(x_from, max_x), _pad(x_to, max_x)
    pad_y_left, pad_y_right = _pad(y_from, max_y), _pad(y_to, max_y)
    padding = [(pad_y_left, pad_y_right), (pad_x_left, pad_x_right), ]

    # clip after padding
    x_from, x_to = [_clip(xi, 0, max_x) for xi in (x_from, x_to)]
    y_from, y_to = [_clip(yi, 0, max_y) for yi in (y_from, y_to)]

    # mind the order of x and y!
    result = image[y_from:y_to, x_from:x_to, ]

    if accessor.z is not None:
        z_from, z_to = z
        # Is it okay to assume that z is always the second dimension? Only works as long as nobody passes z coordinates for images for RGB images or something like that.
        max_z = image.shape[2]
        pad_z_left, pad_z_right = _pad(z_from, max_z), _pad(z_to, max_z)
        padding = [(pad_z_left, pad_z_right)] + padding
        z_from, z_to = [_clip(zi, 0, max_z) for zi in (z_from, z_to)]
        result = result[..., z_from:z_to]

    if accessor.c is not None:
        c_from, c_to = c
        result = result[..., c_from:c_to]
        padding.append((0, 0))

    # Only do padding if necessary.
    if any(any(p > 0 for p in pad) for pad in padding):
        # Additional dimensions without accessor.
        padding = padding + [(0, 0) for _ in range(len(result.shape) - len(padding))]
        result = np.pad(result, padding, constant_values=accessor.fill_value)

    return result


def access_and_rescale_image(image: np.ndarray, accessor: ImageAccessor, image_scale: float | tuple[float, ...] = 1.0):
    image_scale = _expand_to_image_shape(image_scale, image.shape)
    image = access_image(image=image, accessor=accessor, image_scale=image_scale)

    scale = _expand_to_image_shape(accessor.scale, image.shape)
    assert len(scale) == len(image_scale), f"Scale and image scale do not match: {len(scale)} vs. {len(image_scale)}"

    interpolation = get_interpolation_for_accessor(accessor)
    target_scale = tuple(s / s_image for s, s_image in zip(scale, image_scale))
    image = rescale(image, scale=target_scale, interpolation=interpolation, anti_aliasing=accessor.anti_aliasing)

    return image


def get_interpolation_for_accessor(accessor):
    if accessor.interpolation:
        interpolation = accessor.interpolation
    elif accessor.metadata and accessor.metadata.image_type:
        interpolation = get_interpolation_for_image_type(image_type=accessor.metadata.image_type)
    else:
        raise RuntimeError(f"Could not rescale image, as neither 'interpolation' nor 'metadata.image_type' was provided.")
    return interpolation


def get_interpolation_for_image_type(image_type: str) -> int:
    from .. import metadata as md

    image_type_to_interpolation = {
        md.IMAGE_TYPE_IMAGE: INTERPOLATION_TYPE_CUBIC,
        md.IMAGE_TYPE_SEGMENTATION: INTERPOLATION_TYPE_NEAREST,
    }

    assert image_type in image_type_to_interpolation, f"Encountered unknown image type while determining interpolation: {image_type}"

    return image_type_to_interpolation[image_type]
