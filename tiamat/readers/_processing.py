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

from ..io import ImageAccessor, INTERPOLATION_TYPE_NEAREST, INTERPOLATION_TYPE_LINEAR, INTERPOLATION_TYPE_CUBIC

OPENCV_INTERPOLATION_CODES = {INTERPOLATION_TYPE_NEAREST: cv2.INTER_NEAREST, INTERPOLATION_TYPE_LINEAR: cv2.INTER_LINEAR, INTERPOLATION_TYPE_CUBIC: cv2.INTER_CUBIC}


def rescale(img, scale, interpolation=INTERPOLATION_TYPE_CUBIC):
    import numpy as np
    assert scale > 0, f"Scale must be greater than zero, got {scale}."

    if np.isclose(scale, 1.0):
        # Nothing to do
        return img

    # Note: Rescale always rounds up. This is a design decision, that we might want to revisit.
    target_shape = np.ceil(np.array(img.shape[:2], dtype=float) * scale).astype(int)
    return resize(img=img, shape=target_shape, interpolation=interpolation)


def resize(img, shape, interpolation=INTERPOLATION_TYPE_CUBIC):
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
    # take care of rgb images and 2dim shapes
    if len(img.shape) == 3 and len(shape) == 2:
        shape = (shape[0], shape[1], img.shape[2])
    arr = img
    res = cv2.resize(src=arr, dsize=(shape[1], shape[0]), interpolation=OPENCV_INTERPOLATION_CODES[interpolation])

    # Resize tends to loose dimensions with size one, so we need to add them back
    if len(res.shape) < len(arr.shape) and arr.shape[-1] == 1:
        res = res.reshape(*res.shape, 1)

    return res.astype(img.dtype)


def _prepare_coordinates(x, y, z, c):
    prepared = []
    for coord in (x, y, z, c):
        prepated_coord = coord
        if isinstance(coord, int):
            # A single element in the given dimension
            prepated_coord = (coord, coord + 1)
        elif coord is None:
            # All elements in the given dimension
            prepated_coord = (0, None)
        prepared.append(prepated_coord)
    return prepared


def _zero_clip(values):
    return [max(value, 0) if value is not None else None for value in values]


def access_image(image: np.ndarray, accessor: ImageAccessor) -> np.ndarray:
    x, y, z, c = _prepare_coordinates(x=accessor.x,
                                      y=accessor.y,
                                      z=accessor.z,
                                      c=accessor.c)

    x, y, z, c = _prepare_coordinates(x, y, z, c)
    x_from, x_to = _zero_clip(x)
    y_from, y_to = _zero_clip(y)

    # mind the order of x and y!
    result = image[y_from:y_to, x_from:x_to, ]

    if z is not None:
        z_from, z_to = _zero_clip(z)
        result = result[..., z_from:z_to]

    if c is not None:
        c_from, c_to = _zero_clip(c)
        result = result[..., c_from:c_to]

    return result


def access_and_rescale_image(image: np.ndarray, accessor: ImageAccessor):
    image = access_image(image=image, accessor=accessor)

    interpolation = get_interpolation_for_accessor(accessor)
    image = rescale(image, scale=accessor.scale, interpolation=interpolation)

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
