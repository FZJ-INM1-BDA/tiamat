"""
Processing required by readers.
"""
# Try to import OpenCV and use scikit-image as a fallback
import warnings

try:
    # noinspection PyUnresolvedReferences
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    warnings.warn("image: Module cv2 is not available, using scikit-image as a fallback")
    CV2_AVAILABLE = False

# interpolation strategies for rescaling
INTERPOLATION_NEAREST = 0
INTERPOLATION_LINEAR = 1
INTERPOLATIO_CUBIC = 3


def rescale(img, scale, interpolation=INTERPOLATIO_CUBIC):
    import numpy as np
    assert scale > 0, f"Scale must be greater than zero, got {scale}."

    if np.isclose(scale, 1.0):
        # Nothing to do
        return img

    # Note: Rescale always rounds up. This is a design decision, that we might want to revisit.
    target_shape = np.ceil(np.array(img.shape[:2], dtype=float) * scale).astype(int)
    return resize(img=img, shape=target_shape, interpolation=interpolation)


def resize(img, shape, interpolation=INTERPOLATIO_CUBIC):
    """Resize the image to specified shape using the given interpolation.
    If anti-alias is defined, a gauss filter will smooth the image before downsizing.
    If interpolation is NEAREST, anti-aliasing is turned of.

    Args:
        img (array-like): image to resize.
        shape (tuple): shape of the resized image.
        interpolation (int): interpolation strategy.
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
    interpolation_codes = {INTERPOLATION_NEAREST: cv2.INTER_NEAREST, INTERPOLATION_LINEAR: cv2.INTER_LINEAR, INTERPOLATIO_CUBIC: cv2.INTER_CUBIC}
    res = cv2.resize(src=arr, dsize=(shape[1], shape[0]), interpolation=interpolation_codes[interpolation])

    # Resize tends to loose dimensions with size one, so we need to add them back
    if len(res.shape) < len(arr.shape) and arr.shape[-1] == 1:
        res = res.reshape(*res.shape, 1)

    return res.astype(img.dtype)


def _prepare_coordinates(x, y, z, c):
    prepared = []
    for coord in (x, y, z, c):
        prepated_coord = coord
        if isinstance(coord, int):
            prepated_coord = (coord, coord + 1)
        prepared.append(prepated_coord)
    return prepared


def access_image(image, x, y, z=None, c=None):
    x, y, z, c = _prepare_coordinates(x, y, z, c)
    x_from, x_to = x
    y_from, y_to = y

    # mind the order of x and y!
    result = image[y_from:y_to, x_from:x_to, ]

    if z is not None:
        z_from, z_to = z
        result = result[..., z_from:z_to]

    if c is not None:
        c_from, c_to = c
        result = result[..., c_from:c_to]

    return result
