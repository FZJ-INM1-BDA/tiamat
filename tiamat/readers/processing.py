"""
Processing required by readers.
"""

# Try to import OpenCV and use scikit-image as a fallback
import warnings

import numpy as np

from tiamat.metadata.metadata import ImageMetadata

from ..io import (
    ImageAccessor,
    INTERPOLATION_TYPE_NEAREST,
    INTERPOLATION_TYPE_LINEAR,
    INTERPOLATION_TYPE_CUBIC,
    INTERPOLATION_TYPE_AREA,
    INTERPOLATION_TYPE_LANCZOS4,
)

SCIPY_INTERPOLATION_CODES = {
    INTERPOLATION_TYPE_NEAREST: 0,
    INTERPOLATION_TYPE_LINEAR: 1,
    INTERPOLATION_TYPE_CUBIC: 3,
}

try:
    # noinspection PyUnresolvedReferences
    import cv2

    CV2_AVAILABLE = True

    OPENCV_INTERPOLATION_CODES = {
        INTERPOLATION_TYPE_NEAREST: cv2.INTER_NEAREST,
        INTERPOLATION_TYPE_LINEAR: cv2.INTER_LINEAR,
        INTERPOLATION_TYPE_CUBIC: cv2.INTER_CUBIC,
        INTERPOLATION_TYPE_AREA: cv2.INTER_AREA,
        INTERPOLATION_TYPE_LANCZOS4: cv2.INTER_LANCZOS4,
    }
except ImportError:
    warnings.warn(
        "image: Module cv2 is not available, using scikit-image as a fallback"
    )
    CV2_AVAILABLE = False


def _expand_to_dimension(value, image_shape):
    if not isinstance(value, (list, tuple, np.ndarray)):
        return [value for _ in image_shape]
    elif len(value) == 1:
        return [value[0]] * len(image_shape)
    return value


def expand_to_length(value, length):
    if not isinstance(value, (list, tuple, np.ndarray)):
        return [value] * length
    elif len(value) == 1:
        return [value[0]] * length
    return value


def rescale_shape(shape, scale):
    # Assume scale is scalar or (x_scale, y_scale)
    scale = _expand_to_dimension(scale, shape)[:len(shape)][::-1]
    assert all(s > 0 for s in scale), f"Scale must be greater than zero, got {scale}."

    # Round to closest integer to avoid floating precision errors
    target_shape =  np.array([dim * s for dim, s in zip(shape, scale)], dtype=float)
    target_shape = np.maximum(np.round(target_shape).astype(int), 1)

    return target_shape


def _rescale(
    image: np.ndarray,
    scale: float | tuple[float, ...],
    interpolation: str = INTERPOLATION_TYPE_CUBIC,
    anti_aliasing: bool = False,
) -> np.ndarray:
    import numpy as np

    # check valid size of image
    if min(image.shape) == 0:
        warnings.warn("Not possible to resize image of shape {}".format(image.shape))
        return image

    # For simplicity assume spatial information in first ndims dimensions for now
    ndims = len(scale)
    assert ndims == 2 or ndims == 3

    target_shape = rescale_shape(
        image.shape[:ndims],
        scale,
    )

    if np.allclose(target_shape, image.shape[:ndims]):
        # Nothing to do
        return image

    if ndims == 2:
        return resize_2d(
            img=image,
            shape=target_shape,
            interpolation=interpolation,
            anti_aliasing=anti_aliasing,
        )
    if ndims == 3:
        return resize_3d(
            img=image,
            shape=target_shape,
            interpolation=interpolation,
            anti_aliasing=anti_aliasing,
        )


def resize_2d(
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

    # Scaling factors per dimension
    factors = np.divide(img.shape[:2], shape)

    # take care of rgb images and 2dim shapes
    if len(img.shape) == 3 and len(shape) == 2:
        shape = (shape[0], shape[1], img.shape[2])

    if anti_aliasing and np.any(factors > 1):
        sigma = np.maximum(0, (factors - 1) / 2)
        ksize = np.ceil(4.0 * sigma, dtype=int, casting="unsafe")
        ksize = ksize + (1 - ksize % 2)

        arr = cv2.GaussianBlur(img, ksize[::-1], sigmaX=sigma[1], sigmaY=sigma[0])
    else:
        arr = img

    res = cv2.resize(
        src=arr,
        dsize=(shape[1], shape[0]),
        interpolation=OPENCV_INTERPOLATION_CODES[interpolation],
    )

    # Resize tends to loose dimensions with size one, so we need to add them back
    if len(res.shape) < len(arr.shape) and arr.shape[-1] == 1:
        res = res.reshape(*res.shape, 1)

    return res.astype(img.dtype)


def resize_3d(
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

    # Scaling factors per dimension
    factors = np.divide(img.shape, shape)

    # Rescale an image stack
    # if at least one factor is one, use resize_2d with loop
    if np.any(np.isclose(factors, 1.0)):
        # pick first dimension with factor 1
        idx = np.where(np.isclose(factors, 1.0))[0][0]

        # create new shape
        img_rescaled = np.empty(shape, dtype=img.dtype)
        shape_2d = (*shape[:idx], *shape[idx + 1 :])

        for i in range(img.shape[idx]):
            # create slice
            slices = [slice(None)] * len(img.shape)
            slices[idx] = i

            # resize image
            img_rescaled[i] = resize_2d(
                img=img[tuple(slices)],
                shape=shape_2d,
                interpolation=interpolation,
                anti_aliasing=anti_aliasing,
            )
        return img_rescaled

    raise NotImplementedError("Full 3D resizing not supported yet")


def prepare_coordinate(coord, image_scale=1.0, coordinate_scale=1.0):
    import math

    factor = image_scale / coordinate_scale

    # Add 0.5 to coordinates for rounding integer coordinates in a pixel center aligned grid
    if hasattr(coord, '__iter__'):
        # tuple of values
        prepared_coord = tuple(
            math.floor((0.5 + c) * factor) if c is not None else None
            for c in coord
        )
    elif coord is None:
        # All elements in the given dimension
        prepared_coord = (0, None)
    else:
        # A single element in the given dimension
        coord = math.floor((0.5 + coord) * factor)
        prepared_coord = (coord, coord + 1)

    return prepared_coord


def _prepare_coordinates(
    image_scale=(1.0, 1.0), coordinate_scale=(1.0, 1.0), **coordinates
):
    prepared = {}
    for i, (key,coord) in enumerate(coordinates.items()):
        i_scale = image_scale[i] if i < len(image_scale) else 1.0
        c_scale = coordinate_scale[i] if i < len(coordinate_scale) else 1.0
        prepared[key] = prepare_coordinate(coord, i_scale, c_scale)

    return prepared


def _zero_clip(values):
    return [max(value, 0) if value is not None else None for value in values]


def access_image(
    image: np.ndarray,
    metadata: ImageMetadata,
    accessor: ImageAccessor,
    image_scale: float | tuple[float, ...],
) -> np.ndarray:
    """Access image content.

    Assume a row-major coordinate system of image (z, y, x).

    Parameters
    ----------
    image : np.ndarray
        Row-major image content as numpy array
    accessor : ImageAccessor
        Requested image coordinates
    image_scale : float | tuple[float, ...]
        Scaling of each image dimension
    default_value: float or int
        Fill value for out of bounds request

    Returns
    -------
    np.ndarray
        Requested image content
    """

    coordinate_scale = _expand_to_dimension(accessor.coordinate_scale, metadata.spatial_shape)
    image_scale = _expand_to_dimension(image_scale, metadata.spatial_shape)

    access_channels = {
        "x": accessor.x,
        "y": accessor.y,
        "z": accessor.z,
    }
    if isinstance(accessor.c, dict):
        access_channels.update(accessor.c)
    else:
        access_channels["c"] = accessor.c

    access_channels = _prepare_coordinates(
        image_scale=image_scale,
        coordinate_scale=coordinate_scale,
        **access_channels,
    )
    z, y, x = [access_channels[key] for key in ("z", "y", "x")]

    def _pad_left(coordinate):
        if coordinate is None:
            return 0
        return max(-int(coordinate), 0)

    def _pad_right(coordinate, coordinate_max):
        if coordinate is None:
            return 0
        return max(int(coordinate) - int(coordinate_max), 0)

    def _pad(coordinate, coordinate_max):
        return max(_pad_left(coordinate), _pad_right(coordinate, coordinate_max))

    def _clip(coordinate, min_coordinate, max_coordinate):
        if coordinate is None:
            return coordinate
        return min(max(int(coordinate), int(min_coordinate)), int(max_coordinate))

    # Separate image and channel dimensions
    ch_dims = list(metadata.channel_dimensions)
    ch_dims = ch_dims if ch_dims is not None else []

    image_dims = list(metadata.spatial_dimensions)
    n_image_dims = len(image_dims)

    assert n_image_dims == 2 or n_image_dims == 3, "Only 2D or 3D images supported"

    # Filter requested coordinates for each dim, assuming row major order
    access_image_dims = [z, y, x][-n_image_dims:]
    access_ch_dims = [value for key, value in access_channels.items() if key not in ("z", "y", "x")]

    # Loop over all dimensions to create request
    request_slices = [slice(None)] * len(image.shape)
    access_shape = np.ones((len(image.shape)), dtype=np.int64)
    for dim, coord in zip(ch_dims[:len(access_ch_dims)] + image_dims, access_ch_dims + access_image_dims):
        max_c = image.shape[dim]
        c_from, c_to = coord

        request_slices[dim] = slice(_clip(c_from, 0, max_c), _clip(c_to, 0, max_c))
        access_shape[dim] = c_to - c_from if c_to is not None else image.shape[dim]

    # Loop over image dimensions to determine padding
    padding = [(0, 0)] * len(image.shape)
    for dim, coord in zip(image_dims, access_image_dims):
        max_coord = image.shape[dim]
        coord_from, coord_to = coord

        if (
            accessor.fill_value is not None
            and (coord_to is not None and coord_to < 0)
            or coord_from >= max_coord
        ):
            # The image will be empty, just return an empty array
            return np.full(
                access_shape, fill_value=accessor.fill_value, dtype=image.dtype
            )

        padding[dim] = (_pad(coord_from, max_coord), _pad(coord_to, max_coord))
        request_slices[dim] = slice(
            _clip(coord_from, 0, max_coord), _clip(coord_to, 0, max_coord)
        )
        access_shape[dim] = (
            coord_to - coord_from if coord_to is not None else image.shape[dim]
        )

    # Read requested data
    result = image[tuple(request_slices)]

    # Only do padding if fill_value is set and necessary
    if accessor.fill_value is not None and any(
        any(p > 0 for p in pad) for pad in padding
    ):
        result = np.pad(result, padding, constant_values=accessor.fill_value)

    return result


def access_and_rescale_image(
    image: np.ndarray,
    metadata: ImageMetadata,
    accessor: ImageAccessor,
    image_scale: float | tuple[float, ...] = 1.0,
):
    image_scale = _expand_to_dimension(image_scale, metadata.spatial_dimensions)
    image = access_image(image=image, metadata=metadata, accessor=accessor, image_scale=image_scale)

    scale = _expand_to_dimension(accessor.scale, metadata.spatial_dimensions)
    assert len(scale) == len(
        image_scale
    ), f"Scale and image scale do not match: {len(scale)} vs. {len(image_scale)}"

    interpolation = get_interpolation_for_accessor(accessor, metadata)
    target_scale = tuple(s / s_image for s, s_image in zip(scale, image_scale))
    image = _rescale(
        image,
        scale=target_scale,
        interpolation=interpolation,
        anti_aliasing=accessor.anti_aliasing,
    )

    return image


def get_interpolation_for_accessor(accessor: ImageAccessor, metadata: ImageMetadata):
    if accessor.interpolation:
        interpolation = accessor.interpolation
    elif metadata and metadata.image_type:
        interpolation = get_interpolation_for_image_type(
            image_type=metadata.image_type
        )
    else:
        raise RuntimeError(
            f"Could not _rescale image, as neither 'interpolation' nor 'metadata.image_type' was provided."
        )
    return interpolation


def get_interpolation_for_image_type(image_type: str) -> int:
    from .. import metadata as md

    image_type_to_interpolation = {
        md.IMAGE_TYPE_IMAGE: INTERPOLATION_TYPE_CUBIC,
        md.IMAGE_TYPE_SEGMENTATION: INTERPOLATION_TYPE_NEAREST,
    }

    assert (
        image_type in image_type_to_interpolation
    ), f"Encountered unknown image type while determining interpolation: {image_type}"

    return image_type_to_interpolation[image_type]
