"""
Processing required by readers.
"""

# Try to import OpenCV and use scikit-image as a fallback
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from tiamat.constants import (
    INTERPOLATION_TYPE_CUBIC,
    INTERPOLATION_TYPE_NEAREST,
    OPENCV_INTERPOLATION_CODES,
)
from tiamat.io import ImageAccessor
from tiamat.metadata.metadata import ImageMetadata


def _expand_to_dimension(value: int | float | Sequence[int | float], image_shape: Sequence[int]) -> list[float]:
    """
    Expand scalar or sequence to match image dimensions.

    Args:
        value: Scalar or sequence to expand.
        image_shape: Target image shape.

    Returns:
        Expanded list of floats.
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        return [value for _ in image_shape]
    elif len(value) == 1:
        return [value[0]] * len(image_shape)
    return value


def expand_to_length(value: Any, length: int) -> list[Any]:
    """
    Expand scalar or sequence to a list of given length.

    Args:
        value: Input scalar or sequence.
        length: Desired output length.

    Returns:
        List of length `length`.
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        return [value] * length
    elif len(value) == 1:
        return [value[0]] * length
    return value


def rescale_shape(shape: Sequence[int], scale: float | Sequence[float]) -> np.ndarray:
    """
    Compute the target shape of an array when scaled.

    Args:
        shape: Original shape of the array.
        scale: Scaling factor (scalar or sequence) for each dimension.

    Returns:
        Array of integers representing the scaled shape.
    """

    # Assume scale is scalar or (x_scale, y_scale)
    scale = _expand_to_dimension(scale, shape)[: len(shape)][::-1]
    assert all(s > 0 for s in scale), f"Scale must be greater than zero, got {scale}."

    target_shape = np.array([dim * s for dim, s in zip(shape, scale)], dtype=float)
    target_shape = np.maximum(np.round(target_shape).astype(int), 1)

    return target_shape


def _rescale(
    image: np.ndarray,
    scale: float | tuple[float, ...],
    interpolation: str = INTERPOLATION_TYPE_CUBIC,
    anti_aliasing: bool = False,
) -> np.ndarray:
    """
    Rescale a 2D or 3D image according to specified scale factors.

    Args:
        image: Input image as a numpy array.
        scale: Scaling factor(s) for each spatial dimension.
        interpolation: Interpolation method to use.
        anti_aliasing: Apply Gaussian smoothing before downscaling if True.

    Returns:
        Rescaled image as a numpy array.
    """

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
    img: np.ndarray, shape: Sequence[int], interpolation: str = INTERPOLATION_TYPE_CUBIC, anti_aliasing: bool = False
) -> np.ndarray:
    """
    Resize a 2D image to the specified shape using the given interpolation.

    Args:
        img: Image to resize (height x width or height x width x channels).
        shape: Target shape (height, width) or (height, width, channels).
        interpolation: Interpolation strategy to use.
        anti_aliasing: If True, apply Gaussian smoothing before downsizing.

    Returns:
        Resized 2D image as a numpy array.
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
    img: np.ndarray, shape: Sequence[int], interpolation: str = INTERPOLATION_TYPE_CUBIC, anti_aliasing: bool = False
) -> np.ndarray:
    """
    Resize a 3D image stack to the specified shape using the given interpolation.

    Args:
        img: 3D image stack to resize.
        shape: Target shape for the stack.
        interpolation: Interpolation strategy to use.
        anti_aliasing: Apply Gaussian smoothing before downsizing if True.

    Returns:
        Resized 3D image stack as a numpy array.

    Raises:
        NotImplementedError: Full 3D resizing not implemented if all dimensions need rescaling.
    """

    # Scaling factors per dimension
    factors = np.divide(img.shape, shape)

    # Rescale an image stack
    # if at least one factor is one, use resize_2d with loop
    if np.any(np.isclose(factors, 1.0)):
        # pick first dimension with factor 1
        idx = np.where(np.isclose(factors, 1.0))[0][0]

        img_rescaled = np.empty(shape, dtype=img.dtype)
        shape_2d = (*shape[:idx], *shape[idx + 1 :])

        for i in range(img.shape[idx]):
            slices = [slice(None)] * len(img.shape)
            slices[idx] = i
            img_rescaled[i] = resize_2d(
                img=img[tuple(slices)],
                shape=shape_2d,
                interpolation=interpolation,
                anti_aliasing=anti_aliasing,
            )
        return img_rescaled

    raise NotImplementedError("Full 3D resizing not supported yet")


def prepare_coordinate(
    coord: int | Sequence[int] | None, image_scale: float = 1.0, coordinate_scale: float = 1.0
) -> tuple[int, int]:
    """
    Prepare coordinates for image access by applying scaling and flooring.

    Args:
        coord: Single coordinate, sequence, or None for full dimension.
        image_scale: Scale factor of the image.
        coordinate_scale: Scale factor of the coordinates.

    Returns:
        Tuple of (start, end) indices corresponding to the prepared coordinate.
    """
    import math

    factor = image_scale / coordinate_scale

    # Note: We round to the 10 first significant digits here very slightly whenever we multiply the factor,
    # as minimal floating errors can mess the length up quite badly when applying math.ceil or math.floor
    def round_sig(x, sig=10):
        # we need a small eps to avoid infinity
        eps = 1 / (10 ** (sig + 2))
        return np.round(x, sig - int(np.floor(np.log10(abs(x + eps)))) - 1)

    if hasattr(coord, "__iter__"):
        # Assume the coordinate stores an half-open interval [start, stop)
        assert len(coord) == 2, "prepare_coordinate accepts only coordinate intervals [start, stop] of length 2"

        # Simply scale the start by the interval by the factor
        start = math.floor(round_sig(coord[0] * factor, 10))
        if coord[1] is None:
            # If the second coordinate is None, use None
            stop = None
        else:
            # Otherwise, estimate the length of the scaled interval
            length = math.ceil(round_sig((coord[1] - coord[0]) * factor, 10))
            # Compute the stop index by the length
            stop = start + length

        prepared_coord = (start, stop)
    elif coord is None:
        # All elements in the given dimension
        prepared_coord = (0, None)
    else:
        # A single element in the given dimension
        coord = math.floor(round_sig(coord * factor, 10))
        prepared_coord = (coord, coord + 1)

    return prepared_coord


def _prepare_coordinates(
    image_scale=(1.0, 1.0), coordinate_scale=(1.0, 1.0), **coordinates
) -> dict[str, tuple[int, int]]:
    """
    Prepare multiple coordinates for image access.

    Args:
        image_scale: Scale factors for each image dimension.
        coordinate_scale: Scale factors for each coordinate dimension.
        coordinates: Keyword arguments for coordinates (x, y, z, c, ...).

    Returns:
        Dictionary mapping coordinate names to (start, end) tuples.
    """
    prepared = {}
    for i, (key, coord) in enumerate(coordinates.items()):
        i_scale = image_scale[i] if i < len(image_scale) else 1.0
        c_scale = coordinate_scale[i] if i < len(coordinate_scale) else 1.0
        prepared[key] = prepare_coordinate(coord, i_scale, c_scale)

    return prepared


def _zero_clip(values: Sequence[int | None]) -> list[int]:
    """
    Clip values to be zero or positive, preserving None values.

    Args:
        values: Sequence of values to clip.

    Returns:
        List of clipped values.
    """
    return [max(value, 0) if value is not None else None for value in values]


def access_image(
    image: np.ndarray,
    metadata: ImageMetadata,
    accessor: ImageAccessor,
    image_scale: float | tuple[float, ...],
) -> np.ndarray:
    """
    Extract a subregion of an image, with padding if needed.

    Args:
        image: Input array (row-major).
        metadata: Metadata describing dimensions.
        accessor: Requested coordinates.
        image_scale: Scaling of each image dimension.

    Returns:
        The requested subregion as an array (with optional padding).
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

    # Loop over all dimensions to create request
    request_slices = [slice(None)] * len(image.shape)
    access_shape = np.ones((len(image.shape)), dtype=np.int64)
    for dim, dimension_name in enumerate(metadata.dimensions):
        max_c = image.shape[dim]
        coord = access_channels.get(dimension_name, (0, None))
        c_from, c_to = coord

        request_slices[dim] = slice(_clip(c_from, 0, max_c), _clip(c_to, 0, max_c))
        access_shape[dim] = c_to - c_from if c_to is not None else image.shape[dim]

    # Loop over image dimensions to determine padding
    padding = [(0, 0)] * len(image.shape)
    for dim in image_dims:
        coord = access_channels.get(metadata.dimensions[dim], (0, None))
        max_coord = image.shape[dim]
        coord_from, coord_to = coord

        if accessor.fill_value is not None and ((coord_to is not None and coord_to < 0) or (coord_from >= max_coord)):
            # The image will be empty, just return an empty array
            return np.full(access_shape, fill_value=accessor.fill_value, dtype=image.dtype)

        padding[dim] = (_pad(coord_from, max_coord), _pad(coord_to, max_coord))
        request_slices[dim] = slice(_clip(coord_from, 0, max_coord), _clip(coord_to, 0, max_coord))
        access_shape[dim] = coord_to - coord_from if coord_to is not None else image.shape[dim]

    # Read requested data
    result = image[tuple(request_slices)]

    # Only do padding if fill_value is set and necessary
    if accessor.fill_value is not None and any(any(p > 0 for p in pad) for pad in padding):
        result = np.pad(result, padding, constant_values=accessor.fill_value)

    return result


def access_and_rescale_image(
    image: np.ndarray,
    metadata: ImageMetadata,
    accessor: ImageAccessor,
    image_scale: float | tuple[float, ...] = 1.0,
):
    """
    Access an image region and rescale it according to accessor.

    Args:
        image: Input array.
        metadata: Metadata describing dimensions.
        accessor: Requested access details.
        image_scale: Scaling of each image dimension.

    Returns:
        The requested and rescaled image.
    """
    image_scale = _expand_to_dimension(image_scale, metadata.spatial_dimensions)
    image = access_image(image=image, metadata=metadata, accessor=accessor, image_scale=image_scale)

    scale = _expand_to_dimension(accessor.scale, metadata.spatial_dimensions)
    assert len(scale) == len(image_scale), f"Scale and image scale do not match: {len(scale)} vs. {len(image_scale)}"

    interpolation = get_interpolation_for_accessor(accessor, metadata)
    target_scale = tuple(s / s_image for s, s_image in zip(scale, image_scale))
    image = _rescale(
        image,
        scale=target_scale,
        interpolation=interpolation,
        anti_aliasing=accessor.anti_aliasing,
    )

    return image


def get_value_range_from_dtype(dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        value_range = (info.min, info.max)
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        value_range = (info.min, info.max)
    else:
        raise ValueError(f"Cannot determine value range for unsupported dtype: {dtype}")

    return value_range


def get_interpolation_for_accessor(accessor: ImageAccessor, metadata: ImageMetadata) -> int:
    """
    Determine interpolation type from accessor or metadata.

    Args:
        accessor: Image accessor with optional interpolation setting.
        metadata: Metadata including image type.

    Returns:
        Interpolation identifier string.
    """
    if accessor.interpolation:
        interpolation = accessor.interpolation
    elif metadata and metadata.image_type:
        interpolation = get_interpolation_for_image_type(image_type=metadata.image_type)
    else:
        raise RuntimeError(
            "Could not _rescale image, as neither 'interpolation' nor 'metadata.image_type' was provided."
        )
    return interpolation


def get_interpolation_for_image_type(image_type: str) -> str:
    """
    Get a default interpolation method for a given image type.

    Args:
        image_type: Type of the image (e.g., IMAGE, SEGMENTATION).

    Returns:
        Interpolation identifier string.

    Raises:
        AssertionError: If the image type is unknown.
    """
    from .. import metadata as md

    image_type_to_interpolation = {
        md.IMAGE_TYPE_IMAGE: INTERPOLATION_TYPE_CUBIC,
        md.IMAGE_TYPE_SEGMENTATION: INTERPOLATION_TYPE_NEAREST,
    }

    assert (
        image_type in image_type_to_interpolation
    ), f"Encountered unknown image type while determining interpolation: {image_type}"

    return image_type_to_interpolation[image_type]
