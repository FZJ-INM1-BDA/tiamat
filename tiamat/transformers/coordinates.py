import numpy as np


def resolve_coordinate_slice(
    coordinate_slice: tuple[int | float | None, int | float | None] | int | float | None,
    image_dimension: int | float,
) -> tuple[int | float, int | float] | int | float:
    """
    Resolve `None` values in a coordinate slice.

    Args:
        coordinate_slice: Either a single coordinate (int/float) or a tuple of (start, stop),
            where values may be None.
        image_dimension: Length of the image along the corresponding axis.

    Returns:
        Either a single coordinate or a tuple of (start, stop), with all None replaced
        by valid numeric values.
    """
    if coordinate_slice is None:
        return 0, image_dimension
    elif isinstance(coordinate_slice, (np.integer, int)) or isinstance(coordinate_slice, (np.floating, float)):
        return coordinate_slice
    else:
        slice_0, slice_1 = coordinate_slice
        if slice_0 is None:
            slice_0 = 0
        if slice_1 is None:
            slice_1 = image_dimension
        return slice_0, slice_1


def get_coordinate_bounds(
    coordinate_slice: tuple[int | float | None, int | float | None] | int,
    image_dimension: int | float,
) -> tuple[int | float, int | float]:
    """
    Return numeric start and end values from a coordinate slice.

    Args:
        coordinate_slice: Slice or single coordinate to resolve.
        image_dimension: Length of the image along the corresponding axis.

    Returns:
        Tuple of (start, end) coordinates.
    """
    coord_fromto = resolve_coordinate_slice(coordinate_slice, image_dimension)

    if isinstance(coord_fromto, tuple):
        coord_from, coord_to = coord_fromto
    else:
        coord_from = coord_to = coord_fromto

    return coord_from, coord_to
