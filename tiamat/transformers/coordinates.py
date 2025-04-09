import numpy as np


def resolve_coordinate_slice(
        coordinate_slice: tuple[int | float | None, int | float | None] | int,
        image_dimension: int,
    ) -> tuple[int | float, int | float] | int | float:
    """Resolves None values in coordinate slices.

    Parameters
    ----------
    coordinate : tuple[int  |  float  |  None, int  |  float  |  None] | int
        _description_
    image_dimension : int
        Length of the image in the corresponding image dimension

    Returns
    -------
    tuple(int, int) or int
        Resolved coordinate slice interval containing no None values

    Raises
    ------
    RuntimeError
        _description_
    """
    if coordinate_slice is None:
        return (0, image_dimension)
    elif isinstance(coordinate_slice, (np.integer, int)) or isinstance(coordinate_slice, (np.floating, float)):
        return coordinate_slice
    else:
        slice_0, slice_1 = coordinate_slice
        if slice_0 is None:
            slice_0 = 0
        if slice_1 is None:
            slice_1 = image_dimension
        return (slice_0, slice_1)
