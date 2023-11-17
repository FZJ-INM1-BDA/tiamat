"""
Helper functions.
"""


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
