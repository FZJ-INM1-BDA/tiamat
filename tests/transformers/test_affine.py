import numpy as np
import pytest

from tiamat.io import ImageAccessor, ImageResult
from tiamat.metadata import IMAGE_TYPE_IMAGE, ImageMetadata
from tiamat.transformers.affine import AffineTransformer


@pytest.fixture
def sample_metadata():
    yield ImageMetadata(
        image_type=IMAGE_TYPE_IMAGE,
        value_range=(0, 255),
        shape=(1, 2),
        dtype="uint8",
    )


@pytest.fixture
def id_affine_transformer():
    yield AffineTransformer(np.eye(3))


@pytest.fixture
def transl_affine_xformer():
    m = np.array(
        [
            [1, 0, 30],
            [0, 1, 20],
            [0, 0, 1],
        ]
    )
    yield AffineTransformer(m)


@pytest.fixture
def rotate_affine_xformer():
    m = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    yield AffineTransformer(m)


def test_translate_metadata(sample_metadata: ImageMetadata, transl_affine_xformer: AffineTransformer):
    new_metadata = transl_affine_xformer.transform_metadata(sample_metadata)
    assert new_metadata.shape == (1, 2)


def test_rotation_transformer(sample_metadata: ImageMetadata, rotate_affine_xformer: AffineTransformer):
    new_metadata = rotate_affine_xformer.transform_metadata(sample_metadata)
    assert new_metadata.shape == (2, 1)


@pytest.fixture
def src_img_accessor():
    yield ImageAccessor()


four_by_four = [
    [1, 1, 2, 2],
    [1, 1, 2, 2],
    [3, 3, 4, 4],
    [3, 3, 4, 4],
]

two_by_two = [
    [1, 2],
    [3, 4],
]

scale_half = [
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 1],
]

identity = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]

scale_double = [
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 1],
]

translate = [
    [1, 0, 5],
    [0, 1, 10],
    [0, 0, 1],
]

affine_xform_img_args = [
    (four_by_four, scale_half, two_by_two),
    (two_by_two, identity, two_by_two),
    (four_by_four, identity, four_by_four),
    # this does not work, apparently scaling up is not as easy as scaling down
    (two_by_two, scale_double, four_by_four),
    (
        two_by_two,
        [
            [-1, 0, 2],
            [0, 1, 0],
            [0, 0, 1],
        ],
        [
            [2, 1],
            [4, 3],
        ],
    ),
    (
        four_by_four,
        [[-1, 0, 4], [0, 1, 0], [0, 0, 1]],
        [
            [2, 2, 1, 1],
            [2, 2, 1, 1],
            [4, 4, 3, 3],
            [4, 4, 3, 3],
        ],
    ),
]


@pytest.mark.parametrize("src_img, affine, exp_img", affine_xform_img_args)
def test_affine_transform_image(src_img, affine, exp_img):

    img_nd = np.array(src_img)
    exp_nd = np.array(exp_img)

    print("Apply:\n", affine)

    # Backward path
    meta = ImageMetadata(
        image_type="image", shape=img_nd.shape, value_range=(np.min(img_nd), np.max(img_nd)), dtype=img_nd.dtype
    )
    output_accessor = ImageAccessor(x=(0, exp_nd.shape[1]), y=(0, exp_nd.shape[0]), interpolation="nearest")
    xform = AffineTransformer(
        np.array(affine),
        request_margin=0,
    )
    input_accessor = xform.transform_access(output_accessor, metadata=meta)

    print("Request:\n", output_accessor)
    print("Reader:\n", input_accessor)

    # Forward path
    result_nd = xform.transform_image(img_nd, metadata=meta, accessor=input_accessor)

    print("Result:\n", result_nd)
    print("Expected:\n", exp_nd)

    assert np.all(result_nd == exp_nd)


IMG_SIZE = 100

mirror = [
    [-1, 0, IMG_SIZE - 1],
    [0, 1, 0],
    [0, 0, 1],
]

affine_xform_xform_xs_args = [
    (((0, 2), (0, 2)), identity, ((0, 2), (0, 2))),
    (((5, 10), (15, 20)), identity, ((5, 10), (15, 20))),
    (((0, 4), (0, 3)), scale_double, ((0, 2), (0, 2))),
    (((5, 9), (15, 20)), scale_double, ((2, 5), (7, 10))),
    (((0, 2), (0, 2)), scale_half, ((0, 4), (0, 4))),
    (((5, 10), (15, 20)), scale_half, ((10, 20), (30, 40))),
    (((0, 2), (0, 2)), translate, ((-5, -3), (-10, -8))),
    (((5, 10), (15, 20)), translate, ((0, 5), (5, 10))),
    (((0, 2), (0, 2)), mirror, ((97, 99), (0, 2))),
    (((5, 10), (15, 20)), mirror, ((89, 94), (15, 20))),
]


@pytest.mark.parametrize("src_xy, affine, expected_xy", affine_xform_xform_xs_args)
def test_affine_transform_access(src_xy, affine, expected_xy):

    img_nd = np.zeros((IMG_SIZE, IMG_SIZE))
    ex_x, ex_y, *_ = expected_xy

    meta = ImageMetadata(
        image_type="image",
        shape=img_nd.shape,
        value_range=(np.min(img_nd), np.max(img_nd)),
        dtype=img_nd.dtype,
    )

    print("Source xy:", *src_xy)

    accessor = ImageAccessor(
        *src_xy,
    )
    xformer = AffineTransformer(
        np.array(affine),
        request_margin=0,
    )
    result = xformer.transform_access(
        accessor,
        metadata=meta,
    )

    print("Result xy:", result.x, result.y)
    print("Expected xy:", *expected_xy)

    assert result.x == ex_x
    assert result.y == ex_y
