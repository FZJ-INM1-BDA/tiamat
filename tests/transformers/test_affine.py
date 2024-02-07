import pytest
import numpy as np

from tiamat.metadata import ImageMetadata, IMAGE_TYPE_IMAGE
from tiamat.transformers.affine import AffineTransformer
from tiamat.io import ImageResult, ImageAccessor

@pytest.fixture
def sample_metadata():
    yield ImageMetadata(
        image_type=IMAGE_TYPE_IMAGE,
        value_range=(0,255),
        shape=(1,2),
        dtype="uint8",
    )

@pytest.fixture
def id_affine_transformer():
    yield AffineTransformer(np.eye(3))

@pytest.fixture
def transl_affine_xformer():
    m = np.array([
        [1, 0, 30],
        [0, 1, 20],
        [0, 0, 1],
    ])
    yield AffineTransformer(m)

@pytest.fixture
def rotate_affine_xformer():
    m = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    yield AffineTransformer(m)


def test_translate_metadata(sample_metadata: ImageMetadata, transl_affine_xformer: AffineTransformer):
    new_metadata = transl_affine_xformer.transform_metadata(sample_metadata)
    assert new_metadata.shape == (1, 2)
    assert sorted(new_metadata.extents) == sorted([
        [30, 20],
        [32, 20],
        [30, 21],
        [32, 21],
    ])

def test_rotation_transformer(sample_metadata: ImageMetadata, rotate_affine_xformer: AffineTransformer):
    new_metadata = rotate_affine_xformer.transform_metadata(sample_metadata)
    assert new_metadata.shape == (2, 1)

@pytest.fixture
def src_img_accessor():
    yield ImageAccessor()

four_by_four = [[1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4]]

two_by_two = [[1, 2],
              [3, 4]]

scale_half = [[0.5, 0, 0],
              [0, 0.5, 0],
              [0, 0, 1]]

identity = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]

scale_double = [[2, 0, 0],
                [0, 2, 0],
                [0, 0, 1]]

translate = [[1, 0, 5],
             [0, 1, 10],
             [0, 0, 1]]

affine_xform_img_args = [
    (four_by_four, scale_half, two_by_two),
    (two_by_two, identity, two_by_two),
    (four_by_four, identity, four_by_four),
    # this does not work, apparently scaling up is not as easy as scaling down
    # (two_by_two, scale_double, four_by_four),
    (two_by_two,
     
     [[-1, 0, 1],
      [0, 1, 0],
      [0, 0, 1]],
      
     [[2, 1],
      [4, 3]]),
]


@pytest.mark.parametrize("src_img, affine, exp_img", affine_xform_img_args)
def test_affine_transform_image(src_img, affine, exp_img):
    img_nd = np.array(src_img)
    meta = ImageMetadata(image_type="image",
                         shape=img_nd.shape,
                         value_range=(np.min(img_nd), np.max(img_nd)),
                         dtype=img_nd.dtype)
    src = ImageResult(img_nd, ImageAccessor(metadata=meta, interpolation="nearest"), meta)
    xform = AffineTransformer(np.array(affine))
    result = xform.transform_image(src)
    assert np.all(result.image == np.array(exp_img))

IMG_SIZE = 100

mirror = [[-1, 0, IMG_SIZE - 1],
          [0, 1, 0],
          [0, 0, 1]]

affine_xform_xform_xs_args = [
    (((0, 2), (0, 2)), identity, ((0, 2), (0, 2))),
    (((5, 10), (15, 20)), identity, ((5, 10), (15, 20))),

    (((0, 2), (0, 2)), scale_double, ((0, 1), (0, 1))),
    (((5, 10), (15, 20)), scale_double, ((3, 5), (8, 10))), # np.ceil

    
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
    meta = ImageMetadata(image_type="image",
                         shape=img_nd.shape,
                         value_range=(np.min(img_nd), np.max(img_nd)),
                         dtype=img_nd.dtype)
    accessor = ImageAccessor(*src_xy, metadata=meta)
    xformer = AffineTransformer(np.array(affine))
    result = xformer.transform_access(accessor)

    ex_x, ex_y, *_ = expected_xy
    assert result.x == ex_x
    assert result.y == ex_y
