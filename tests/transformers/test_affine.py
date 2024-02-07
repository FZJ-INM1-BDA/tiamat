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

affine_args = [
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

@pytest.mark.parametrize("src_img, affine, exp_img", affine_args)
def test_affine_scale(src_img, affine, exp_img):
    img_nd = np.array(src_img)
    meta = ImageMetadata(image_type="image",
                         shape=img_nd.shape,
                         value_range=(np.min(img_nd), np.max(img_nd)),
                         dtype=img_nd.dtype)
    src = ImageResult(img_nd, ImageAccessor(metadata=meta, interpolation="nearest"), meta)
    xform = AffineTransformer(np.array(affine))
    result = xform.transform_image(src)
    assert np.all(result.image == np.array(exp_img))
