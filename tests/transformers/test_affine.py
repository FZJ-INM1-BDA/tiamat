import pytest
import numpy as np

from tiamat.metadata import ImageMetadata, IMAGE_TYPE_IMAGE
from tiamat.transformers.affine import AffineTransformer

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
