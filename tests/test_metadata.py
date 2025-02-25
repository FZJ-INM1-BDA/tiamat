import pytest
import numpy as np

from tiamat.metadata import ImageMetadata

@pytest.fixture
def metadata():
    yield ImageMetadata(image_type="image", shape=(1,2),
                        value_range=(0,1),
                        dtype=np.uint32)


def test_default_extents(metadata: ImageMetadata):
    assert metadata.extents == [(0, 0), (0, 2), (1, 0), (1,2)]


def test_assign_extents(metadata: ImageMetadata):
    metadata.extents = 'foo bar'
    assert metadata.extents == 'foo bar'

def test_del_extents(metadata: ImageMetadata):
    metadata.extents = 'foo bar'
    del metadata.extents
    assert metadata.extents == [(0, 0), (0, 2), (1, 0), (1,2)]

