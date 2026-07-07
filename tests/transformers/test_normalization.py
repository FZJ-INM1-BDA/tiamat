import numpy as np
import pytest

from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.transformers.normalization import MinMaxNormalizationTransformer


def test_minmax_normalization_transform_access_passthrough():
    metadata = ImageMetadata("image", (2, 3), (0, 255), np.uint8)
    accessor = ImageAccessor(x=(0, 3), y=(0, 2), scale=0.5)
    transformer = MinMaxNormalizationTransformer()

    assert transformer.transform_access(accessor, metadata) is accessor


def test_minmax_normalization_transform_metadata_sets_value_range():
    metadata = ImageMetadata("image", (2, 3), (10, 110), np.uint8)
    metadata_dict = metadata.__dict__.copy()
    transformer = MinMaxNormalizationTransformer()

    transformed_metadata = transformer.transform_metadata(metadata)

    assert transformed_metadata is not metadata
    assert transformed_metadata.value_range == (0.0, 1.0)
    assert transformed_metadata.dtype == np.float32

    # Test that the original metadata remains unchanged
    assert metadata.value_range == (10, 110)
    assert metadata.dtype == np.uint8
    assert metadata.__dict__ == metadata_dict


def test_minmax_normalization_transform_metadata_sets_dtype():
    metadata = ImageMetadata("image", (2, 3), (10, 110), np.uint8)
    transformer = MinMaxNormalizationTransformer(target_dtype=np.float64)

    transformed_metadata = transformer.transform_metadata(metadata)

    assert transformed_metadata.dtype == np.float64
    assert transformed_metadata.value_range == (0.0, 1.0)


@pytest.mark.parametrize("target_dtype", [np.float32, np.float64])
def test_minmax_normalization_transform_image_scales_values(target_dtype):
    image = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, (0, 255), image.dtype)
    transformer = MinMaxNormalizationTransformer(target_dtype=target_dtype)

    result = transformer.transform_image(image, metadata, ImageAccessor())
    expected = image.astype(target_dtype) / 255.0

    np.testing.assert_allclose(result, expected)
    assert result.dtype == target_dtype
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_minmax_normalization_transform_image_requires_value_range():
    image = np.array([[0, 127, 255]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, None, image.dtype)
    transformer = MinMaxNormalizationTransformer()

    with pytest.raises(AssertionError, match="requires metadata.value_range"):
        transformer.transform_image(image, metadata, ImageAccessor())
