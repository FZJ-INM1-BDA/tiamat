import numpy as np
import pytest

from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.transformers.intensity_mapping import (
    BrightnessContrastIntensityMappingTransformer,
    ClipIntensityMappingTransformer,
    GammaIntensityMappingTransformer,
    Log1pIntensityMappingTransformer,
    LogIntensityMappingTransformer,
    MappingTransformer,
)


@pytest.mark.parametrize(
    "target_dtype",
    [np.float32, np.float64],
)
def test_mapping_transformer_with_identity_lambda(target_dtype):
    """Base class behavior is tested once with an identity lambda."""

    image = np.array([[10, 20, 30]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, (0, 255), image.dtype)
    accessor = ImageAccessor(x=(0, 100), y=(0, 100), scale=0.5)

    transformer = MappingTransformer(
        mapping_function=lambda value: value,
        target_dtype=target_dtype,
    )

    assert transformer.transform_access(accessor, metadata) is accessor

    result = transformer.transform_image(image, metadata, accessor)
    transformed_metadata = transformer.transform_metadata(metadata)

    np.testing.assert_array_equal(result, image.astype(target_dtype))
    assert result.dtype == target_dtype
    assert transformed_metadata.dtype == target_dtype
    assert transformed_metadata.value_range == metadata.value_range


def test_mapping_transformer_with_lambda():
    """Test the generic mapping transformer with a caller-provided lambda."""

    image = np.array([[0, 127, 255]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, (0, 255), image.dtype)

    transformer = MappingTransformer(
        mapping_function=lambda value: value / 255.0,
        target_dtype=np.float64,
    )
    result = transformer.transform_image(image, metadata, ImageAccessor())
    transformed_metadata = transformer.transform_metadata(metadata)

    expected = image.astype(np.float64) / 255.0

    np.testing.assert_array_almost_equal(result, expected)
    assert result.dtype == np.float64
    assert transformed_metadata.dtype == np.float64
    assert transformed_metadata.value_range == (0.0, 1.0)


def test_mapping_transformer_preserve_range_keeps_metadata_range():
    """With preserve_range enabled, metadata.value_range is left unchanged."""

    image = np.array([[0, 127, 255]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, (0, 255), image.dtype)

    transformer = MappingTransformer(
        mapping_function=lambda value: value / 255.0,
        target_dtype=np.float64,
    )
    transformed_metadata = transformer.transform_metadata(metadata)

    assert transformed_metadata.dtype == np.float64
    assert transformed_metadata.value_range == (0.0, 1.0)


def test_log_intensity_mapping_uses_log_on_data():
    """Subclass test: LogIntensityMappingTransformer applies np.log to image data."""

    image = np.array([[1, 127, 255]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, (1, 255), image.dtype)

    transformer = LogIntensityMappingTransformer()
    result = transformer.transform_image(image, metadata, ImageAccessor())

    expected = np.log(image).astype(np.float32)
    np.testing.assert_array_almost_equal(result, expected)
    assert result.dtype == np.float32


def test_log1p_intensity_mapping_uses_log1p_on_data():
    """Subclass test: Log1pIntensityMappingTransformer applies np.log1p to image data."""

    image = np.array([[0, 127, 255]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, (0, 255), image.dtype)

    transformer = Log1pIntensityMappingTransformer()
    result = transformer.transform_image(image, metadata, ImageAccessor())

    expected = np.log1p(image).astype(np.float32)
    np.testing.assert_array_almost_equal(result, expected)
    assert result.dtype == np.float32


def test_gamma_intensity_mapping_uses_power_on_data():
    """Subclass test: GammaIntensityMappingTransformer applies power mapping to image data."""

    image = np.array([[1, 127, 255]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, (1, 255), image.dtype)

    transformer = GammaIntensityMappingTransformer(gamma=2.0)
    result = transformer.transform_image(image, metadata, ImageAccessor())

    expected = np.power(image, 2.0).astype(np.float32)
    np.testing.assert_array_almost_equal(result, expected)
    assert result.dtype == np.float32


def test_clip_intensity_mapping():
    """Test clipping intensity mapping with explicit vmin and vmax."""

    image = np.array([[10, 15, 20]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, (0, 255), image.dtype)

    transformer = ClipIntensityMappingTransformer(vmin=10, vmax=20)
    result = transformer.transform_image(image, metadata, ImageAccessor())

    expected = np.array([[10.0, 15.0, 20.0]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_brightness_contrast_intensity_mapping():
    """Test brightness/contrast mapping formula in native intensity range."""

    image = np.array([[0, 127, 255]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, (0, 255), image.dtype)

    transformer = BrightnessContrastIntensityMappingTransformer(brightness=0.1, contrast=1.2)
    result = transformer.transform_image(image, metadata, ImageAccessor())

    vmin, vmax = metadata.value_range
    midpoint = (vmin + vmax) / 2.0
    value_range = vmax - vmin
    expected = (image.astype(np.float32) - midpoint) * 1.2 + midpoint + 0.1 * value_range
    expected = np.clip(expected, vmin, vmax)

    np.testing.assert_array_almost_equal(result, expected)
    assert result.dtype == np.float32
    assert result.min() >= vmin
    assert result.max() <= vmax


def test_brightness_contrast_dtype_and_metadata():
    """Test output dtype and metadata update for brightness/contrast mapping."""

    image = np.array([[10, 20, 30]], dtype=np.uint8)
    metadata = ImageMetadata("image", image.shape, (0, 255), image.dtype)

    transformer = BrightnessContrastIntensityMappingTransformer(
        brightness=-0.2,
        contrast=0.8,
        target_dtype=np.float64,
    )
    result = transformer.transform_image(image, metadata, ImageAccessor())
    transformed_metadata = transformer.transform_metadata(metadata)

    assert result.dtype == np.float64
    assert transformed_metadata.dtype == np.float64
    assert transformed_metadata.value_range != metadata.value_range
