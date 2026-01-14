"""
Comprehensive tests for normalization transformers.

Combines unit tests with real image validation using Koala.jpg.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.metadata.dimensions import RGB, C, X, Y
from tiamat.transformers.normalization import MinMaxNormalizationTransformer


# Fixtures
@pytest.fixture
def koala_image():
    """Load Koala.jpg for real-world testing."""
    koala_path = Path(__file__).parent.parent.parent / "examples" / "data" / "Koala.jpg"

    if not koala_path.exists():
        pytest.skip(f"Koala image not found at {koala_path}")

    image_bgr = cv2.imread(str(koala_path))
    if image_bgr is None:
        pytest.skip(f"Could not load {koala_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture
def koala_metadata(koala_image):
    """Metadata for koala image."""
    return ImageMetadata(
        image_type="image",
        shape=koala_image.shape,
        value_range=(0, 255),
        dtype=koala_image.dtype,
        dimensions=(Y, X, RGB),
    )


# Unit Tests - Compact Version
class TestMinMaxNormalizationTransformer:
    """Compact unit tests for MinMaxNormalizationTransformer."""

    def test_initialization_accessor_and_basic_functionality(self):
        """Test init, accessor passthrough, and basic normalization formula."""
        # Test initialization with default and custom dtype
        t_f32 = MinMaxNormalizationTransformer()
        assert t_f32.target_dtype == np.float32

        t_f64 = MinMaxNormalizationTransformer(target_dtype=np.float64)
        assert t_f64.target_dtype == np.float64

        # Test accessor passthrough (unchanged)
        accessor = ImageAccessor(x=(0, 100), y=(0, 100), scale=0.5)
        metadata = ImageMetadata("image", (100, 100), (0, 255), np.uint8)
        assert t_f32.transform_access(accessor, metadata) is accessor

        # Test basic normalization formula: (value - vmin) / (vmax - vmin)
        image = np.array([[0, 127, 255]], dtype=np.uint8)
        result = t_f32.transform_image(image, metadata, accessor)

        expected = np.array([[0.0, 127 / 255, 1.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
        assert result.dtype == np.float32

    def test_metadata_updates_and_mutation_bug(self):
        """Test metadata updates dtype/value_range and check for mutation bug."""
        transformer = MinMaxNormalizationTransformer(target_dtype=np.float64)

        original = ImageMetadata("image", (50, 50, 3), (10, 200), np.uint8, dimensions=(Y, X, RGB))
        original_shape = original.shape
        original_dtype = original.dtype
        original_range = original.value_range
        original_id = id(original)

        result = transformer.transform_metadata(original)

        # Should update dtype and value_range
        assert result.dtype == np.float64
        assert result.value_range == (0.0, 1.0)

        # Should preserve other fields
        assert result.shape == (50, 50, 3)
        assert result.image_type == "image"
        assert result.dimensions == (Y, X, RGB)

        # CRITICAL BUG CHECK: Should NOT mutate original
        # This will FAIL with current code (missing dataclasses.replace)!
        assert original.dtype == original_dtype, "BUG: Original metadata was mutated! Missing dataclasses.replace()"
        assert (
            original.value_range == original_range
        ), "BUG: Original metadata was mutated! Missing dataclasses.replace()"
        assert id(result) != original_id

    def test_normalization_formula_with_various_ranges(self):
        """Test normalization formula with standard, custom, negative, and float ranges."""
        transformer = MinMaxNormalizationTransformer()
        accessor = ImageAccessor()

        # Test case 1: Standard range [0, 255]
        img1 = np.array([[0, 128, 255]], dtype=np.uint8)
        meta1 = ImageMetadata("image", img1.shape, (0, 255), np.uint8)
        result1 = transformer.transform_image(img1, meta1, accessor)
        np.testing.assert_array_almost_equal(result1, [[0.0, 128 / 255, 1.0]])

        # Test case 2: Custom range [20, 70]
        img2 = np.array([[20, 45, 70]], dtype=np.int16)
        meta2 = ImageMetadata("image", img2.shape, (20, 70), np.int16)
        result2 = transformer.transform_image(img2, meta2, accessor)
        np.testing.assert_array_almost_equal(result2, [[0.0, 0.5, 1.0]])

        # Test case 3: Negative range [-10, 40]
        img3 = np.array([[-10, 15, 40]], dtype=np.int16)
        meta3 = ImageMetadata("image", img3.shape, (-10, 40), np.int16)
        result3 = transformer.transform_image(img3, meta3, accessor)
        np.testing.assert_array_almost_equal(result3, [[0.0, 0.5, 1.0]])

        # Test case 4: Float input [0.0, 6.0]
        img4 = np.array([[0.0, 3.0, 6.0]], dtype=np.float32)
        meta4 = ImageMetadata("image", img4.shape, (0.0, 6.0), np.float32)
        result4 = transformer.transform_image(img4, meta4, accessor)
        np.testing.assert_array_almost_equal(result4, [[0.0, 0.5, 1.0]])

    def test_edge_cases_and_errors(self):
        """Test missing value_range, zero range, large range, and various shapes."""
        transformer = MinMaxNormalizationTransformer()
        accessor = ImageAccessor()

        # Error case: Missing value_range
        image = np.array([[1, 2, 3]], dtype=np.uint8)
        meta_no_range = ImageMetadata("image", image.shape, None, np.uint8)
        with pytest.raises(AssertionError, match="requires metadata.value_range"):
            transformer.transform_image(image, meta_no_range, accessor)

        # Edge case: Zero range (vmin == vmax) -> division by zero -> NaN
        img_zero = np.array([[100, 100, 100]], dtype=np.uint8)
        meta_zero = ImageMetadata("image", img_zero.shape, (100, 100), np.uint8)
        result_zero = transformer.transform_image(img_zero, meta_zero, accessor)
        assert np.all(np.isnan(result_zero))

        # Edge case: Very large dynamic range
        img_large = np.array([[0, 1000000, 2000000]], dtype=np.int32)
        meta_large = ImageMetadata("image", img_large.shape, (0, 2000000), np.int32)
        result_large = transformer.transform_image(img_large, meta_large, accessor)
        np.testing.assert_array_almost_equal(result_large, [[0.0, 0.5, 1.0]])

        # Test various shapes: 2D, 3D, 4D
        for shape in [(10, 20), (5, 10, 15), (2, 5, 10, 15)]:
            img = np.random.randint(0, 256, size=shape, dtype=np.uint8)
            meta = ImageMetadata("image", shape, (0, 255), np.uint8)
            result = transformer.transform_image(img, meta, accessor)

            assert result.shape == shape
            assert result.dtype == np.float32
            assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_different_dtypes_and_target_dtypes(self):
        """Test various input/output dtype combinations."""
        accessor = ImageAccessor()
        test_cases = [
            (np.uint8, 0, 255),
            (np.int16, -100, 100),
            (np.uint16, 0, 65535),
            (np.int32, 0, 1000000),
            (np.float32, 0.0, 1.0),
        ]

        # Test input dtypes: uint8, uint16, int16, int32, float32

        for input_dtype, vmin, vmax in test_cases:
            # Create test image with min, mid, max values
            if np.issubdtype(input_dtype, np.integer):
                img = np.array([[vmin, (vmin + vmax) // 2, vmax]], dtype=input_dtype)
            else:
                img = np.array([[vmin, (vmin + vmax) / 2, vmax]], dtype=input_dtype)

            meta = ImageMetadata("image", img.shape, (vmin, vmax), input_dtype)

            # Test both target dtypes: float32 and float64
            for target_dtype in [np.float32, np.float64]:
                transformer = MinMaxNormalizationTransformer(target_dtype=target_dtype)
                result = transformer.transform_image(img, meta, accessor)

                assert result.dtype == target_dtype
                np.testing.assert_array_almost_equal(result, [[0.0, 0.5, 1.0]], 0.001)


# Real Image Tests with Koala
class TestWithKoalaImage:
    """Test normalization with real Koala.jpg image."""

    def test_normalization_formula_and_correlation(self, koala_image, koala_metadata):
        """Test formula correctness and perfect correlation preservation."""
        transformer = MinMaxNormalizationTransformer()
        accessor = ImageAccessor()

        # Normalize
        result = transformer.transform_image(koala_image, koala_metadata, accessor)

        # Test 1: Output properties
        assert result.shape == koala_image.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

        # Test 2: Formula verification
        expected = koala_image.astype(np.float32) / 255.0
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

        # Test 3: Perfect correlation (normalization is linear transform)
        for channel in range(3):
            orig_flat = koala_image[:, :, channel].flatten().astype(np.float64)
            result_flat = result[:, :, channel].flatten().astype(np.float64)
            correlation = np.corrcoef(orig_flat, result_flat)[0, 1]
            assert correlation > 0.9999, f"Channel {channel}: correlation {correlation}"

    def test_roundtrip_and_metadata(self, koala_image, koala_metadata):
        """Test roundtrip (normalize→denormalize) and metadata updates."""
        transformer = MinMaxNormalizationTransformer()
        accessor = ImageAccessor()

        # Test metadata transformation
        new_metadata = transformer.transform_metadata(koala_metadata)
        assert new_metadata.dtype == np.float32
        assert new_metadata.value_range == (0.0, 1.0)
        assert new_metadata.shape == koala_image.shape

        # Normalize
        normalized = transformer.transform_image(koala_image, koala_metadata, accessor)

        # Denormalize back to [0, 255]
        denormalized = (normalized * 255.0).astype(np.uint8)

        # Should match original exactly
        np.testing.assert_array_equal(denormalized, koala_image)

    def test_custom_range_with_koala_crop(self, koala_image):
        """Test normalization with actual min/max from koala crop."""
        transformer = MinMaxNormalizationTransformer()
        accessor = ImageAccessor()

        # Take center crop
        h, w = koala_image.shape[:2]
        crop = koala_image[h // 4 : h // 2, w // 4 : w // 2]

        # Use actual min/max from crop
        crop_min = int(crop.min())
        crop_max = int(crop.max())

        crop_metadata = ImageMetadata("image", crop.shape, (crop_min, crop_max), crop.dtype, dimensions=(Y, X, RGB))

        # Normalize
        result = transformer.transform_image(crop, crop_metadata, accessor)

        # Test 1: Formula verification
        expected = (crop.astype(np.float32) - crop_min) / (crop_max - crop_min)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

        # Test 2: Range is [0, 1]
        assert result.min() >= 0.0
        assert result.max() <= 1.0

        # Test 3: Min/max values map to 0/1
        min_mask = crop == crop_min
        max_mask = crop == crop_max
        assert np.all(result[min_mask] < 0.01)
        assert np.all(result[max_mask] > 0.99)

    def test_grayscale_koala_and_different_dtypes(self, koala_image):
        """Test with grayscale koala and both float32/float64."""
        accessor = ImageAccessor()

        # Convert to grayscale
        koala_bgr = cv2.cvtColor(koala_image, cv2.COLOR_RGB2BGR)
        koala_gray = cv2.cvtColor(koala_bgr, cv2.COLOR_BGR2GRAY)

        gray_metadata = ImageMetadata("image", koala_gray.shape, (0, 255), koala_gray.dtype, dimensions=(Y, X))

        # Test both target dtypes
        for target_dtype in [np.float32, np.float64]:
            transformer = MinMaxNormalizationTransformer(target_dtype=target_dtype)

            result = transformer.transform_image(koala_gray, gray_metadata, accessor)

            # Verify dtype
            assert result.dtype == target_dtype
            assert result.shape == koala_gray.shape

            # Verify formula
            expected = koala_gray.astype(target_dtype) / 255.0
            np.testing.assert_array_almost_equal(result, expected)


# Integration Tests
class TestIntegrationAndPipeline:
    """Integration tests for full pipeline and metadata consistency."""

    def test_full_pipeline_metadata_access_image(self):
        """Test complete pipeline: metadata → access → image."""
        transformer = MinMaxNormalizationTransformer(target_dtype=np.float64)

        # Setup
        original_meta = ImageMetadata("image", (50, 50), (10, 200), np.uint8)
        original_accessor = ImageAccessor(x=(0, 50), y=(0, 50))
        image = np.random.randint(10, 201, size=(50, 50), dtype=np.uint8)

        # Step 1: Transform metadata
        new_meta = transformer.transform_metadata(original_meta)
        assert new_meta.dtype == np.float64
        assert new_meta.value_range == (0.0, 1.0)
        assert original_meta.dtype == np.uint8  # Not mutated (will fail!)

        # Step 2: Transform access
        new_accessor = transformer.transform_access(original_accessor, new_meta)
        assert new_accessor is original_accessor  # Unchanged

        # Step 3: Transform image
        result = transformer.transform_image(image, original_meta, new_accessor)

        assert result.dtype == np.float64
        assert result.shape == image.shape
        assert 0.0 <= result.min() <= result.max() <= 1.0

        # Verify formula
        expected = (image.astype(np.float64) - 10) / 190.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_metadata_preservation(self):
        """Test that additional_metadata is preserved through transformation."""
        original = ImageMetadata(
            "image",
            (50, 50, 3),
            (0, 255),
            np.uint8,
            dimensions=(Y, X, RGB),
            additional_metadata={"source": "test", "version": 1},
        )

        transformer = MinMaxNormalizationTransformer()
        result = transformer.transform_metadata(original)

        # Additional metadata should be preserved
        assert result.additional_metadata == {"source": "test", "version": 1}

        # Original not mutated (will fail!)
        assert original.additional_metadata == {"source": "test", "version": 1}


# Parametrized tests for compact coverage
@pytest.mark.parametrize(
    "value_range,test_values,expected",
    [
        ((0, 255), [0, 127, 255], [0.0, 127 / 255, 1.0]),
        ((0, 100), [0, 50, 100], [0.0, 0.5, 1.0]),
        ((-50, 50), [-50, 0, 50], [0.0, 0.5, 1.0]),
        ((10, 20), [10, 15, 20], [0.0, 0.5, 1.0]),
    ],
)
def test_various_value_ranges(value_range, test_values, expected):
    """Test normalization with various value ranges produces correct output."""
    transformer = MinMaxNormalizationTransformer()

    image = np.array([test_values], dtype=np.float32)
    metadata = ImageMetadata("image", image.shape, value_range, np.float32)

    result = transformer.transform_image(image, metadata, ImageAccessor())

    expected_array = np.array([expected], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected_array)
