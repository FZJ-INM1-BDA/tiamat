"""
Comprehensive tests for color transformers.

Combines unit tests with real image validation using Koala.jpg and reference libraries.
"""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.metadata.dimensions import RGB, RGBA, X, Y
from tiamat.transformers.color import (
    FloatToByteTransformer,
    GrayscaleToRGBTransformer,
    GrayscaleTransformer,
    LUTTransformer,
)


# Fixtures
@pytest.fixture
def koala_image():
    """Load Koala.jpg for real-world testing."""
    project_root = Path(__file__).resolve().parents[2]
    koala_path = project_root / "examples" / "data" / "Koala.jpg"

    if not koala_path.exists():
        pytest.skip(f"Koala image not found at {koala_path}")

    image_bgr = cv2.imread(str(koala_path))
    if image_bgr is None:
        pytest.skip(f"Could not load {koala_path}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


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
class TestLUTTransformer:
    """Compact unit tests for LUTTransformer."""

    def test_basic_functionality_and_errors(self):
        """Test initialization, transform methods, and error cases."""
        # Test various colormap types
        color_map_array = np.array([[1, 0, 0], [0, 1, 0]])
        for cmap in ["viridis", color_map_array, [(1, 0, 0)]]:
            transformer = LUTTransformer(color_map=cmap)
            if isinstance(cmap, np.ndarray):
                np.testing.assert_array_equal(transformer.color_map, cmap)
            else:
                assert transformer.color_map == cmap

        # Test accessor unchanged
        transformer = LUTTransformer(color_map=color_map_array)
        accessor = ImageAccessor(x=(0, 10))
        metadata = ImageMetadata("image", (10, 10), (0, 1), np.uint8)
        assert transformer.transform_access(accessor, metadata) is accessor

        # Test missing value_range error
        with pytest.raises(AssertionError, match="requires metadata.value_range"):
            metadata_no_range = ImageMetadata("image", (10, 10), None, np.uint8)
            transformer.transform_image(np.zeros((10, 10), np.uint8), metadata_no_range, accessor)

        # Test invalid colormap type
        bad_transformer = LUTTransformer(color_map=123)
        with pytest.raises(RuntimeError, match="Unknown type"):
            bad_transformer.transform_image(np.zeros((10, 10), np.uint8), metadata, accessor)

        # Test out of bounds indices
        transformer = LUTTransformer(color_map=color_map_array)
        with pytest.raises(IndexError):
            transformer.transform_image(np.array([[5]], dtype=np.uint8), metadata, accessor)

    def test_metadata_transformation_and_mutation(self):
        """Test metadata updates and non-mutation."""
        color_map = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        transformer = LUTTransformer(color_map=color_map)

        original = ImageMetadata("image", (50, 50), (0, 1), np.uint8, dimensions=(Y, X))
        original_shape = original.shape
        original_id = id(original)

        result = transformer.transform_metadata(original)

        # Should add color dimension
        assert result.shape == (50, 50, 3)
        assert result.dtype == np.float32

        # Should not mutate original
        assert original.shape == original_shape
        assert id(result) != original_id

    def test_exact_color_mapping(self):
        """Test LUT produces exact mappings."""
        color_map = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        transformer = LUTTransformer(color_map=color_map)

        image = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        metadata = ImageMetadata("image", image.shape, (0, 3), np.uint8)

        result = transformer.transform_image(image, metadata, ImageAccessor())

        np.testing.assert_array_equal(result[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(result[0, 1], [1, 0, 0])
        np.testing.assert_array_equal(result[0, 2], [0, 1, 0])
        np.testing.assert_array_equal(result[0, 3], [0, 0, 1])


class TestGrayscaleTransformer:
    """Compact unit tests for GrayscaleTransformer."""

    def test_basic_functionality_and_dimensions(self):
        """Test transform methods and dimension handling."""
        transformer = GrayscaleTransformer()

        # Test accessor unchanged
        accessor = ImageAccessor(x=(0, 10))
        metadata = ImageMetadata("image", (100, 100, 3), (0, 255), np.uint8, dimensions=(Y, X, RGB))
        assert transformer.transform_access(accessor, metadata) is accessor

        # Test RGB dimension removal
        result = transformer.transform_metadata(metadata)
        assert result.shape == (100, 100)
        assert RGB not in result.dimensions

        # Test RGBA dimension removal
        rgba_meta = ImageMetadata("image", (100, 100, 4), (0, 255), np.uint8, dimensions=(Y, X, RGBA))
        result = transformer.transform_metadata(rgba_meta)
        assert result.shape == (100, 100)
        assert RGBA not in result.dimensions

        # Test grayscale preserved
        gray_meta = ImageMetadata("image", (100, 100), (0, 255), np.uint8, dimensions=(Y, X))
        result = transformer.transform_metadata(gray_meta)
        assert result.shape == (100, 100)

    def test_metadata_non_mutation(self):
        """Test metadata is not mutated."""
        transformer = GrayscaleTransformer()
        original = ImageMetadata("image", (100, 100, 3), (0, 255), np.uint8, dimensions=(Y, X, RGB))
        original_shape = original.shape

        result = transformer.transform_metadata(original)

        assert original.shape == original_shape
        assert RGB in original.dimensions
        assert id(result) != id(original)

    @patch("cv2.cvtColor")
    def test_image_conversion(self, mock_cvt):
        """Test image conversion calls OpenCV correctly."""
        mock_cvt.return_value = np.array([[100]], dtype=np.uint8)

        transformer = GrayscaleTransformer()
        image = np.array([[[255, 0, 0]]], dtype=np.uint8)
        metadata = ImageMetadata("image", image.shape, (0, 255), np.uint8, dimensions=(Y, X, RGB))

        result = transformer.transform_image(image, metadata, ImageAccessor())
        mock_cvt.assert_called_once()
        assert result.shape == (1, 1)

        # Test already grayscale (no conversion needed)
        gray_image = np.array([[100]], dtype=np.uint8)
        gray_meta = ImageMetadata("image", gray_image.shape, (0, 255), np.uint8, dimensions=(Y, X))
        result = transformer.transform_image(gray_image, gray_meta, ImageAccessor())
        np.testing.assert_array_equal(result, gray_image)


class TestGrayscaleToRGBTransformer:
    """Compact unit tests for GrayscaleToRGBTransformer."""

    def test_basic_functionality_and_dimensions(self):
        """Test transform methods and dimension handling."""
        transformer = GrayscaleToRGBTransformer()
        meta = ImageMetadata("image", (100, 100), (0, 255), np.uint8, dimensions=(Y, X))
        # Test accessor unchanged
        accessor = ImageAccessor()
        assert transformer.transform_access(accessor, meta) is accessor

        # Test RGB dimension added
        gray_meta = ImageMetadata("image", (100, 100), (0, 255), np.uint8, dimensions=(Y, X))
        result = transformer.transform_metadata(gray_meta)
        assert result.shape == (100, 100, 3)
        assert RGB in result.dimensions

        # Test RGB preserved
        rgb_meta = ImageMetadata("image", (100, 100, 3), (0, 255), np.uint8, dimensions=(Y, X, RGB))
        result = transformer.transform_metadata(rgb_meta)
        assert result.shape == (100, 100, 3)

    def test_metadata_non_mutation(self):
        """Test metadata is not mutated."""
        transformer = GrayscaleToRGBTransformer()
        original = ImageMetadata("image", (100, 100), (0, 255), np.uint8, dimensions=(Y, X))
        original_shape = original.shape

        result = transformer.transform_metadata(original)

        assert original.shape == original_shape
        assert id(result) != id(original)

    @patch("cv2.cvtColor")
    def test_image_conversion(self, mock_cvt):
        """Test grayscale to RGB conversion."""
        mock_cvt.return_value = np.array([[[100, 100, 100]]], dtype=np.uint8)

        transformer = GrayscaleToRGBTransformer()
        image = np.array([[100]], dtype=np.uint8)
        metadata = ImageMetadata("image", image.shape, (0, 255), np.uint8, dimensions=(Y, X))

        result = transformer.transform_image(image, metadata, ImageAccessor())

        mock_cvt.assert_called_once()

        # Test already RGB (no conversion needed)
        rgb_image = np.array([[[255, 0, 0]]], dtype=np.uint8)
        rgb_meta = ImageMetadata("image", rgb_image.shape, (0, 255), np.uint8, dimensions=(Y, X, RGB))
        result = transformer.transform_image(rgb_image, rgb_meta, ImageAccessor())
        np.testing.assert_array_equal(result, rgb_image)


class TestFloatToByteTransformer:
    """Compact unit tests for FloatToByteTransformer."""

    def test_basic_functionality(self):
        """Test conversion and metadata updates."""
        transformer = FloatToByteTransformer()

        # Test accessor unchanged
        accessor = ImageAccessor()
        metadata = ImageMetadata("image", (10, 10), (0.0, 1.0), np.float32)
        assert transformer.transform_access(accessor, metadata) is accessor

        # Test metadata value_range update
        result_meta = transformer.transform_metadata(metadata)
        assert result_meta.value_range == (0, 255)

        # Test float32 conversion
        float_img = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = transformer.transform_image(float_img, metadata, accessor)
        np.testing.assert_array_equal(result, np.array([[0, 127, 255]], dtype=np.uint8))

        # Test integer unchanged
        int_img = np.array([[100, 150, 200]], dtype=np.uint8)
        int_meta = ImageMetadata("image", int_img.shape, (0, 255), np.uint8)
        result = transformer.transform_image(int_img, int_meta, accessor)
        np.testing.assert_array_equal(result, int_img)

    def test_edge_cases(self):
        """Test edge cases and special values."""
        transformer = FloatToByteTransformer()
        metadata = ImageMetadata("image", (1, 3), (0.0, 1.0), np.float32)

        # Test all zeros/ones
        zeros = np.zeros((5, 5), dtype=np.float32)
        result = transformer.transform_image(zeros, metadata, ImageAccessor())
        assert np.all(result == 0)

        ones = np.ones((5, 5), dtype=np.float32)
        result = transformer.transform_image(ones, metadata, ImageAccessor())
        assert np.all(result == 255)


# Real Image Tests with Koala
class TestWithKoalaImage:
    """Test transformers with real Koala.jpg image against reference libraries."""

    def test_grayscale_matches_opencv(self, koala_image, koala_metadata):
        """Test grayscale conversion matches OpenCV exactly."""
        transformer = GrayscaleTransformer()
        accessor = ImageAccessor()

        # tiamat implementation
        tiamat_result = transformer.transform_image(koala_image, koala_metadata, accessor)
        # OpenCV reference - EXACT SAME STEPS

        opencv_result = cv2.cvtColor(koala_image, cv2.COLOR_RGB2GRAY)
        # Should match exactly
        np.testing.assert_equal(opencv_result, tiamat_result)
        # np.testing.assert_array_equal(tiamat_result, opencv_result)

        # Additional validation
        assert tiamat_result.shape == koala_image.shape[:2]
        assert tiamat_result.dtype == np.uint8
        assert tiamat_result.min() >= 0
        assert tiamat_result.max() <= 255

    def test_grayscale_matches_pil(self, koala_image):
        """Test grayscale conversion matches PIL."""
        transformer = GrayscaleTransformer()

        metadata = ImageMetadata(
            image_type="image",
            shape=koala_image.shape,
            value_range=(0, 255),
            dtype=koala_image.dtype,
            dimensions=(Y, X, RGB),
        )

        # tiamat implementationa
        tiamat_result = transformer.transform_image(koala_image, metadata, ImageAccessor())

        # PIL reference - EXACT SAME STEPS
        # Step 1: Create PIL Image from RGB array
        pil_img = Image.fromarray(koala_image, mode="RGB")
        # Step 2: Convert to grayscale
        pil_gray = pil_img.convert("L")
        # Step 3: Convert back to numpy
        pil_result = np.array(pil_gray)

        # Should be very close (PIL and OpenCV use slightly different formulas)
        # OpenCV: 0.299*R + 0.587*G + 0.114*B
        # PIL: Similar but may have minor differences
        correlation = np.corrcoef(tiamat_result.flatten(), pil_result.flatten())[0, 1]
        assert correlation > 0.99, f"Correlation with PIL: {correlation}"

    def test_grayscale_to_rgb_matches_opencv(self, koala_image, koala_metadata):
        """Test RGB→Gray→RGB pipeline matches OpenCV exactly."""
        to_gray = GrayscaleTransformer()
        to_rgb = GrayscaleToRGBTransformer()
        accessor = ImageAccessor()

        # tiamat pipeline
        # Step 1: RGB to grayscale
        tiamat_gray = to_gray.transform_image(koala_image, koala_metadata, accessor)

        gray_metadata = ImageMetadata(
            image_type="image",
            shape=tiamat_gray.shape,
            value_range=(0, 255),
            dtype=tiamat_gray.dtype,
            dimensions=(Y, X),
        )

        # Step 2: Grayscale to RGB
        tiamat_result = to_rgb.transform_image(tiamat_gray, gray_metadata, accessor)

        # OpenCV pipeline - EXACT SAME STEPS

        opencv_gray = cv2.cvtColor(koala_image, cv2.COLOR_RGB2GRAY)

        opencv_result = cv2.cvtColor(opencv_gray, cv2.COLOR_GRAY2RGB)

        # Should match exactly
        np.testing.assert_array_equal(tiamat_result, opencv_result)

        # Verify all channels are identical (grayscale replicated)
        np.testing.assert_array_equal(tiamat_result[:, :, 0], tiamat_result[:, :, 1])
        np.testing.assert_array_equal(tiamat_result[:, :, 1], tiamat_result[:, :, 2])

        # Verify first channel matches grayscale
        np.testing.assert_array_equal(tiamat_result[:, :, 0], tiamat_gray)

    def test_float_byte_roundtrip_with_koala(self, koala_image, koala_metadata):
        """Test float→byte conversion preserves image quality."""
        transformer = FloatToByteTransformer()
        accessor = ImageAccessor()

        # Convert to float [0, 1]
        koala_float = koala_image.astype(np.float32) / 255.0

        float_metadata = ImageMetadata(
            image_type="image",
            shape=koala_float.shape,
            value_range=(0.0, 1.0),
            dtype=np.float32,
            dimensions=(Y, X, RGB),
        )

        # tiamat implementation: float to byte
        tiamat_result = transformer.transform_image(koala_float, float_metadata, accessor)

        # Direct formula (what we expect)
        expected_result = (koala_float * 255).astype(np.uint8)

        # Should match exactly
        np.testing.assert_array_equal(tiamat_result, expected_result)

        # Should be very close to original
        max_diff = np.max(np.abs(koala_image.astype(int) - tiamat_result.astype(int)))
        assert max_diff <= 1, f"Max difference: {max_diff}"

    def test_full_pipeline_matches_opencv(self, koala_image, koala_metadata):
        """Test complete pipeline: RGB→Gray→Float→Byte→RGB matches OpenCV."""
        to_gray = GrayscaleTransformer()
        to_rgb = GrayscaleToRGBTransformer()
        to_byte = FloatToByteTransformer()
        accessor = ImageAccessor()

        # tiamat pipeline
        # Step 1: RGB to Gray
        gray = to_gray.transform_image(koala_image, koala_metadata, accessor)

        # Step 2: Gray to Float
        gray_float = gray.astype(np.float32) / 255.0

        # Step 3: Float to Byte
        float_meta = ImageMetadata("image", gray_float.shape, (0.0, 1.0), np.float32)
        byte_gray = to_byte.transform_image(gray_float, float_meta, accessor)

        # Step 4: Byte to RGB
        byte_meta = ImageMetadata("image", byte_gray.shape, (0, 255), np.uint8, dimensions=(Y, X))
        tiamat_final = to_rgb.transform_image(byte_gray, byte_meta, accessor)

        # OpenCV pipeline - EXACT SAME STEPS
        # Step 1: RGB to BGR
        koala_bgr = cv2.cvtColor(koala_image, cv2.COLOR_RGB2BGR)
        # Step 2: BGR to GRAY
        opencv_gray = cv2.cvtColor(koala_bgr, cv2.COLOR_BGR2GRAY)
        # Step 3: Gray to Float
        opencv_float = opencv_gray.astype(np.float32) / 255.0
        # Step 4: Float to Byte
        opencv_byte = (opencv_float * 255).astype(np.uint8)
        # Step 5: GRAY to BGR
        opencv_bgr_result = cv2.cvtColor(opencv_byte, cv2.COLOR_GRAY2BGR)
        # Step 6: BGR to RGB
        opencv_final = cv2.cvtColor(opencv_bgr_result, cv2.COLOR_BGR2RGB)

        # Should match exactly (or within 1 due to rounding)
        max_diff = np.max(np.abs(tiamat_final.astype(int) - opencv_final.astype(int)))
        assert max_diff <= 1, f"Max difference in pipeline: {max_diff}"

        # Verify shape preserved
        assert tiamat_final.shape == koala_image.shape


# Integration and Edge Cases
class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases."""

    def test_metadata_preservation_through_pipeline(self):
        """Test metadata preserved through multi-step pipeline."""
        original = ImageMetadata(
            "image",
            (50, 50, 3),
            (0, 255),
            np.uint8,
            dimensions=(Y, X, RGB),
            additional_metadata={"source": "test", "version": 1},
        )

        transformers = [GrayscaleTransformer(), FloatToByteTransformer()]

        current = original
        for t in transformers:
            result = t.transform_metadata(current)

            # Additional metadata preserved
            assert result.additional_metadata == original.additional_metadata

            # Original not mutated
            assert current.additional_metadata == original.additional_metadata

            current = result

    def test_all_transformers_accessor_preservation(self):
        """Test all transformers preserve accessor."""
        accessor = ImageAccessor(x=(10, 50), y=(20, 60), scale=0.5)
        metadata = ImageMetadata("image", (100, 100, 3), (0, 255), np.uint8, dimensions=(Y, X, RGB))

        # Test LUTTransformer
        lut = LUTTransformer(np.array([[1, 0, 0]]))
        assert lut.transform_access(accessor, metadata) is accessor

        # Test GrayscaleTransformer
        gray = GrayscaleTransformer()
        assert gray.transform_access(accessor, metadata) is accessor

        # Test GrayscaleToRGBTransformer
        to_rgb = GrayscaleToRGBTransformer()
        assert to_rgb.transform_access(accessor, metadata) is accessor

        # Test FloatToByteTransformer
        to_byte = FloatToByteTransformer()
        assert to_byte.transform_access(accessor, metadata) is accessor


# Metadata-Image Consistency Tests
class TestPixelValueConsistency:
    """Test that metadata.value_range matches actual pixel values after transformation."""

    def test_lut_transformer_metadata_consistency(self):
        """Check that LUT metadata.value_range matches actual pixel values."""
        # Note: This test reveals that LUTTransformer SHOULD update value_range but doesn't!
        # Matplotlib colormaps output float [0, 1] but metadata still says (0, 255)

        color_map = "Blues"
        transformer = LUTTransformer(color_map=color_map)

        image = np.array(np.arange(255), dtype=np.uint8)
        metadata = ImageMetadata("image", image.shape, (0, 255), np.uint8)

        result = transformer.transform_image(image, metadata, ImageAccessor())
        result_meta = transformer.transform_metadata(metadata)

        # Actual pixel range
        vmin, vmax = float(result.min()), float(result.max())

        # Metadata should reflect actual range
        # For numpy array colormap, it should match
        assert result_meta.value_range is not None, "LUT should set value_range"
        mmin, mmax = result_meta.value_range

        # Should be close
        assert np.allclose(vmin, mmin, atol=0.1), f"Min mismatch: {vmin} vs {mmin}"
        assert np.allclose(vmax, mmax, atol=0.1), f"Max mismatch: {vmax} vs {mmax}"

    def test_grayscale_metadata_consistency(self):
        """Check that grayscale conversion preserves value_range."""
        transformer = GrayscaleTransformer()

        # RGB image with known range
        image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        metadata = ImageMetadata("image", image.shape, (0, 255), np.uint8, dimensions=(Y, X, RGB))

        result = transformer.transform_image(image, metadata, ImageAccessor())
        result_meta = transformer.transform_metadata(metadata)

        # Grayscale should maintain roughly the same range
        vmin, vmax = int(result.min()), int(result.max())
        mmin, mmax = result_meta.value_range

        # Should still be (0, 255) for uint8 grayscale
        assert mmin == 0
        assert mmax == 255

        # Actual values should be in range
        assert vmin >= mmin
        assert vmax <= mmax

    def test_grayscale_to_rgb_metadata_consistency(self):
        """Check that gray to RGB preserves value_range."""
        transformer = GrayscaleToRGBTransformer()

        image = np.array([[0, 128, 255]], dtype=np.uint8)
        metadata = ImageMetadata("image", image.shape, (0, 255), np.uint8, dimensions=(Y, X))

        result = transformer.transform_image(image, metadata, ImageAccessor())
        result_meta = transformer.transform_metadata(metadata)

        # Should preserve range
        vmin, vmax = int(result.min()), int(result.max())
        mmin, mmax = result_meta.value_range

        assert mmin == 0
        assert mmax == 255
        assert vmin >= mmin
        assert vmax <= mmax

    def test_float_to_byte_metadata_consistency(self):
        """Check that float to byte updates value_range correctly."""
        transformer = FloatToByteTransformer()

        image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        metadata = ImageMetadata("image", image.shape, (0.0, 1.0), np.float32)

        result = transformer.transform_image(image, metadata, ImageAccessor())
        result_meta = transformer.transform_metadata(metadata)

        # Should update to byte range
        vmin, vmax = int(result.min()), int(result.max())
        mmin, mmax = result_meta.value_range

        assert mmin == 0
        assert mmax == 255
        assert vmin >= mmin
        assert vmax <= mmax

    def test_consistency_with_koala(self, koala_image, koala_metadata):
        """Test metadata consistency on real koala image through pipeline."""
        # RGB -> Gray -> Float -> Byte -> RGB
        to_gray = GrayscaleTransformer()
        to_byte = FloatToByteTransformer()
        to_rgb = GrayscaleToRGBTransformer()

        # Step 1: RGB to Gray
        gray = to_gray.transform_image(koala_image, koala_metadata, ImageAccessor())
        gray_meta = to_gray.transform_metadata(koala_metadata)

        # Check consistency
        assert gray.min() >= gray_meta.value_range[0]
        assert gray.max() <= gray_meta.value_range[1]

        # Step 2: Gray to Float
        float_gray = gray.astype(np.float32) / 255.0
        float_meta = ImageMetadata("image", float_gray.shape, (0.0, 1.0), np.float32)

        # Step 3: Float to Byte
        byte_gray = to_byte.transform_image(float_gray, float_meta, ImageAccessor())
        byte_meta = to_byte.transform_metadata(float_meta)

        # Check consistency
        assert byte_gray.min() >= byte_meta.value_range[0]
        assert byte_gray.max() <= byte_meta.value_range[1]

        # Step 4: Byte to RGB
        final = to_rgb.transform_image(byte_gray, byte_meta, ImageAccessor())
        final_meta = to_rgb.transform_metadata(byte_meta)

        # Check consistency
        assert final.min() >= final_meta.value_range[0]
        assert final.max() <= final_meta.value_range[1]


# Parametrized tests for compact coverage
@pytest.mark.parametrize(
    "input_dtype,is_float",
    [(np.uint8, False), (np.uint16, False), (np.int16, False), (np.float32, True), (np.float64, True)],
)
def test_float_to_byte_dtypes(input_dtype, is_float):
    """Test FloatToByteTransformer with various dtypes."""
    transformer = FloatToByteTransformer()

    if is_float:
        image = np.array([[0.5]], dtype=input_dtype)
        metadata = ImageMetadata("image", image.shape, (0.0, 1.0), input_dtype)
        result = transformer.transform_image(image, metadata, ImageAccessor())
        assert result.dtype == np.uint8
        assert result[0, 0] == 127
    else:
        image = np.array([[100]], dtype=input_dtype)
        metadata = ImageMetadata("image", image.shape, (0, 255), input_dtype)
        result = transformer.transform_image(image, metadata, ImageAccessor())
        assert result.dtype == input_dtype
        np.testing.assert_array_equal(result, image)
