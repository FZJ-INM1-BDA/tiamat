"""
Comprehensive tests for view transformers.

Combines unit tests with real image validation using Koala.jpg.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.metadata.dimensions import RGB, X, Y, Z
from tiamat.transformers.view import BoundingBoxTransformer


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

    # Convert BGR to RGB
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
class TestBoundingBoxTransformer:
    """Compact unit tests for BoundingBoxTransformer."""

    def test_initialization_and_basic_functionality(self):
        """Test initialization with various bound types."""
        # Test with tuples
        bbox1 = BoundingBoxTransformer(
            bounds_x=(10, 100),
            bounds_y=(20, 200),
            bounds_z=(5, 50),
        )
        assert bbox1.bounds_x == (10, 100)
        assert bbox1.bounds_y == (20, 200)
        assert bbox1.bounds_z == (5, 50)

        # Test with single int
        bbox2 = BoundingBoxTransformer(bounds_x=50, bounds_y=100)
        assert bbox2.bounds_x == 50
        assert bbox2.bounds_y == 100

        # Test with None (no bounds)
        bbox3 = BoundingBoxTransformer()
        assert bbox3.bounds_x is None
        assert bbox3.bounds_y is None
        assert bbox3.bounds_z is None

    def test_get_coordinate_shape(self):
        """Test coordinate shape computation."""
        # Test simple slice
        shape = BoundingBoxTransformer.get_coordinate_shape(
            coord=(0, 100),
            image_dimension=100,
        )
        assert shape == 100

        # Test with scaling
        shape_scaled = BoundingBoxTransformer.get_coordinate_shape(
            coord=(0, 50),
            image_dimension=100,
            coord_scale=0.5,
        )
        assert shape_scaled == 100  # !!!hier sollte doch 100 erwartet werden, aber es wurde 50 erwartet.

    def test_crop_coordinate(self):
        """Test coordinate cropping logic."""
        # Test basic crop
        out_coords, residuals = BoundingBoxTransformer.crop_coordinate(
            coord_slice=(0, 100),
            bounds_slice=(10, 90),
            image_dimension=90,
        )

        # Output should be within bounds
        assert out_coords[0] >= 10
        assert out_coords[1] <= 90

        # Test with no overlap (requires padding)
        out_coords2, residuals2 = BoundingBoxTransformer.crop_coordinate(
            coord_slice=(0, 50),
            bounds_slice=(60, 100),
            image_dimension=100,
        )

        # Should have residual padding
        assert residuals2[0] > 0 or residuals2[1] > 0

    def test_metadata_transformation_and_mutation(self):
        """Test metadata updates and non-mutation."""
        bbox = BoundingBoxTransformer(bounds_x=(10, 90), bounds_y=(20, 180))

        original = ImageMetadata(
            "image",
            (200, 100, 3),
            (0, 255),
            np.uint8,
            dimensions=(Y, X, RGB),
        )
        original_shape = original.shape
        original_id = id(original)

        result = bbox.transform_metadata(original)

        # Should update shape
        assert result.shape != original_shape
        print(result.shape, original_shape)
        # Y: 20 to 180 = 160
        # X: 10 to 90 = 80
        # RGB: unchanged = 3
        assert result.shape == (160, 80, 3)

        # Should preserve other fields
        assert result.image_type == "image"
        assert result.value_range == (0, 255)
        assert result.dtype == np.uint8

        # CRITICAL: Should not mutate original
        assert original.shape == original_shape
        assert id(result) != original_id

    def test_bounds_spatial_shape_2d(self):
        """Test spatial shape computation for 2D images."""
        bbox = BoundingBoxTransformer(bounds_x=(0, 50), bounds_y=(0, 100))

        spatial_shape = (200, 100)  # (Y, X)
        result = bbox.bounds_spatial_shape(spatial_shape)

        # Should match bounds
        assert result == (100, 50)  # (Y from 0-100, X from 0-50)

    def test_bounds_spatial_shape_3d(self):
        """Test spatial shape computation for 3D images."""
        bbox = BoundingBoxTransformer(bounds_x=(0, 50), bounds_y=(0, 100), bounds_z=(0, 20))

        spatial_shape = (30, 200, 100)  # (Z, Y, X)
        result = bbox.bounds_spatial_shape(spatial_shape)

        # Should match bounds
        assert result == (20, 100, 50)  # (Z, Y, X)

    def test_transform_access_basic(self):
        """Test accessor transformation."""
        bbox = BoundingBoxTransformer(bounds_x=(10, 90), bounds_y=(20, 180))

        metadata = ImageMetadata("image", (200, 100, 3), (0, 255), np.uint8, dimensions=(Y, X, RGB))

        accessor = ImageAccessor(x=(0, 100), y=(0, 200))

        result = bbox.transform_access(accessor, metadata)

        # Should not be same object (replaced)
        assert result is not accessor

        # Should have cropped coordinates
        assert result.x is not None
        assert result.y is not None

        # Should store residuals in history
        assert id(bbox) in result.history

    def test_transform_image_no_padding(self):
        """Test image transformation without padding."""
        bbox = BoundingBoxTransformer(bounds_x=(10, 90), bounds_y=(20, 180))

        metadata = ImageMetadata("image", (160, 80, 3), (0, 255), np.uint8, dimensions=(Y, X, RGB))

        # Create test image
        image = np.random.randint(0, 256, size=(160, 80, 3), dtype=np.uint8)

        # Create accessor with no padding needed (residuals = 0)
        accessor = ImageAccessor()
        accessor.fill_value = None
        accessor.history[id(bbox)] = [(0, 0), (0, 0)]  # No residuals

        result = bbox.transform_image(image, metadata, accessor)

        # Should be unchanged (no padding)
        np.testing.assert_array_equal(result, image)

    def test_transform_image_with_padding(self):
        """Test image transformation with padding."""
        bbox = BoundingBoxTransformer(bounds_x=(10, 90), bounds_y=(20, 180))

        metadata = ImageMetadata("image", (150, 70, 3), (0, 255), np.uint8, dimensions=(Y, X, RGB))

        # Create test image
        image = np.random.randint(0, 256, size=(150, 70, 3), dtype=np.uint8)

        # Create accessor with padding needed
        accessor = ImageAccessor()
        accessor.fill_value = 0
        accessor.history[id(bbox)] = [(5, 5), (10, 10)]  # Residuals: (Y, X)

        result = bbox.transform_image(image, metadata, accessor)

        # Should be padded
        assert result.shape == (160, 90, 3)  # 150+10, 70+20

        # Original image should be inside
        np.testing.assert_array_equal(result[5:-5, 10:-10, :], image)

        # Padding should be fill_value (0)
        assert np.all(result[:5, :, :] == 0)  # Top padding
        assert np.all(result[-5:, :, :] == 0)  # Bottom padding
        assert np.all(result[:, :10, :] == 0)  # Left padding
        assert np.all(result[:, -10:, :] == 0)  # Right padding


# Real Image Tests with Koala
class TestWithKoalaImage:
    """Test BoundingBoxTransformer with real Koala.jpg image."""

    def test_center_crop_koala(self, koala_image, koala_metadata):
        """Test center cropping on koala image."""
        h, w = koala_image.shape[:2]

        # Crop to center 50%
        crop_h, crop_w = h // 2, w // 2
        start_y, start_x = h // 4, w // 4

        bbox = BoundingBoxTransformer(bounds_x=(start_x, start_x + crop_w), bounds_y=(start_y, start_y + crop_h))

        # Transform metadata
        new_metadata = bbox.transform_metadata(koala_metadata)
        assert new_metadata.shape == (crop_h, crop_w, 3)

        # Transform accessor
        accessor = ImageAccessor(x=(0, w), y=(0, h))
        new_accessor = bbox.transform_access(accessor, koala_metadata)

        # Manually crop for comparison
        expected_crop = koala_image[start_y : start_y + crop_h, start_x : start_x + crop_w]

        # Transform image (with accessor that simulates the crop)
        # In real usage, the reader would provide the cropped image
        # Here we simulate by setting appropriate residuals
        accessor_for_image = ImageAccessor()
        accessor_for_image.fill_value = None
        accessor_for_image.history[id(bbox)] = [(0, 0), (0, 0)]  # No padding

        # The actual crop would happen in the reader, here we verify dimensions
        assert new_metadata.shape[:2] == expected_crop.shape[:2]

    def test_bounds_beyond_image_with_koala(self, koala_image, koala_metadata):
        """Test bounding box extending beyond image (requires padding)."""
        h, w = koala_image.shape[:2]

        # Bounds that extend beyond image
        bbox = BoundingBoxTransformer(bounds_x=(-10, w + 10), bounds_y=(-20, h + 20))

        # Transform metadata
        new_metadata = bbox.transform_metadata(koala_metadata)

        # Should be larger than original (due to negative start)
        assert new_metadata.shape[0] > h  # Y dimension
        assert new_metadata.shape[1] > w  # X dimension

    def test_small_crop_from_koala(self, koala_image, koala_metadata):
        """Test small crop from koala (e.g., koala's face region)."""
        h, w = koala_image.shape[:2]

        # Small crop (simulating face region)
        bbox = BoundingBoxTransformer(bounds_x=(w // 3, 2 * w // 3), bounds_y=(h // 3, 2 * h // 3))

        # Transform metadata
        new_metadata = bbox.transform_metadata(koala_metadata)

        expected_h = h // 3
        expected_w = w // 3

        assert new_metadata.shape == (expected_h, expected_w, 3)

        # Original should not be mutated
        assert koala_metadata.shape == koala_image.shape

    def test_edge_crop_from_koala(self, koala_image, koala_metadata):
        """Test edge crops (top-left, bottom-right, etc.)."""
        h, w = koala_image.shape[:2]

        # Test top-left corner
        bbox_tl = BoundingBoxTransformer(bounds_x=(0, w // 4), bounds_y=(0, h // 4))
        meta_tl = bbox_tl.transform_metadata(koala_metadata)
        assert meta_tl.shape == (h // 4, w // 4, 3)

        # Test bottom-right corner
        bbox_br = BoundingBoxTransformer(bounds_x=(3 * w // 4, w), bounds_y=(3 * h // 4, h))
        meta_br = bbox_br.transform_metadata(koala_metadata)
        assert meta_br.shape == (h // 4, w // 4, 3)

        # Original unchanged
        assert koala_metadata.shape == koala_image.shape

    def test_strip_crops_from_koala(self, koala_image, koala_metadata):
        """Test horizontal and vertical strip crops."""
        h, w = koala_image.shape[:2]

        # Horizontal strip (middle third)
        bbox_h = BoundingBoxTransformer(bounds_x=(0, w), bounds_y=(h // 3, 2 * h // 3))
        meta_h = bbox_h.transform_metadata(koala_metadata)
        assert meta_h.shape == (h // 3, w, 3)

        # Vertical strip (middle third)
        bbox_v = BoundingBoxTransformer(bounds_x=(w // 3, 2 * w // 3), bounds_y=(0, h))
        meta_v = bbox_v.transform_metadata(koala_metadata)
        assert meta_v.shape == (h, w // 3, 3)

    def test_multiple_crops_pipeline(self, koala_image, koala_metadata):
        """Test applying multiple crops in sequence."""
        h, w = koala_image.shape[:2]

        # First crop: center 80%
        bbox1 = BoundingBoxTransformer(bounds_x=(w // 10, 9 * w // 10), bounds_y=(h // 10, 9 * h // 10))
        meta1 = bbox1.transform_metadata(koala_metadata)

        # Second crop: center 50% of the first crop
        h1, w1 = meta1.shape[:2]
        bbox2 = BoundingBoxTransformer(bounds_x=(w1 // 4, 3 * w1 // 4), bounds_y=(h1 // 4, 3 * h1 // 4))
        meta2 = bbox2.transform_metadata(meta1)

        # Final size should be 40% of original (0.8 * 0.5)
        expected_h = int(h * 0.8 * 0.5)
        expected_w = int(w * 0.8 * 0.5)

        assert abs(meta2.shape[0] - expected_h) <= 2  # Allow small rounding
        assert abs(meta2.shape[1] - expected_w) <= 2

        # Original still unchanged
        assert koala_metadata.shape == koala_image.shape


# Integration and Edge Cases
class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases."""

    def test_full_pipeline_integration(self):
        """Test complete pipeline: metadata → access → image."""
        bbox = BoundingBoxTransformer(bounds_x=(10, 90), bounds_y=(20, 180))

        original_metadata = ImageMetadata("image", (200, 100, 3), (0, 255), np.uint8, dimensions=(Y, X, RGB))
        original_accessor = ImageAccessor(x=(0, 100), y=(0, 200))

        # Step 1: Transform metadata
        new_metadata = bbox.transform_metadata(original_metadata)
        assert new_metadata.shape == (160, 80, 3)
        assert original_metadata.shape == (200, 100, 3)  # Not mutated

        # Step 2: Transform accessor
        new_accessor = bbox.transform_access(original_accessor, original_metadata)
        assert id(bbox) in new_accessor.history
        assert new_accessor is not original_accessor

        # Step 3: Transform image (no padding case)
        image = np.random.randint(0, 256, size=(160, 80, 3), dtype=np.uint8)
        new_accessor.fill_value = None
        new_accessor.history[id(bbox)] = [(0, 0), (0, 0)]

        result = bbox.transform_image(image, new_metadata, new_accessor)
        np.testing.assert_array_equal(result, image)

    def test_padding_with_different_fill_values(self):
        """Test padding with various fill values."""
        bbox = BoundingBoxTransformer(bounds_x=(10, 90), bounds_y=(20, 180))

        metadata = ImageMetadata("image", (150, 70, 3), (0, 255), np.uint8, dimensions=(Y, X, RGB))

        image = np.ones((150, 70, 3), dtype=np.uint8) * 100

        # Test different fill values
        for fill_value in [0, 128, 255]:
            accessor = ImageAccessor()
            accessor.fill_value = fill_value
            accessor.history[id(bbox)] = [(5, 5), (10, 10)]

            result = bbox.transform_image(image, metadata, accessor)

            # Check padding has correct fill value
            assert np.all(result[:5, :, :] == fill_value)
            assert np.all(result[-5:, :, :] == fill_value)
            assert np.all(result[:, :10, :] == fill_value)
            assert np.all(result[:, -10:, :] == fill_value)

            # Original region unchanged
            np.testing.assert_array_equal(result[5:-5, 10:-10, :], image)

    def test_3d_image_cropping(self):
        """Test cropping 3D images (with Z dimension)."""
        bbox = BoundingBoxTransformer(bounds_x=(10, 90), bounds_y=(20, 180), bounds_z=(5, 25))

        metadata = ImageMetadata("image", (30, 200, 100, 3), (0, 255), np.uint8, dimensions=(Z, Y, X, RGB))

        # Transform metadata
        new_metadata = bbox.transform_metadata(metadata)

        # Z: 5 to 25 = 20
        # Y: 20 to 180 = 160
        # X: 10 to 90 = 80
        # RGB: unchanged = 3
        assert new_metadata.shape == (20, 160, 80, 3)

        # Original not mutated
        assert metadata.shape == (30, 200, 100, 3)

    def test_single_pixel_bounds(self):
        """Test with single pixel bounds."""
        bbox = BoundingBoxTransformer(bounds_x=50, bounds_y=100)

        metadata = ImageMetadata("image", (200, 100), (0, 255), np.uint8, dimensions=(Y, X))

        with pytest.raises(TypeError):
            bbox.transform_metadata(metadata)

    def test_metadata_preservation(self):
        """Test that additional metadata is preserved."""
        bbox = BoundingBoxTransformer(bounds_x=(10, 90), bounds_y=(20, 180))

        original = ImageMetadata(
            "image",
            (200, 100, 3),
            (0, 255),
            np.uint8,
            dimensions=(Y, X, RGB),
            additional_metadata={"source": "test", "version": 1},
        )

        result = bbox.transform_metadata(original)

        # Additional metadata preserved
        assert result.additional_metadata == {"source": "test", "version": 1}

        # Original not mutated
        assert original.additional_metadata == {"source": "test", "version": 1}


# Parametrized tests for compact coverage
@pytest.mark.parametrize(
    "bounds_x,bounds_y,image_shape,expected_shape",
    [
        ((0, 50), (0, 100), (200, 100, 3), (100, 50, 3)),
        ((25, 75), (50, 150), (200, 100, 3), (100, 50, 3)),
        ((10, 90), (20, 180), (200, 100, 3), (160, 80, 3)),
    ],
)
def test_various_bounds(bounds_x, bounds_y, image_shape, expected_shape):
    """Test various bounding box configurations."""
    bbox = BoundingBoxTransformer(bounds_x=bounds_x, bounds_y=bounds_y)

    metadata = ImageMetadata("image", image_shape, (0, 255), np.uint8, dimensions=(Y, X, RGB))

    result = bbox.transform_metadata(metadata)

    assert result.shape == expected_shape
    assert metadata.shape == image_shape  # Original unchanged
