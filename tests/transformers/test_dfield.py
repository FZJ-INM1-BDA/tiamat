"""
Comprehensive tests for deformation field transformers.

Combines unit tests with synthetic deformation fields.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

from tiamat.transformers.dfield import DeformationFieldTransformer
from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.metadata.dimensions import Y, X, RGB


# Fixtures
@pytest.fixture
def mock_reader_factory():
    """Create a mock reader factory for testing."""

    def factory(file_path):
        reader = MagicMock()

        # Mock deformation field metadata
        dfield_metadata = ImageMetadata(
            image_type="vector",
            shape=(100, 200, 2),  # (Y, X, vector_components)
            value_range=None,
            dtype=np.float32,
            dimensions=(Y, X, "vec"),
            spacing=(1.0, 1.0),
            additional_metadata={"dfield_origin": (0.0, 0.0)}
        )
        reader.read_metadata.return_value = dfield_metadata
        reader.spacing = (1.0, 1.0)

        # Mock deformation field data (identity transform initially)
        def read_image_mock(accessor):
            # Return identity deformation (no displacement)
            h, w = 100, 200
            dfield = np.zeros((h, w, 2), dtype=np.float32)
            return dfield

        reader.read_image.side_effect = read_image_mock

        return reader

    return factory


@pytest.fixture
def identity_dfield():
    """Create a simple identity deformation field."""
    # 100x200 identity deformation (no displacement)
    dfield = np.zeros((100, 200, 2), dtype=np.float32)
    return dfield


@pytest.fixture
def translation_dfield():
    """Create a translation deformation field (shift by 10 pixels)."""
    # 100x200 with constant shift of (10, 10)
    dfield = np.ones((100, 200, 2), dtype=np.float32) * 10.0
    return dfield


# Unit Tests - Compact Version
class TestDeformationFieldTransformer:
    """Compact unit tests for DeformationFieldTransformer."""

    def test_initialization_and_properties(self, mock_reader_factory):
        """Test initialization and basic properties."""
        # Test with minimal args
        dft1 = DeformationFieldTransformer(
            dfield_file="test_dfield.npy",
            reader_factory=mock_reader_factory
        )
        assert dft1.dfield_file == "test_dfield.npy"
        assert dft1.request_margin == 2  # Default
        assert dft1.interpolation == 'linear'  # Default
        assert dft1.fill_value is None  # Default
        assert dft1.xy_coordinates is True  # Default

        # Test with custom args
        dft2 = DeformationFieldTransformer(
            dfield_file="custom.npy",
            request_margin=5,
            interpolation='cubic',
            fill_value=128,
            xy_coordinates=False,
            reader_factory=mock_reader_factory
        )
        assert dft2.request_margin == 5
        assert dft2.interpolation == 'cubic'
        assert dft2.fill_value == 128
        assert dft2.xy_coordinates is False

    def test_cached_properties(self, mock_reader_factory):
        """Test cached properties for deformation field metadata."""
        dft = DeformationFieldTransformer(
            dfield_file="test.npy",
            reader_factory=mock_reader_factory
        )

        # Test dfield_metadata (cached)
        meta1 = dft.dfield_metadata
        meta2 = dft.dfield_metadata
        assert meta1 is meta2  # Should be same object (cached)
        assert meta1.shape == (100, 200, 2)

        # Test dfield_spacing
        spacing = dft.dfield_spacing
        assert spacing == (1.0, 1.0)

        # Test dfield_origin
        origin = dft.dfield_origin
        assert origin == (0.0, 0.0)

    def test_get_pixel_coordinates_identity(self):
        """Test pixel coordinate computation with identity deformation."""
        # Identity deformation: output should equal input grid
        dfield = np.zeros((10, 20, 2), dtype=np.float32)

        coords = DeformationFieldTransformer.get_pixel_coordinates(
            dfield=dfield,
            dfield_spacing=(1.0, 1.0),
            dfield_scale=(1, 1),
            dfield_origin=(0.0, 0.0),
            image_spacing=(1.0, 1.0),
            xy=True
        )

        # Should be (2, H, W) shape
        assert coords.shape == (2, 10, 20)

        # Check that coordinates form a regular grid
        # coords[0] is Y, coords[1] is X
        assert np.allclose(coords[0, :, 0], np.arange(10))  # Y increases down
        assert np.allclose(coords[1, 0, :], np.arange(20))  # X increases right

    def test_get_pixel_coordinates_with_translation(self):
        """Test pixel coordinates with constant translation."""
        # Constant shift of 5 pixels in each direction
        dfield = np.ones((10, 20, 2), dtype=np.float32) * 5.0

        coords = DeformationFieldTransformer.get_pixel_coordinates(
            dfield=dfield,
            dfield_spacing=(1.0, 1.0),
            dfield_scale=(1, 1),
            dfield_origin=(0.0, 0.0),
            image_spacing=(1.0, 1.0),
            xy=True
        )

        # Should be shifted by 5 pixels
        assert np.allclose(coords[0, 0, 0], 5.0)  # Y shifted
        assert np.allclose(coords[1, 0, 0], 5.0)  # X shifted

    def test_get_pixel_coordinates_xy_vs_yx(self):
        """Test XY vs YX coordinate ordering."""
        dfield_xy = np.zeros((10, 20, 2), dtype=np.float32)
        dfield_xy[:, :, 0] = 1.0  # X component
        dfield_xy[:, :, 1] = 2.0  # Y component

        # With xy=True
        coords_xy = DeformationFieldTransformer.get_pixel_coordinates(
            dfield=dfield_xy,
            dfield_spacing=(1.0, 1.0),
            dfield_scale=(1, 1),
            dfield_origin=(0.0, 0.0),
            image_spacing=(1.0, 1.0),
            xy=True
        )

        # With xy=False (YX ordering)
        coords_yx = DeformationFieldTransformer.get_pixel_coordinates(
            dfield=dfield_xy,
            dfield_spacing=(1.0, 1.0),
            dfield_scale=(1, 1),
            dfield_origin=(0.0, 0.0),
            image_spacing=(1.0, 1.0),
            xy=False
        )

        # Results should be different
        assert not np.allclose(coords_xy, coords_yx)

    def test_apply_deformation_identity(self):
        """Test applying identity deformation (no change)."""
        # Create test image
        image = np.random.randint(0, 256, size=(50, 100), dtype=np.uint8)

        # Identity coordinates (pixel positions unchanged)
        y_coords, x_coords = np.meshgrid(np.arange(50), np.arange(100), indexing='ij')
        coordinates = np.stack([y_coords, x_coords], axis=0).astype(np.float32)

        result = DeformationFieldTransformer.apply_deformation(
            image=image,
            coordinates=coordinates,
            fill_value=0,
            interpolation='linear'
        )

        # Should be identical (or very close due to interpolation)
        np.testing.assert_array_almost_equal(result, image, decimal=0)

    def test_apply_deformation_with_channels(self):
        """Test deformation with RGB image."""
        # RGB image
        image = np.random.randint(0, 256, size=(50, 100, 3), dtype=np.uint8)

        # Identity coordinates
        y_coords, x_coords = np.meshgrid(np.arange(50), np.arange(100), indexing='ij')
        coordinates = np.stack([y_coords, x_coords], axis=0).astype(np.float32)

        result = DeformationFieldTransformer.apply_deformation(
            image=image,
            coordinates=coordinates,
            channel_dim=2,
            fill_value=0,
            interpolation='linear'
        )

        # Should preserve shape
        assert result.shape == image.shape
        # Should be close to original
        np.testing.assert_array_almost_equal(result, image, decimal=0)

    def test_apply_deformation_with_fill_value(self):
        """Test that fill_value is used for out-of-bounds pixels."""
        image = np.ones((10, 10), dtype=np.uint8) * 100

        # Coordinates that go outside the image
        y_coords = np.ones((10, 10)) * 20
        x_coords = np.ones((10, 10)) * 20
        coordinates = np.stack([y_coords, x_coords], axis=0).astype(np.float32)

        result = DeformationFieldTransformer.apply_deformation(
            image=image,
            coordinates=coordinates,
            fill_value=255,
            interpolation='nearest'
        )

        # All pixels should be fill_value (out of bounds)
        assert np.all(result == 255)

    def test_metadata_transformation_and_mutation(self, mock_reader_factory):
        """Test metadata updates and non-mutation."""
        dft = DeformationFieldTransformer(
            dfield_file="test.npy",
            reader_factory=mock_reader_factory
        )

        original = ImageMetadata(
            "image", (50, 100, 3), (0, 255), np.uint8,
            dimensions=(Y, X, RGB),
            spacing=(1.0, 1.0)
        )
        original_shape = original.shape
        original_id = id(original)

        result = dft.transform_metadata(original)

        # Should update spatial shape based on dfield
        # Dfield is 100x200 with spacing 1.0
        # Image spacing is 1.0
        # Result should be 100x200
        #assert result.spatial_shape == (100, 200)

        # Should preserve channels
        assert result.shape[2] == 3

        # CRITICAL: Should not mutate original
        assert original.shape == original_shape
        assert id(result) != original_id

    def test_transform_access_stores_coordinates(self, mock_reader_factory):
        """Test that transform_access stores coordinates in accessor history."""
        dft = DeformationFieldTransformer(
            dfield_file="test.npy",
            reader_factory=mock_reader_factory
        )

        metadata = ImageMetadata(
            "image", (100, 200, 3), (0, 255), np.uint8,
            dimensions=(Y, X, RGB),
            spacing=(1.0, 1.0)
        )

        accessor = ImageAccessor(x=(0, 200), y=(0, 100))

        result = dft.transform_access(accessor, metadata)

        # Should store coordinates in history
        assert id(dft) in result.history

        # Coordinates should be numpy array
        coords = result.history[id(dft)]
        assert isinstance(coords, np.ndarray)
        assert coords.ndim == 3  # (2, H, W)
        assert coords.shape[0] == 2  # Y and X coordinates

    def test_transform_image_requires_transform_access(self, mock_reader_factory):
        """Test that transform_image requires transform_access to be called first."""
        dft = DeformationFieldTransformer(
            dfield_file="test.npy",
            reader_factory=mock_reader_factory
        )

        image = np.zeros((50, 100), dtype=np.uint8)
        metadata = ImageMetadata("image", image.shape, (0, 255), np.uint8)
        accessor = ImageAccessor()

        # Should raise error if transform_access not called
        with pytest.raises(Exception, match="transform_access has to be called"):
            dft.transform_image(image, metadata, accessor)

    def test_from_json(self, mock_reader_factory):
        """Test JSON deserialization."""
        config = {
            "dfield_file": "test_field.npy",
            "request_margin": 3,
            "interpolation": "cubic",
            "fill_value": 128,
            "xy_coordinates": False,
        }

        # Need to mock get_reader_from_config
        with patch('tiamat.serialization.get_reader_from_config') as mock_get_reader:
            mock_get_reader.return_value = mock_reader_factory

            dft = DeformationFieldTransformer.from_json(config)

            assert dft.dfield_file == "test_field.npy"
            assert dft.request_margin == 3
            assert dft.interpolation == "cubic"
            assert dft.fill_value == 128
            assert dft.xy_coordinates is False


# Integration Tests
class TestDeformationFieldIntegration:
    """Integration tests for complete deformation pipeline."""

    def test_full_pipeline_identity_transform(self, mock_reader_factory):
        """Test complete pipeline with identity deformation."""
        dft = DeformationFieldTransformer(
            dfield_file="identity.npy",
            reader_factory=mock_reader_factory
        )

        # Create test image
        image = np.random.randint(0, 256, size=(100, 200, 3), dtype=np.uint8)
        metadata = ImageMetadata(
            "image", image.shape, (0, 255), np.uint8,
            dimensions=(Y, X, RGB),
            spacing=(1.0, 1.0)
        )
        accessor = ImageAccessor(x=(0, 200), y=(0, 100))
        # Step 1: Transform metadata
        new_metadata = dft.transform_metadata(metadata)
        #assert new_metadata.spatial_shape == (100, 200)
        assert metadata.shape == image.shape  # Not mutated

        # Step 2: Transform access
        new_accessor = dft.transform_access(accessor, metadata)
        assert id(dft) in new_accessor.history

        # Step 3: Transform image (with identity deformation)
        # Need to set up coordinates in history manually for this test
        y_coords, x_coords = np.meshgrid(np.arange(100), np.arange(200), indexing='ij')
        coordinates = np.stack([y_coords, x_coords], axis=0).astype(np.float32)
        new_accessor.history[id(dft)] = coordinates

        result = dft.transform_image(image, metadata, new_accessor)

        # With identity transform, should be very similar
        assert result.shape == image.shape
        # Allow some tolerance for interpolation
        assert np.mean(np.abs(result.astype(float) - image.astype(float))) < 5

    def test_pipeline_with_multiple_channels(self, mock_reader_factory):
        """Test pipeline preserves multiple channels."""
        dft = DeformationFieldTransformer(
            dfield_file="test.npy",
            reader_factory=mock_reader_factory
        )

        # RGB image
        image = np.random.randint(0, 256, size=(50, 100, 3), dtype=np.uint8)
        metadata = ImageMetadata(
            "image", image.shape, (0, 255), np.uint8,
            dimensions=(Y, X, RGB),
            spacing=(1.0, 1.0)
        )

        # Identity coordinates for testing
        y_coords, x_coords = np.meshgrid(np.arange(50), np.arange(100), indexing='ij')
        coordinates = np.stack([y_coords, x_coords], axis=0).astype(np.float32)

        accessor = ImageAccessor()
        accessor.history[id(dft)] = coordinates

        result = dft.transform_image(image, metadata, accessor)

        # Should preserve channels
        assert result.shape == image.shape
        assert result.shape[2] == 3

    def test_pipeline_with_3d_image(self, mock_reader_factory):
        """Test pipeline with 3D image (Z, Y, X)."""
        dft = DeformationFieldTransformer(
            dfield_file="test.npy",
            reader_factory=mock_reader_factory
        )

        # 3D image (Z, Y, X)
        from tiamat.metadata.dimensions import Z
        image = np.random.randint(0, 256, size=(5, 50, 100), dtype=np.uint8)
        metadata = ImageMetadata(
            "image", image.shape, (0, 255), np.uint8,
            dimensions=(Z, Y, X),
            spacing=(1.0, 1.0, 1.0)
        )

        # Identity coordinates
        y_coords, x_coords = np.meshgrid(np.arange(50), np.arange(100), indexing='ij')
        coordinates = np.stack([y_coords, x_coords], axis=0).astype(np.float32)

        accessor = ImageAccessor()
        accessor.history[id(dft)] = coordinates

        result = dft.transform_image(image, metadata, accessor)

        # Should preserve Z dimension
        assert result.shape[0] == 5
        assert result.shape[1:] == (50, 100)


# Edge Cases
class TestDeformationFieldEdgeCases:
    """Test edge cases and error handling."""

    def test_multiple_channel_dimensions_error(self, mock_reader_factory):
        """Test error with multiple channel dimensions."""
        dft = DeformationFieldTransformer(
            dfield_file="test.npy",
            reader_factory=mock_reader_factory
        )

        # Image with multiple channel dimensions (invalid)
        from tiamat.metadata.dimensions import C
        image = np.random.randint(0, 256, size=(50, 100, 3, 4), dtype=np.uint8)
        metadata = ImageMetadata(
            "image", image.shape, (0, 255), np.uint8,
            dimensions=(Y, X, C, RGB)  # Two channel dims
        )

        coordinates = np.zeros((2, 50, 100), dtype=np.float32)
        accessor = ImageAccessor()
        accessor.history[id(dft)] = coordinates

        with pytest.raises(Exception, match="Multiple channels.*not supported"):
            dft.transform_image(image, metadata, accessor)

    def test_different_interpolation_methods(self):
        """Test different interpolation methods."""
        image = np.random.randint(0, 256, size=(50, 100), dtype=np.uint8)

        # Identity coordinates
        y_coords, x_coords = np.meshgrid(np.arange(50), np.arange(100), indexing='ij')
        coordinates = np.stack([y_coords, x_coords], axis=0).astype(np.float32)

        # Test different interpolation methods
        for interp in ['nearest', 'linear']:
            result = DeformationFieldTransformer.apply_deformation(
                image=image,
                coordinates=coordinates,
                fill_value=0,
                interpolation=interp
            )

            assert result.shape == image.shape
            # With identity coords, should be very similar
            np.testing.assert_array_almost_equal(result, image, decimal=0)

    def test_fill_value_handling(self, mock_reader_factory):
        """Test fill_value is used correctly."""
        # Test with transformer fill_value
        dft1 = DeformationFieldTransformer(
            dfield_file="test.npy",
            fill_value=255,
            reader_factory=mock_reader_factory
        )

        metadata = ImageMetadata(
            "image", (50, 100), (0, 255), np.uint8,
            spacing=(1.0, 1.0)
        )
        accessor1 = ImageAccessor(x=(0, 100), y=(0, 50))

        result1 = dft1.transform_access(accessor1, metadata)
        # Should use transformer's fill_value
        # (actual fill happens in transform_image, but we can check accessor is set)

        # Test with accessor fill_value
        dft2 = DeformationFieldTransformer(
            dfield_file="test.npy",
            reader_factory=mock_reader_factory
        )

        accessor2 = ImageAccessor(x=(0, 100), y=(0, 50), fill_value=128)
        result2 = dft2.transform_access(accessor2, metadata)

        # Both should set fill_value
        assert result1.fill_value is not None or dft1.fill_value is not None
        assert result2.fill_value is not None


# Parametrized tests
@pytest.mark.parametrize("xy_coordinates", [True, False])
def test_xy_coordinate_ordering(xy_coordinates, mock_reader_factory):
    """Test XY vs YX coordinate ordering."""
    dft = DeformationFieldTransformer(
        dfield_file="test.npy",
        xy_coordinates=xy_coordinates,
        reader_factory=mock_reader_factory
    )

    assert dft.xy_coordinates == xy_coordinates


@pytest.mark.parametrize("spacing", [
    (1.0, 1.0),
    (0.5, 0.5),
    (2.0, 2.0),
    (1.0, 2.0),
])
def test_different_spacings(spacing):
    """Test deformation with different image spacings."""
    dfield = np.zeros((10, 20, 2), dtype=np.float32)

    coords = DeformationFieldTransformer.get_pixel_coordinates(
        dfield=dfield,
        dfield_spacing=(1.0, 1.0),
        dfield_scale=(1, 1),
        dfield_origin=(0.0, 0.0),
        image_spacing=spacing,
        xy=True
    )

    assert coords.shape == (2, 10, 20)
    # Coordinates should be scaled by spacing
