"""
Comprehensive tests for axes transformers.

Combines unit tests with real image validation using Koala.jpg.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.metadata.dimensions import RGB, C, X, Y, Z
from tiamat.transformers.axes import (
    ImageToVolumeTransformer,
    MirrorTransformer,
    ReorderCoordinatesTransformer,
)


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
        spacing=(1.0, 1.0),
        scales=(1.0, 1.0),
    )


# Unit Tests - ImageToVolumeTransformer
class TestImageToVolumeTransformer:
    """Compact unit tests for ImageToVolumeTransformer."""

    def test_initialization_and_basic_functionality(self):
        """Test initialization with and without z_spacing."""
        # Default (None)
        t1 = ImageToVolumeTransformer()
        assert t1.z_spacing is None

        # With custom z_spacing
        t2 = ImageToVolumeTransformer(z_spacing=2.0)
        assert t2.z_spacing == 2.0

    def test_metadata_transformation_and_mutation(self):
        """Test metadata adds Z dimension and doesn't mutate original."""
        transformer = ImageToVolumeTransformer(z_spacing=1.5)

        original = ImageMetadata(
            "image",
            (100, 200, 3),
            (0, 255),
            np.uint8,
            dimensions=(Y, X, RGB),
            spacing=(1.0, 1.0),
            scales=(1.0, 1.0),
        )
        original_shape = original.shape
        original_id = id(original)

        result = transformer.transform_metadata(original)

        # Should add Z dimension
        # Original: (Y=100, X=200, RGB=3)
        # Result: (Z=1, Y=100, X=200, RGB=3)
        assert result.shape == (1, 100, 200, 3)
        assert Z in result.dimensions
        assert result.dimensions.index(Z) < result.dimensions.index(Y)

        # Should update spacing
        assert result.spacing == (1.0, 1.0, 1.5)

        # Should update scales (add 1.0 for Z)
        assert result.scales[0] == (1.0, 1.0, 1.0)

        # Should add stack_dimension metadata
        assert result.additional_metadata["stack_dimension"] == Z

        # CRITICAL: Should not mutate original
        assert original.shape == original_shape
        assert id(result) != original_id

    def test_transform_access_removes_z_info(self):
        """Test that transform_access removes Z-related accessor info."""
        transformer = ImageToVolumeTransformer()

        metadata = ImageMetadata(
            "image",
            (100, 200),
            (0, 255),
            np.uint8,
            dimensions=(Y, X),
            spacing=(1.0, 1.0),
        )

        # Accessor with Z info (shouldn't have for 2D image)
        accessor = ImageAccessor(
            x=(0, 200),
            y=(0, 100),
            z=(0, 1),
            scale=(1.0, 1.0, 1.0),
            spacing=(1.0, 1.0, 1.0),
        )

        result = transformer.transform_access(accessor, metadata)

        # Should remove z
        assert result.z is None

        # Should truncate scale and spacing to 2D
        assert len(result.scale) == 2
        assert len(result.spacing) == 2

        # Should not be same object
        assert result is not accessor

    def test_transform_image_adds_z_dimension(self):
        """Test that image gets Z dimension added."""
        transformer = ImageToVolumeTransformer()

        # 2D image (Y, X, RGB)
        image = np.random.randint(0, 256, size=(100, 200, 3), dtype=np.uint8)
        metadata = ImageMetadata(
            "image",
            image.shape,
            (0, 255),
            np.uint8,
            dimensions=(Y, X, RGB),
        )

        result = transformer.transform_image(image, metadata, ImageAccessor())

        # Should add singleton Z dimension
        # (Y=100, X=200, RGB=3) -> (Z=1, Y=100, X=200, RGB=3)
        assert result.shape == (1, 100, 200, 3)

        # Content should be unchanged
        np.testing.assert_array_equal(result[0], image)

    def test_from_json(self):
        """Test JSON deserialization."""
        config = {"z_spacing": 2.5}
        transformer = ImageToVolumeTransformer.from_json(config)

        assert transformer.z_spacing == 2.5

    def test_error_on_missing_z_spacing_with_nonuniform_spacing(self):
        """Test that missing z_spacing with non-uniform spacing raises error."""
        transformer = ImageToVolumeTransformer()  # No z_spacing

        metadata = ImageMetadata(
            "image",
            (100, 200),
            (0, 255),
            np.uint8,
            dimensions=(Y, X),
            spacing=(1.0, 2.0),  # Non-uniform
        )

        # with pytest.raises(Exception, match="Need provide z_spacing"):
        with pytest.raises(TypeError):
            transformer.transform_metadata(metadata)


# Unit Tests - ReorderCoordinatesTransformer
class TestReorderCoordinatesTransformer:
    """Compact unit tests for ReorderCoordinatesTransformer."""

    def test_initialization_2d_and_3d(self):
        """Test initialization for 2D and 3D axes."""
        # 2D
        t2d = ReorderCoordinatesTransformer(axes=("x", "y"))
        assert t2d.reorder_axes == ("x", "y")
        assert len(t2d.from_indices) == 2

        # 3D
        t3d = ReorderCoordinatesTransformer(axes=("x", "y", "z"))
        assert t3d.reorder_axes == ("x", "y", "z")
        assert len(t3d.from_indices) == 3

        # Custom order
        t_custom = ReorderCoordinatesTransformer(axes=("y", "x", "z"))
        assert t_custom.reorder_axes == ("y", "x", "z")

    @pytest.mark.parametrize(
        "axes, input_dims, expected_dims, expected_shape",
        [
            # Case 1: Swap 2D (Y, X) -> (X, Y)
            (("y", "x"), (X, Y, RGB), (Y, X, RGB), (200, 100, 3)),
            # Case 2: Reorder 3D (Z, Y, X) -> (X, Y, Z)
            (("z", "y", "x"), (X, Y, Z, RGB), (Z, Y, X, RGB), (200, 100, 10, 3)),
        ],
    )
    def test_reorder_full_flow(self, axes, input_dims, expected_dims, expected_shape):
        """Tests metadata and image transformation for reordering."""
        transformer = ReorderCoordinatesTransformer(axes=axes)
        # Setup inputs (Base shape: Z=10, Y=100, X=200)
        base_shape = {Z: 10, Y: 100, X: 200, RGB: 3}
        shape = tuple(base_shape[d] for d in input_dims)

        meta = ImageMetadata(
            "image",
            shape,
            (0, 255),
            np.uint8,
            dimensions=input_dims,
            spacing=(1,) * len(shape),
            scales=(1,) * len(shape),
        )

        # 1. Test Metadata
        res_meta = transformer.transform_metadata(meta)
        print(f" {res_meta.shape} == {expected_shape}")
        assert res_meta.shape == expected_shape

        # 2. Test Image Transpose
        image = np.zeros(shape, dtype=np.uint8)
        res_img = transformer.transform_image(image, meta, ImageAccessor())
        print(f" {res_img.shape} == {expected_shape}")
        assert res_img.shape == expected_shape

    def test_transform_access_reorders_coordinates(self):
        """Test that accessor coordinates are reordered."""
        # Swap X and Y
        transformer = ReorderCoordinatesTransformer(axes=("x", "y"))

        metadata = ImageMetadata(
            "image",
            (100, 200),
            (0, 255),
            np.uint8,
            dimensions=(Y, X),
            spacing=(1.0, 2.0),
        )

        accessor = ImageAccessor(
            x=(10, 50),
            y=(20, 80),
            scale=(1.0, 2.0),
        )

        result = transformer.transform_access(accessor, metadata)

        # X and Y should be swapped
        assert result.x == (20, 80)  # Was Y
        assert result.y == (10, 50)  # Was X

        # Scale should be reordered
        assert result.scale == (2.0, 1.0)

    def test_from_json(self):
        """Test JSON deserialization."""
        config = {"axes": ("y", "x", "z")}
        transformer = ReorderCoordinatesTransformer.from_json(config)

        assert transformer.reorder_axes == ("y", "x", "z")


# Unit Tests - MirrorTransformer
class TestMirrorTransformer:
    """Compact unit tests for MirrorTransformer."""

    def test_initialization(self):
        """Test initialization with different mirror flags."""
        # No mirroring
        t1 = MirrorTransformer()
        assert not t1.mirror_x
        assert not t1.mirror_y
        assert not t1.mirror_z

        # Mirror X only
        t2 = MirrorTransformer(mirror_x=True)
        assert t2.mirror_x
        assert not t2.mirror_y

        # Mirror all
        t3 = MirrorTransformer(mirror_x=True, mirror_y=True, mirror_z=True)
        assert t3.mirror_x and t3.mirror_y and t3.mirror_z

    def test_metadata_unchanged(self):
        """Test that metadata is not changed by mirroring."""
        transformer = MirrorTransformer(mirror_x=True, mirror_y=True)

        original = ImageMetadata(
            "image",
            (100, 200, 3),
            (0, 255),
            np.uint8,
            dimensions=(Y, X, RGB),
        )

        result = transformer.transform_metadata(original)

        # Should be unchanged (mirroring doesn't affect metadata)
        assert result is original

        @pytest.mark.parametrize(
            "settings, input_shape, axis_to_check, expected_val",
            [
                # 1. Mirror X only: Pixel at 0 moves to end
                ({"mirror_x": True}, (10, 10, 3), 1, 9),
                # 2. Mirror Y only: Pixel at 0 moves to end
                ({"mirror_y": True}, (10, 10, 3), 0, 9),
                # 3. Mirror Both: Pixel at (0,0) moves to (9,9)
                ({"mirror_x": True, "mirror_y": True}, (10, 10, 3), (0, 1), (9, 9)),
            ],
        )
        def test_mirror_logic(self, settings, input_shape, axis_to_check, expected_val):
            """
            Tests initialization, accessor transform, AND image transform in one go.
            """
            transformer = MirrorTransformer(**settings)
            meta = ImageMetadata("image", input_shape, (0, 255), np.uint8, dimensions=(Y, X, RGB))

            # 1. Test Accessor Transform
            # We use a specific slice (0, 1) to see if it flips to the end of the image
            accessor = ImageAccessor(x=(0, 1), y=(0, 1))
            acc_res = transformer.transform_access(accessor, meta)

            # Check X coordinate
            if settings.get("mirror_x"):
                # If image is width 10, requesting (0,1) becomes (9, 10)
                assert acc_res.x == (9, 10)
            else:
                assert acc_res.x == (0, 1)

            # 2. Test Image Transform
            image = np.zeros(input_shape, dtype=np.uint8)
            # Set a specific pixel to 255 to track it
            image[0, 0] = 255

            img_res = transformer.transform_image(image, meta, ImageAccessor())

            # Check where the pixel moved
            if settings.get("mirror_x") and settings.get("mirror_y"):
                assert img_res[9, 9, 0] == 255
            elif settings.get("mirror_x"):
                assert img_res[0, 9, 0] == 255
            elif settings.get("mirror_y"):
                assert img_res[9, 0, 0] == 255

    def test_mirror_3d_image(self):
        """Test mirroring 3D image with Z axis."""
        transformer = MirrorTransformer(mirror_z=True)

        # Simple 3D image (Z, Y, X)
        image = np.array(
            [
                [[1, 2], [3, 4]],  # Z=0
                [[5, 6], [7, 8]],  # Z=1
            ],
            dtype=np.uint8,
        )

        metadata = ImageMetadata("image", image.shape, (0, 255), np.uint8, dimensions=(Z, Y, X))

        result = transformer.transform_image(image, metadata, ImageAccessor())

        # Should flip Z axis
        # Z=0 and Z=1 should be swapped
        np.testing.assert_array_equal(result[0], image[1])
        np.testing.assert_array_equal(result[1], image[0])

    def test_from_json(self):
        """Test JSON deserialization."""
        config = {"mirror_x": True, "mirror_y": False, "mirror_z": True}
        transformer = MirrorTransformer.from_json(config)

        assert transformer.mirror_x is True
        assert transformer.mirror_y is False
        assert transformer.mirror_z is True

    def test_transform_access_handles_none_roi(self):
        """Test that mirroring works when user requests full image (None)."""
        transformer = MirrorTransformer(mirror_x=True)
        # 100px wide image
        metadata = ImageMetadata("image", (100, 100, 3), (0, 255), np.uint8)

        # Request full image (x=None)
        accessor = ImageAccessor(x=None, y=(10, 20))

        result = transformer.transform_access(accessor, metadata)

        # The transformer should strictly return None (preserve 'full image' semantic)
        # OR resolve it to (0, 100). Both are valid, but it MUST NOT CRASH.
        assert result.x is None or result.x == (0, 100)


# Real Image Tests with Koala
class TestWithKoalaImage:
    """Test axes transformers with real Koala.jpg image."""

    def test_image_to_volume_with_koala(self, koala_image, koala_metadata):
        """Test converting 2D koala to 3D volume."""
        transformer = ImageToVolumeTransformer(z_spacing=1.0)
        # Transform metadata
        new_metadata = transformer.transform_metadata(koala_metadata)
        assert new_metadata.shape == (1, *koala_image.shape)
        assert Z in new_metadata.dimensions

        # Transform image
        result = transformer.transform_image(koala_image, koala_metadata, ImageAccessor())

        # Should add Z dimension
        assert result.shape == (1, *koala_image.shape)

        # Content should be unchanged
        np.testing.assert_array_equal(result[0], koala_image)

    def test_reorder_koala_axes(self, koala_image, koala_metadata):
        """Test reordering koala image axes."""
        # Swap X and Y
        transformer = ReorderCoordinatesTransformer(axes=("x", "y"))
        # Transform metadata
        new_metadata2 = transformer.transform_metadata(koala_metadata)
        h, w = koala_image.shape[:2]

        # Transform image
        result = transformer.transform_image(koala_image, koala_metadata, ImageAccessor())

        # Dimensions should be swapped
        assert result.shape == (w, h, 3)

        # Verify actual transpose: check a specific pixel
        np.testing.assert_array_equal(result[100, 50, :], koala_image[50, 100, :])

    def test_mirror_koala_horizontally(self, koala_image, koala_metadata):
        """Test horizontal mirroring of koala."""
        transformer = MirrorTransformer(mirror_x=True)

        result = transformer.transform_image(koala_image, koala_metadata, ImageAccessor())

        # Should be flipped horizontally
        assert result.shape == koala_image.shape

        # Left edge should match original right edge
        np.testing.assert_array_equal(result[:, 0, :], koala_image[:, -1, :])
        np.testing.assert_array_equal(result[:, -1, :], koala_image[:, 0, :])

    def test_mirror_koala_vertically(self, koala_image, koala_metadata):
        """Test vertical mirroring of koala."""
        transformer = MirrorTransformer(mirror_y=True)

        result = transformer.transform_image(koala_image, koala_metadata, ImageAccessor())

        # Should be flipped vertically
        assert result.shape == koala_image.shape

        # Top edge should match original bottom edge
        np.testing.assert_array_equal(result[0, :, :], koala_image[-1, :, :])
        np.testing.assert_array_equal(result[-1, :, :], koala_image[0, :, :])

    def test_mirror_koala_both_axes(self, koala_image, koala_metadata):
        """Test mirroring koala both horizontally and vertically (180° rotation)."""
        transformer = MirrorTransformer(mirror_x=True, mirror_y=True)

        result = transformer.transform_image(koala_image, koala_metadata, ImageAccessor())

        # Should be rotated 180°
        assert result.shape == koala_image.shape

        # Top-left should match original bottom-right
        np.testing.assert_array_equal(result[0, 0, :], koala_image[-1, -1, :])
        np.testing.assert_array_equal(result[-1, -1, :], koala_image[0, 0, :])
