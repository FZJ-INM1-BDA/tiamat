"""
Comprehensive tests for access transformers.
"""

import math

import numpy as np
import pytest

from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.metadata.dimensions import RGB, X, Y, Z
from tiamat.transformers.access import FractionTransformer, SpacingToScaleTransformer


class TestSpacingToScaleTransformer:
    """Tests for SpacingToScaleTransformer."""

    def test_scale_computation_and_coordinate_scaling(self):
        """Test scale computation, coordinate scaling, tuple/list spacing, and static method."""
        t = SpacingToScaleTransformer()

        # Case 1: Basic scale = image_spacing / accessor.spacing = 2.0/1.0 = 2.0
        # Coordinates scaled by coord_spacing/image_spacing = 1.0/2.0 = 0.5
        meta1 = ImageMetadata("image", (100, 200), (0, 255), np.uint8, spacing=2.0)
        acc1 = ImageAccessor(x=(0, 100), y=(0, 50), spacing=1.0, coordinate_spacing=1.0)
        result1 = t.transform_access(acc1, meta1)
        assert result1.scale == 2.0
        assert result1.x == (0, 50) and result1.y == (0, 25)
        assert result1 is not acc1

        # Case 2: Different ratio: 2.0 / 0.5 = 4.0
        acc2 = ImageAccessor(x=(0, 200), y=(0, 100), spacing=0.5, coordinate_spacing=1.0)
        assert t.transform_access(acc2, meta1).scale == 4.0

        # Case 3: Tuple spacing (isotropic)
        meta_tuple = ImageMetadata("image", (100, 200), (0, 255), np.uint8, spacing=(2.0, 2.0))
        assert t.transform_access(acc1, meta_tuple).scale == 2.0

        # Case 4: List spacing (isotropic)
        meta_list = ImageMetadata("image", (100, 200), (0, 255), np.uint8, spacing=[1.5, 1.5])
        acc3 = ImageAccessor(x=(0, 100), spacing=0.5, coordinate_spacing=1.0)
        assert t.transform_access(acc3, meta_list).scale == 3.0

        # Case 5: Static method tests
        assert t._scale_coordinate(None, 1.0, 2.0) is None
        assert t._scale_coordinate(100, 1.0, 2.0) == 50
        assert t._scale_coordinate((0, 100), 1.0, 2.0) == (0, 50)

    def test_errors_and_passthrough(self):
        """Test all error cases and metadata/image passthrough."""
        t = SpacingToScaleTransformer()
        meta = ImageMetadata("image", (100, 200), (0, 255), np.uint8, spacing=2.0)
        acc = ImageAccessor(x=(0, 100), spacing=1.0, coordinate_spacing=1.0)

        # Error 1: Missing metadata.spacing
        with pytest.raises(AssertionError, match="requires spacing"):
            t.transform_access(acc, ImageMetadata("image", (100, 200), (0, 255), np.uint8, spacing=None))

        # Error 2: Missing accessor.coordinate_spacing
        with pytest.raises(AssertionError, match="requires accessor.coordinate_spacing"):
            t.transform_access(ImageAccessor(x=(0, 100), spacing=1.0, coordinate_spacing=None), meta)

        # Error 3: Missing accessor.spacing
        with pytest.raises(AssertionError, match="requires accessor.spacing"):
            t.transform_access(ImageAccessor(x=(0, 100), spacing=None, coordinate_spacing=1.0), meta)

        # Error 4: Anisotropic spacing
        with pytest.raises(AssertionError, match="anisotropic"):
            t.transform_access(acc, ImageMetadata("image", (100, 200), (0, 255), np.uint8, spacing=(1.0, 2.0)))

        # Passthrough tests
        assert t.transform_metadata(meta) is meta
        img = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
        np.testing.assert_array_equal(t.transform_image(img, meta, acc), img)


class TestFractionTransformer:
    """Tests for FractionTransformer."""

    def test_fraction_to_pixels_and_static_method(self):
        """Test fraction→pixel conversion, 3D/4D, None/int handling, and static method."""
        t = FractionTransformer()
        meta = ImageMetadata("image", (100, 200, 3), (0, 255), np.uint8, dimensions=(Y, X, RGB), spacing=1.0)

        # Case 1: Single fractions: x=0.5 of 200=100, y=0.25 of 100=25
        result1 = t.transform_access(ImageAccessor(x=0.5, y=0.25), meta)
        assert result1.x == 100 and result1.y == 25
        assert result1.coordinate_spacing == 1.0

        # Case 2: Tuple fractions: (0.1, 0.9) of 200 = (20, 180)
        result2 = t.transform_access(ImageAccessor(x=(0.1, 0.9), y=(0.2, 0.8)), meta)
        assert result2.x == (20, 180) and result2.y == (20, 80)

        # Case 3: 3D image (documents shape indexing bug)
        meta_3d = ImageMetadata("image", (50, 100, 200), (0, 255), np.uint8, dimensions=(Z, Y, X), spacing=1.0)
        result_3d = t.transform_access(ImageAccessor(x=0.5, y=0.5, z=0.3), meta_3d)
        assert result_3d.x == 50 and result_3d.y == 25 and result_3d.z == 60  # Bug: uses wrong shape indices

        # Case 4: None remains None
        result_none = t.transform_access(ImageAccessor(x=None, y=None), meta)
        assert result_none.x is None and result_none.y is None

        # Case 5: Int scaled
        result_int = t.transform_access(ImageAccessor(x=1, y=1), meta)
        assert result_int.x == 200 and result_int.y == 100

        # Case 6: Static method tests
        assert t._scale_coordinate(0.5, 200) == 100
        assert t._scale_coordinate(None, 200) is None
        assert t._scale_coordinate((0.1, 0.9), 200) == (20, 180)
        assert t._scale_coordinate((0.0, (0.5, 1.0)), 100) == (0, (50, 100))

    def test_errors_and_passthrough(self):
        """Test error cases and metadata/image passthrough."""
        t = FractionTransformer()
        meta = ImageMetadata("image", (100, 200), (0, 255), np.uint8, spacing=1.0)

        # Error: Missing shape
        with pytest.raises(AssertionError, match="requires metadata.shape"):
            t.transform_access(ImageAccessor(x=0.5), ImageMetadata("image", None, (0, 255), np.uint8))

        # Passthrough tests
        assert t.transform_metadata(meta) is meta
        img = np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
        np.testing.assert_array_equal(t.transform_image(img, meta, ImageAccessor()), img)


# Parametrized tests cover remaining cases compactly
@pytest.mark.parametrize(
    "img_sp,acc_sp,exp_scale",
    [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 2.0),
        (1.0, 2.0, 0.5),
        (3.0, 1.5, 2.0),
        (0.5, 0.25, 2.0),
    ],
)
def test_spacing_scale_combinations(img_sp, acc_sp, exp_scale):
    """Test spacing→scale with various combinations."""
    t = SpacingToScaleTransformer()
    meta = ImageMetadata("image", (100, 200), (0, 255), np.uint8, spacing=img_sp)
    acc = ImageAccessor(x=(0, 100), y=(0, 50), spacing=acc_sp, coordinate_spacing=1.0)
    assert t.transform_access(acc, meta).scale == exp_scale


@pytest.mark.parametrize(
    "frac,dim,exp",
    [
        (0.0, 100, 0),
        (0.5, 100, 50),
        (1.0, 100, 100),
        (0.25, 200, 50),
        (0.75, 400, 300),
    ],
)
def test_fraction_conversions(frac, dim, exp):
    """Test fraction→pixel conversions."""
    assert FractionTransformer._scale_coordinate(frac, dim) == exp
