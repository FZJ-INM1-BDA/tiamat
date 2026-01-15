"""
Comprehensive tests for coordinate utility functions.
"""

import numpy as np
import pytest

from tiamat.transformers.coordinates import (
    get_coordinate_bounds,
    resolve_coordinate_slice,
)


class TestResolveCoordinateSlice:
    """Tests for resolve_coordinate_slice."""

    def test_all_input_types_and_none_handling(self):
        """Test None, int, float, numpy types, tuples with None."""
        # None → (0, dimension)
        assert resolve_coordinate_slice(None, 100) == (0, 100)

        # Single int/float unchanged
        assert resolve_coordinate_slice(50, 100) == 50
        assert resolve_coordinate_slice(50.5, 100) == 50.5

        # Numpy types
        assert resolve_coordinate_slice(np.int32(50), 100) == 50
        assert resolve_coordinate_slice(np.float64(50.5), 100) == 50.5

        # Tuple with both values
        assert resolve_coordinate_slice((10, 90), 100) == (10, 90)

        # Tuple with None start → 0
        assert resolve_coordinate_slice((None, 50), 100) == (0, 50)

        # Tuple with None stop → dimension
        assert resolve_coordinate_slice((50, None), 100) == (50, 100)

        # Tuple with both None → full
        assert resolve_coordinate_slice((None, None), 100) == (0, 100)

        # Float dimension works
        assert resolve_coordinate_slice((None, None), 100.5) == (0, 100.5)

        # Negative coords (no special handling)
        assert resolve_coordinate_slice((-10, 90), 100) == (-10, 90)

        # Beyond dimension (no clamping)
        assert resolve_coordinate_slice((50, 150), 100) == (50, 150)


class TestGetCoordinateBounds:
    """Tests for get_coordinate_bounds."""

    def test_all_input_types_and_conversion_to_bounds(self):
        """Test None, int, float, tuples, and conversion to (start, end) format."""
        # None → (0, dimension)
        assert get_coordinate_bounds(None, 100) == (0, 100)

        # Single int → (int, int)
        assert get_coordinate_bounds(50, 100) == (50, 50)
        assert get_coordinate_bounds(0, 100) == (0, 0)

        # Single float → (float, float)
        assert get_coordinate_bounds(50.5, 100) == (50.5, 50.5)

        # Tuple → (start, end)
        assert get_coordinate_bounds((10, 90), 100) == (10, 90)

        # Tuple with None resolved
        assert get_coordinate_bounds((None, 50), 100) == (0, 50)
        assert get_coordinate_bounds((50, None), 100) == (50, 100)
        assert get_coordinate_bounds((None, None), 100) == (0, 100)

        # Numpy types
        assert get_coordinate_bounds(np.int32(50), 100) == (50, 50)
        assert get_coordinate_bounds((np.int32(10), np.int32(90)), 100) == (10, 90)

        # Edge cases
        assert get_coordinate_bounds(None, 0) == (0, 0)  # Zero dimension
        assert get_coordinate_bounds((50, 50), 100) == (50, 50)  # Zero-width
        assert get_coordinate_bounds((50, 40), 100) == (50, 40)  # Inverted (no validation)


class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases."""

    def test_typical_accessor_patterns_and_type_consistency(self):
        """Test common usage patterns and type consistency."""
        dim = 200

        # Typical patterns
        assert get_coordinate_bounds(None, dim) == (0, 200)  # Full
        assert get_coordinate_bounds((50, 150), dim) == (50, 150)  # Slice
        assert get_coordinate_bounds(100, dim) == (100, 100)  # Point
        assert get_coordinate_bounds((None, 100), dim) == (0, 100)  # From start
        assert get_coordinate_bounds((100, None), dim) == (100, 200)  # To end

        # Large dimensions
        large = 1000000
        assert resolve_coordinate_slice(None, large) == (0, large)
        assert get_coordinate_bounds((100000, 900000), large) == (100000, 900000)

        # Float vs int consistency
        assert resolve_coordinate_slice((10, 90), 100) == resolve_coordinate_slice((10, 90), 100.0)

        # Return types are tuples
        assert isinstance(resolve_coordinate_slice((None, None), 100), tuple)
        assert isinstance(get_coordinate_bounds(50, 100), tuple)


# Parametrized tests cover all cases compactly
@pytest.mark.parametrize(
    "coord,dim,expected",
    [
        (None, 100, (0, 100)),
        (50, 100, 50),
        ((10, 90), 100, (10, 90)),
        ((None, 50), 100, (0, 50)),
        ((50, None), 100, (50, 100)),
        ((None, None), 100, (0, 100)),
        (0, 100, 0),
        (99, 100, 99),
        ((0, 100), 100, (0, 100)),
    ],
)
def test_resolve_parametrized(coord, dim, expected):
    """Parametrized test for resolve_coordinate_slice."""
    assert resolve_coordinate_slice(coord, dim) == expected


@pytest.mark.parametrize(
    "coord,dim,expected",
    [
        (None, 100, (0, 100)),
        (50, 100, (50, 50)),
        ((10, 90), 100, (10, 90)),
        ((None, 50), 100, (0, 50)),
        ((50, None), 100, (50, 100)),
        ((None, None), 100, (0, 100)),
        (0, 200, (0, 0)),
        (199, 200, (199, 199)),
    ],
)
def test_get_bounds_parametrized(coord, dim, expected):
    """Parametrized test for get_coordinate_bounds."""
    assert get_coordinate_bounds(coord, dim) == expected
