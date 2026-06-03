"""Unit tests for mask transformers."""

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest

from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.transformers.mask import ApplyMaskTransformer


@pytest.fixture
def sample_metadata():
    return ImageMetadata("image", (2, 2), (0, 255), np.uint8)


def test_apply_mask_sets_masked_pixels_and_preserves_input():
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    result = ApplyMaskTransformer.apply_mask(image, mask, mask_value=42)

    npt.assert_array_equal(result, np.array([[1, 42], [42, 4]], dtype=np.uint8))
    npt.assert_array_equal(image, np.array([[1, 2], [3, 4]], dtype=np.uint8))
    npt.assert_array_equal(mask, np.array([[1, 0], [0, 1]], dtype=np.uint8))


def test_mask_file_handle_is_cached():
    reader = MagicMock()
    reader_factory = MagicMock(return_value=reader)

    transformer = ApplyMaskTransformer(
        mask_file="mask.npy",
        mask_value=42,
        reader_factory=reader_factory,
    )

    first = transformer.mask_file_handle
    second = transformer.mask_file_handle

    assert first is second
    reader_factory.assert_called_once_with("mask.npy")


def test_transform_methods_and_image_masking_pipeline(sample_metadata):
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    mask = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    reader = MagicMock()
    reader.read_image.return_value = mask
    transformer = ApplyMaskTransformer(
        mask_file="mask.npy",
        mask_value=42,
        reader_factory=lambda _: reader,
    )

    accessor = ImageAccessor(x=(0, 2), y=(0, 2))

    assert transformer.transform_access(accessor, sample_metadata) is accessor
    assert transformer.transform_metadata(sample_metadata) is sample_metadata

    result = transformer.transform_image(image, sample_metadata, accessor)

    npt.assert_array_equal(result, np.array([[10, 42], [30, 42]], dtype=np.uint8))
    reader.read_image.assert_called_once_with(accessor)


def test_from_json_builds_transformer_with_reader_factory_from_config():
    config = {
        "mask_file": "mask.tif",
        "mask_value": 42,
        "reader_factory": {"type": "reader", "path": "somewhere"},
    }
    resolved_factory = MagicMock()

    with patch("tiamat.serialization.get_reader_from_config", return_value=resolved_factory) as mock_get_reader:
        transformer = ApplyMaskTransformer.from_json(config)

    assert transformer.mask_file == "mask.tif"
    assert transformer.mask_value == 42
    assert transformer.reader_factory is resolved_factory
    mock_get_reader.assert_called_once_with(config["reader_factory"])
