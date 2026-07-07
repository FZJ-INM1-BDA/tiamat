"""
Mask transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tiamat.cache import instance_cached_property
from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.readers.protocol import ImageReader
from tiamat.transformers.protocol import Transformer


class ApplyMaskTransformer(Transformer):
    """Transformer which applies a mask to an image, setting masked pixels to a specified value."""

    def __init__(
        self,
        mask_file: str,
        mask_value: int | float = 0,
        reader_factory: Callable[[str], ImageReader] | None = None,
    ):
        """
        Creates an instance of ApplyMaskTransformer.

        Args:
            mask_file: Path to the mask file.
            mask_value: Value to set masked pixels to. Defaults to 0.
            reader_factory: Callable returning ImageReader for a given file path.
        """

        from tiamat.readers.factory import get_reader

        self.mask_file = mask_file
        self.mask_value = mask_value
        self.reader_factory = reader_factory or get_reader

    @instance_cached_property
    def mask_file_handle(self) -> ImageReader:
        """Return a cached ImageReader for the mask file."""
        return self.reader_factory(self.mask_file)

    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray, mask_value: int | float) -> np.ndarray:
        """Apply the mask to the image, setting masked pixels to mask_value."""
        masked_image = image.copy()
        masked_image[mask == 0] = mask_value
        return masked_image

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:

        metadata_mask = self.mask_file_handle.read_metadata()
        if metadata.spatial_shape != metadata_mask.spatial_shape:
            raise ValueError(
                "Spatial shape mismatch: "
                f"mask={metadata_mask.spatial_shape}, "
                f"image={metadata.spatial_shape}. "
                "It is currently required to match exactly."
            )
        if metadata.spacing != metadata_mask.spacing:
            raise ValueError(
                "Spacing mismatch: "
                f"mask={metadata_mask.spacing}, "
                f"image={metadata.spacing}. "
                "It is currently required to match exactly."
            )
        if metadata.scales != metadata_mask.scales:
            raise ValueError(
                "Scales mismatch: "
                f"mask={metadata_mask.scales}, "
                f"image={metadata.scales}. "
                "It is currently required to match exactly."
            )

        return self.apply_mask(image, self.mask_file_handle.read_image(accessor), self.mask_value)

    @classmethod
    def from_json(cls, args: dict[str, Any]):
        """
        Instantiate an ApplyMaskTransformer from a JSON-like dictionary.

        Args:
            args: Dictionary containing configuration keys:
                - mask_file: Path to the mask file (required).
                - mask_value: Value to set masked pixels to (optional, default=0).
                - reader_factory: Optional configuration for a custom ImageReader factory.

        Returns:
            An instance of ApplyMaskTransformer configured according to args.
        """
        from tiamat.serialization import get_reader_from_config

        return cls(
            mask_file=args["mask_file"],
            mask_value=args.get("mask_value", 0),
            reader_factory=get_reader_from_config(args.get("reader_factory")),
        )
