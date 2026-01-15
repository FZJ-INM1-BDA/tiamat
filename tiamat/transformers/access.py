"""
Transformers that affect how files are accessed.
"""

import math

import numpy as np

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import Transformer


class SpacingToScaleTransformer(Transformer):
    """
    Transformer that converts spacing information from metadata into scaling factors for accessors.

    This transformer adjusts the `scale`, `x`, and `y` coordinates of an ImageAccessor
    based on the image spacing provided in the metadata.
    """

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Adjusts the accessor's scale and coordinates according to metadata spacing.

        Args:
            accessor: The ImageAccessor to transform.
            metadata:  The metadata of the image to transform.

        Returns:
            A new ImageAccessor with adjusted scale and coordinates.

        Raises:
            AssertionError: If metadata or required spacing fields are missing.
            AssertionError: If anisotropic image spacing is encountered (unsupported).
        """
        from dataclasses import replace

        assert metadata.spacing is not None, (
            "SpacingToScaleTransformer requires spacing, but metadata does not provide it. "
            "Make sure to use a suitable reader, or provide the metadata yourself."
        )
        assert (
            accessor.coordinate_spacing is not None
        ), "SpacingToScaleTransformer requires accessor.coordinate_spacing."
        assert accessor.spacing is not None, "SpacingToScaleTransformer requires accessor.spacing."

        accessor = replace(accessor)
        # Compute the scale
        image_spacing = metadata.spacing
        if isinstance(image_spacing, (list, tuple)):

            assert all(image_spacing[0] == i for i in image_spacing), (
                "SpacingToScaleTransformer does currently not support anisotropic "
                f"image spacing (got {image_spacing}). PRs welcome."
            )
            image_spacing = image_spacing[0]

        accessor.scale = image_spacing / accessor.spacing
        accessor.x = self._scale_coordinate(
            coordinate=accessor.x, input_spacing=accessor.coordinate_spacing, output_spacing=image_spacing
        )
        accessor.y = self._scale_coordinate(
            coordinate=accessor.y, input_spacing=accessor.coordinate_spacing, output_spacing=image_spacing
        )

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """Return metadata unchanged."""
        from dataclasses import replace

        metadata = replace(metadata)

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Return the image unchanged.

        Args:
            image (np.ndarray): The input image.
            metadata (ImageMetadata): The image metadata.
            accessor (ImageAccessor): The accessor used for this transformation.

        Returns:
            np.ndarray: The unchanged image.
        """
        return image

    @classmethod
    def _scale_coordinate(
        cls,
        coordinate: int | tuple[int, ...],
        input_spacing: float,
        output_spacing: float,
    ) -> int | tuple[int, ...]:
        """
        Scale a coordinate from one spacing system to another.

        Args:
            coordinate (int | tuple[int, ...] | None): The coordinate to scale.
            input_spacing (float): The input spacing value.
            output_spacing (float): The output spacing value.

        Returns:
            int | tuple[int, ...] | None: The scaled coordinate, or None if not provided.
        """
        if coordinate is None:
            # Read entire image, no change needed with different coordinate.
            return coordinate
        elif isinstance(coordinate, int):
            return int(math.ceil(coordinate * input_spacing / output_spacing))
        else:
            # assume tuple
            return tuple(
                cls._scale_coordinate(coordinate_i, input_spacing=input_spacing, output_spacing=output_spacing)
                for coordinate_i in coordinate
            )


class FractionTransformer(Transformer):
    """
    Transformer that converts fractional coordinates into absolute pixel coordinates
    based on the metadata shape.
    """

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        Scale accessor coordinates from fractions of image dimensions to absolute pixel positions.

        Args:
            accessor (ImageAccessor): The ImageAccessor to transform.
            metadata (ImageMetadata): The metadata providing the image shape.

        Returns:
            ImageAccessor: A new accessor with scaled coordinates.

        Raises:
            AssertionError: If `metadata.shape` is missing.
        """
        from dataclasses import replace

        assert metadata.shape is not None, "FractionTransformer requires metadata.shape."

        accessor = replace(accessor)
        # Note the correct the dimensions for x and y.
        accessor.x = self._scale_coordinate(accessor.x, metadata.shape[1])
        accessor.y = self._scale_coordinate(accessor.y, metadata.shape[0])
        if len(metadata.shape) > 2:
            accessor.z = self._scale_coordinate(accessor.z, metadata.shape[2])
        if len(metadata.shape) > 3:
            accessor.c = self._scale_coordinate(accessor.c, metadata.shape[3])

        # In case there is a spacing provided, the coordinates are now given in this spacing.
        accessor.coordinate_spacing = metadata.spacing

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Return metadata unchanged.

        Args:
            metadata (ImageMetadata): The metadata to process.

        Returns:
            ImageMetadata: The unchanged metadata.
        """
        from dataclasses import replace

        metadata = replace(metadata)

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Return the image unchanged.

        Args:
            image (np.ndarray): The input image.
            metadata (ImageMetadata): The image metadata.
            accessor (ImageAccessor): The accessor used for this transformation.

        Returns:
            np.ndarray: The unchanged image.
        """
        return image

    @classmethod
    def _scale_coordinate(
        cls,
        fraction: int | float | tuple[int | float, ...],
        image_dimension: int,
    ) -> int | tuple[int, ...]:
        """
        Scale a fractional coordinate into an absolute pixel coordinate.

        Args:
            fraction (int | float | tuple[int | float, ...] | None): Fraction(s) of the dimension.
            image_dimension (int): The corresponding image dimension length.

        Returns:
            int | tuple[int, ...] | None: The scaled coordinate as an int, tuple of ints, or None.
        """
        import math

        if isinstance(fraction, (int, float)):
            # scale dimension by fraction
            return math.ceil(image_dimension * fraction)
        elif fraction is None:
            # leave untouched
            return fraction
        else:
            # scale each coordinate
            return tuple(cls._scale_coordinate(fraction_i, image_dimension) for fraction_i in fraction)
