"""
Module containing transformers for intensity mapping of images.
"""

from collections.abc import Callable

import numpy as np

from ..io import ImageAccessor
from ..metadata import ImageMetadata
from .protocol import Transformer


class MappingTransformer(Transformer):
    """
    Pointwise intensity mapping defined by a mathematical function.
    """

    def __init__(
        self,
        mapping_function: Callable[[np.ndarray | float], np.ndarray | float],
        target_dtype: type[np.floating] = np.float32,
    ) -> None:
        """
        Initialize the transformer.

        Args:
            mapping_function: Callable used to map both image values and metadata
                range bounds.
            target_dtype: Floating point dtype for the mapped image.
                Defaults to np.float32.
        """

        self.mapping_function = mapping_function
        self.target_dtype = target_dtype

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        """
        No changes to the accessor are needed for intensity mapping.

        Args:
            accessor: ImageAccessor of the target image.
            metadata: Metadata of the target image.

        Returns:
            The unmodified ImageAccessor.
        """

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """
        Update metadata to reflect intensity mapping.

        Args:
            metadata: Original ImageMetadata.

        Returns:
            Updated ImageMetadata with dtype set to target_dtype and value_range mapped
            through the transform function.
        """

        from dataclasses import replace

        metadata = replace(metadata)

        metadata.dtype = self.target_dtype
        assert metadata.value_range is not None, f"{type(self).__name__} requires metadata.value_range."
        metadata.value_range = self._map_value_range(metadata.value_range)

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Apply pointwise intensity mapping to the image.

        Args:
            image: Image to transform.
            metadata: Metadata of the image.
            accessor: Accessor to the image.

        Returns:
            np.ndarray: Mapped image.
        """

        transformed_image = self._map_array(image)
        return transformed_image.astype(self.target_dtype)

    def _map_value_range(self, value_range: tuple[float, float]) -> tuple[float, float]:
        """Map the metadata value range through the transform function."""

        vmin, vmax = value_range
        return (self._map_scalar(vmin), self._map_scalar(vmax))

    def _map_array(self, image: np.ndarray) -> np.ndarray:
        """Map the image values through the transform function."""

        return self.mapping_function(image)

    def _map_scalar(self, value: float) -> float:
        """Map a scalar through the transform function."""

        return self.mapping_function(value)


# maybe only in tiamat-openseadragon?
# Commen used functions
class LogIntensityMappingTransformer(MappingTransformer):
    """
    Map image intensities using a logarithmic response curve.
    """

    def __init__(
        self,
        target_dtype: type[np.floating] = np.float32,
    ) -> None:
        """Initialize the transformer with a logarithmic mapping."""

        super().__init__(
            mapping_function=np.log,
            target_dtype=target_dtype,
        )


class Log1pIntensityMappingTransformer(MappingTransformer):
    """
    Map image intensities using a logarithmic response curve with log1p.
    """

    def __init__(
        self,
        target_dtype: type[np.floating] = np.float32,
    ) -> None:
        """Initialize the transformer with a log1p mapping."""

        super().__init__(
            mapping_function=np.log1p,
            target_dtype=target_dtype,
        )


class GammaIntensityMappingTransformer(MappingTransformer):
    """
    Map image intensities using gamma correction.
    """

    def __init__(
        self,
        gamma: float,
        target_dtype: type[np.floating] = np.float32,
    ) -> None:
        """
        Initialize the transformer.

        Args:
            gamma: Gamma value for correction.
            target_dtype: Floating point dtype for the mapped image.
                Defaults to np.float32.
        """

        self.gamma = gamma
        super().__init__(
            mapping_function=lambda value: np.power(value, self.gamma),
            target_dtype=target_dtype,
        )


class ClipIntensityMappingTransformer(MappingTransformer):
    """
    Map image intensities by clipping to a specified range.
    """

    def __init__(
        self,
        vmin: float,
        vmax: float,
        target_dtype: type[np.floating] = np.float32,
    ) -> None:
        """
        Initialize the transformer.

        Args:
            vmin: Minimum intensity value for clipping.
            vmax: Maximum intensity value for clipping.
            target_dtype: Floating point dtype for the mapped image.
                Defaults to np.float32.
        """

        self.vmin = vmin
        self.vmax = vmax
        super().__init__(
            mapping_function=lambda value: np.clip(value, self.vmin, self.vmax),
            target_dtype=target_dtype,
        )


class BrightnessContrastIntensityMappingTransformer(Transformer):
    """
    Map image intensities using brightness and contrast adjustment.
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 1.0,
        target_dtype: type[np.floating] = np.float32,
    ) -> None:
        """
        Initialize the transformer.

        Args:
            brightness: Additive offset expressed as a fraction of the
                metadata value range.
            contrast: Multiplicative contrast factor around the metadata
                range midpoint.
            target_dtype: Floating point dtype for the mapped image.
                Defaults to np.float32.
        """

        self.brightness = brightness
        self.contrast = contrast
        self.target_dtype = target_dtype
        super().__init__(
            mapping_function=lambda value: value,
            target_dtype=target_dtype,
        )

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """Update metadata to reflect brightness and contrast mapping."""

        from dataclasses import replace

        metadata = replace(metadata)
        metadata.dtype = self.target_dtype

        assert (
            metadata.value_range is not None
        ), "BrightnessContrastIntensityMappingTransformer requires metadata.value_range."

        vmin, vmax = metadata.value_range
        midpoint = (vmin + vmax) / 2.0
        value_range = vmax - vmin

        mapped_vmin = np.clip(
            (vmin - midpoint) * self.contrast + midpoint + self.brightness * value_range,
            vmin,
            vmax,
        )
        mapped_vmax = np.clip(
            (vmax - midpoint) * self.contrast + midpoint + self.brightness * value_range,
            vmin,
            vmax,
        )
        metadata.value_range = (float(mapped_vmin), float(mapped_vmax))

        return metadata

    def transform_image(self, image: np.ndarray, metadata: ImageMetadata, accessor: ImageAccessor) -> np.ndarray:
        """
        Apply brightness and contrast intensity mapping to the image.

        Args:
            image: Image to transform.
            metadata: Metadata of the image.
            accessor: Accessor to the image.

        Returns:
            np.ndarray: Brightness/contrast-mapped image clipped to metadata
            value_range.
        """

        assert (
            metadata.value_range is not None
        ), "BrightnessContrastIntensityMappingTransformer requires metadata.value_range."

        vmin, vmax = metadata.value_range
        midpoint = (vmin + vmax) / 2.0
        value_range = vmax - vmin

        # Apply contrast around native midpoint and brightness as a fraction of range,
        # then clip to metadata range.
        self.mapping_function = lambda value: np.clip(
            (value.astype(self.target_dtype) - midpoint) * self.contrast + midpoint + self.brightness * value_range,
            vmin,
            vmax,
        )

        return self.mapping_function(image).astype(self.target_dtype)
