"""
Transformers that affect how files are accessed.
"""
import math
from .protocol import Transformer
from ..io import ImageAccessor, ImageResult
from ..metadata import ImageMetadata


class SpacingToScaleTransformer(Transformer):

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        from dataclasses import replace

        assert metadata.spacing is not None, f"SpacingToScaleTransformer requires spacing, but metadata does not provide it. Make sure to use a suitable reader, or provide the metadata yourself."
        assert accessor.coordinate_spacing is not None, f"SpacingToScaleTransformer requires accessor.coordinate_spacing."
        assert accessor.spacing is not None, f"SpacingToScaleTransformer requires accessor.spacing."

        accessor = replace(accessor)
        # Compute the scale
        image_spacing = metadata.spacing
        if isinstance(image_spacing, (list, tuple)):
            assert all(image_spacing[0] == i for i in image_spacing), f"SpacingToScaleTransformer does currently not support anisotropic image spacing (got {image_spacing}). PRs welcome."
            image_spacing = image_spacing[0]

        accessor.scale = image_spacing / accessor.spacing
        accessor.x = self._scale_coordinate(coordinate=accessor.x, input_spacing=accessor.coordinate_spacing, output_spacing=image_spacing)
        accessor.y = self._scale_coordinate(coordinate=accessor.y, input_spacing=accessor.coordinate_spacing, output_spacing=image_spacing)

        return accessor

    def transform_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        return metadata

    def transform_image(self, image_result: ImageResult, accessor: ImageAccessor) -> ImageResult:
        return image_result

    @classmethod
    def _scale_coordinate(cls, coordinate, input_spacing, output_spacing):
        if coordinate is None:
            # Read entire image, no change needed with different coordinate.
            return coordinate
        elif isinstance(coordinate, int):
            return int(math.ceil(coordinate * input_spacing / output_spacing))
        else:
            # assume tuple
            return tuple(cls._scale_coordinate(coordinate_i, input_spacing=input_spacing, output_spacing=output_spacing) for coordinate_i in coordinate)


class FractionTransformer(Transformer):

    def transform_access(self, accessor: ImageAccessor, metadata: ImageMetadata) -> ImageAccessor:
        from dataclasses import replace

        assert metadata.shape is not None, f"FractionTransformer requires metadata.shape."

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
        return metadata

    def transform_image(self, image_result: ImageResult, accessor: ImageAccessor) -> ImageResult:
        return image_result

    @classmethod
    def _scale_coordinate(cls, fraction, image_dimension):
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
