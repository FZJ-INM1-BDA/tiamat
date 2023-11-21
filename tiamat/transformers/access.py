"""
Transformers that affect how files are accessed.
"""
from .protocol import Transformer


class SpacingToScaleTransformer(Transformer):

    def transform_coordinates(self, x, y, scale, metadata=None, z=None, c=None):
        assert metadata is not None, f"SpacingToScaleTransformer requires metadata."
        assert metadata.spacing is not None, f"SpacingToScaleTransformer requires spacing, but metadata does not provide it. Make sure to use a suitable reader."

        # This transformer expectes spacing as scale
        target_spacing = scale
        image_spacing = metadata.spacing
        scale = image_spacing / target_spacing

        return x, y, scale, metadata, z, c

    def transform_image(self, image, metadata=None):
        return image, metadata
