"""
Example for transforming metadata.
"""

from tiamat.transformers.metadata import MetadataLambdaTransformer, MetadataKwargsTransformer
from tiamat.pipeline import Pipeline
from tiamat.io import ImageAccessor
import tiamat.metadata.dimensions as d


fname = "./data/Koala.jpg"


# Change interpretation of channels
pipeline = Pipeline(
    transformers=[
        MetadataKwargsTransformer(dimensions=(d.Y, d.X, d.C))
    ]
)

result = pipeline(
    file_name=fname,
    accessor=ImageAccessor(spacing=2.0, coordinate_spacing=1.0),
    image_spacing=2.0,
)
print(result.metadata)
print(result.image.shape)


def metadata_transformation(metadata):
    """
    Inject a valid spacing into the generic image.
    """
    metadata.spacing = 2.0
    return metadata


pipeline = Pipeline(
    transformers=[
        MetadataLambdaTransformer(metadata_transformation),
    ]
)

result = pipeline(
    file_name=fname,
    accessor=ImageAccessor(spacing=2.0, coordinate_spacing=1.0),
    image_spacing=2.0,
)
print(result.metadata)
print(result.image.shape)
