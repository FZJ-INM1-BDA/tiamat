"""
Example for transforming metadata.
"""

import numpy as np
from tiamat.transformers.metadata import MetadataLambdaTransformer
from tiamat.pipeline import Pipeline
from tiamat.io import ImageAccessor


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
    file_name="./data/Koala.jpg",
    accessor=ImageAccessor(spacing=2.0, coordinate_spacing=1.0),
    image_spacing=2.0,
)
print(result.metadata)
print(result.image.shape)
