"""
Pipeline for color mapping.
"""

import json

import matplotlib.pyplot as plt

from tiamat.transformers.affine import AffineTransformer
from tiamat.transformers.normalization import MinMaxNormalizationTransformer
from tiamat.transformers.view import BoundingBoxTransformer
from tiamat.io import ImageAccessor
from tiamat.serialization import load_pipeline_from_config, register_class

# register some classes as valid transformers
register_class(AffineTransformer)
register_class(MinMaxNormalizationTransformer)
register_class(BoundingBoxTransformer)

file_name="./data/Koala.jpg"

with open("./data/affine_pipeline.json", "r") as f:
    pipeline_config = json.load(f)
pipeline = load_pipeline_from_config(pipeline_config)

metadata = pipeline.read_metadata(file_name=file_name)
print(metadata)

result = pipeline(
    file_name=file_name, accessor=ImageAccessor(
        x=(-100, metadata.spatial_shape[-1] + 100),
        y=(-100, metadata.spatial_shape[-2] + 100),
    )
)

print(result.image.shape)
plt.imshow(result.image)
plt.show()
