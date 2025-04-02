"""
Pipeline for color mapping.
"""

import json

import matplotlib.pyplot as plt

from tiamat.transformers.affine import AffineTransformer
from tiamat.transformers.normalization import MinMaxNormalizationTransformer
from tiamat.io import ImageAccessor
from tiamat.serialization import load_pipeline_from_config, register_class

# register some classes as valid transformers
register_class(AffineTransformer)
register_class(MinMaxNormalizationTransformer)

with open("./data/affine_pipeline.json", "r") as f:
    pipeline_config = json.load(f)
pipeline = load_pipeline_from_config(pipeline_config)
result = pipeline(
    file_name="./data/Koala.jpg", accessor=ImageAccessor(x=(0, 3840), y=(0, 2400))
)

plt.imshow(result.image)
plt.show()
