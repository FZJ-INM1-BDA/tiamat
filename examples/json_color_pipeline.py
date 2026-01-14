"""
Pipeline for color mapping.
"""

import json

import matplotlib.pyplot as plt

from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline
from tiamat.serialization import load_pipeline_from_config, register_class
from tiamat.transformers.access import FractionTransformer
from tiamat.transformers.color import GrayscaleTransformer, LUTTransformer

# register some classes as valid transformers
register_class(GrayscaleTransformer)
register_class(FractionTransformer)
register_class(LUTTransformer)

with open("./data/color_pipeline.json", "r") as f:
    pipeline_config = json.load(f)
pipeline = load_pipeline_from_config(pipeline_config)
result = pipeline(
    file_name="./data/Koala.jpg",
    accessor=ImageAccessor(x=(0.25, 0.75), y=(0.25, 0.75)),
)

plt.imshow(result.image)
plt.show()
