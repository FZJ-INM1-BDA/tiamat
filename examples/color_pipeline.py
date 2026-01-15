"""
Pipeline for color mapping.
"""

import matplotlib.pyplot as plt

from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline
from tiamat.transformers.access import FractionTransformer
from tiamat.transformers.color import GrayscaleTransformer, LUTTransformer

# Convert an image to grayscale, then apply a colormap.
# Let's also combine it with some coordinate transformers.
pipeline = Pipeline(
    access_transformers=[FractionTransformer()],
    image_transformers=[GrayscaleTransformer(), LUTTransformer(color_map="plasma")],
)
result = pipeline(file_name="./data/Koala.jpg", accessor=ImageAccessor(x=(0.25, 0.75), y=(0.25, 0.75)))

plt.imshow(result.image)
plt.show()
