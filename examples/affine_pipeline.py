"""
Pipeline for affine transformation.
"""
import numpy as np
import matplotlib.pyplot as plt
from tiamat.transformers.affine import AffineTransformer
from tiamat.transformers.normalization import MinMaxNormalizationTransformer
from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline

scale = 0.5
scale_matrix = np.eye(3)
scale_matrix[:2, :2] *= scale

affine_matrix = scale_matrix

# Convert an image to grayscale, then apply a colormap.
# Let's also combine it with some coordinate transformers.
pipeline = Pipeline(
    transformers=[MinMaxNormalizationTransformer(), AffineTransformer(affine_matrix=affine_matrix), ],
)
result = pipeline(ImageAccessor("./data/Koala.jpg", x=(int(1350 * scale), int(1550 * scale)), y=(int(330 * scale), int(530 * scale))))

print(result.image.shape)
plt.imshow(result.image)
plt.show()
