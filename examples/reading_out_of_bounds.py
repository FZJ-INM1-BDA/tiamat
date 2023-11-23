"""
Demonstration of reading out of bounds images.
"""
import matplotlib.pyplot as plt
from tiamat.transformers.normalization import MinMaxNormalizationTransformer
from tiamat.transformers.access import FractionTransformer
from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline

# Convert an image to grayscale, then apply a colormap.
# Let's also combine it with some coordinate transformers.
pipeline = Pipeline(
    transformers=[FractionTransformer(), MinMaxNormalizationTransformer(), ],
)
result_0 = pipeline(ImageAccessor("./data/Koala.jpg", x=(-0.1, 0.1), y=(-0.1, 0.1)))
print(result_0.image.shape)
result_1 = pipeline(ImageAccessor("./data/Koala.jpg", x=(0.9, 1.1), y=(0.9, 1.1)))
print(result_1.image.shape)
result_2 = pipeline(ImageAccessor("./data/Koala.jpg", x=(-0.1, 1.1), y=(-0.1, 1.1)))
print(result_2.image.shape)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(result_0.image)
ax1.imshow(result_1.image)
ax2.imshow(result_2.image)
plt.show()
