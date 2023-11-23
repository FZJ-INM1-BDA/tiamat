"""
Simple example of using a pipeline.
"""
import numpy as np
from tiamat.transformers.access import SpacingToScaleTransformer, FractionTransformer
from tiamat.pipeline import Pipeline
from tiamat.io import ImageAccessor

# We want to be able to access our image using physical coordinates, so we add a SpacingToScaleTransformer.
pipeline = Pipeline(transformers=[SpacingToScaleTransformer(), ])

# We want to read the entire image on spacing 2.0.
# The generic reader has no way of determining the spacing of the image, so we have to pass it here.
result = pipeline(ImageAccessor(file_name="./data/Koala.jpg", spacing=2.0, coordinate_spacing=1.0), image_spacing=1.0)
print(result.image.shape)

# Now something more sophisticated: We specify the coordinates, in a different spacing than what we want to read...
result_4micron = pipeline(ImageAccessor(file_name="./data/Koala.jpg", spacing=2.0, coordinate_spacing=4.0, x=(200, 300), y=(200, 300)), image_spacing=1.0)
print(result_4micron.image.shape)

# ...which is the same as:
result_2micron = pipeline(ImageAccessor(file_name="./data/Koala.jpg", spacing=2.0, coordinate_spacing=2.0, x=(400, 600), y=(400, 600)), image_spacing=1.0)
assert np.allclose(result_2micron.image, result_4micron.image)

# ...and:
result_1micron = pipeline(ImageAccessor(file_name="./data/Koala.jpg", spacing=2.0, coordinate_spacing=1.0, x=(800, 1200), y=(800, 1200)), image_spacing=1.0)
assert np.allclose(result_1micron.image, result_4micron.image)

# An alternative pipeline. It allows us to specify image coordinates relative to the size of the image.
pipeline = Pipeline(transformers=[FractionTransformer()])
# We can read the entire image as usual.
result = pipeline(ImageAccessor(file_name="./data/Koala.jpg"))
print(result.image.shape)

# But we can now also specify image dimension in fractions of the total image size.
result = pipeline(ImageAccessor(file_name="./data/Koala.jpg", x=(0, 0.5), y=(0, 0.5)))
print(result.image.shape)

result = pipeline(ImageAccessor(file_name="./data/Koala.jpg", x=(0, 0.5), y=(0, 0.5)))
print(result.image.shape)

# We can even also combine the two transformers.
pipeline = Pipeline(transformers=[SpacingToScaleTransformer(), FractionTransformer()])
# The following happens here:
# - image_spacing=1.0 tells our reader that the image has spacing 1.0. This is required for generic images.
# - we specify that we want our result to be on spacing=2.0, so a downscaling of 1/2.
# - our coordinates are specified in fractions.
result = pipeline(ImageAccessor(file_name="./data/Koala.jpg", x=(0, 0.5), y=(0, 0.5), spacing=2.0), image_spacing=1.0)
print(result.image.shape)
