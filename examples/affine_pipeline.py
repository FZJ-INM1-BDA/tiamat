"""
Pipeline for affine transformation.
"""
import numpy as np
import matplotlib.pyplot as plt
from tiamat.transformers.affine import AffineTransformer
from tiamat.transformers.normalization import MinMaxNormalizationTransformer
from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline

# Image is 1600, 2560  shape
image_width, image_height = 2560, 1600
scale = 1.0
rotation = 45
mirror_x = 0
mirror_y = True
translate = [0, image_height]

access_frame = np.array([
    [0 * scale, (image_width) * scale],
    [0 * scale, (image_height) * scale],
], dtype=int)

###

def build_affine(scale=1.0, rotation=0, mirror_x=False, mirror_y=False, translate=[0., 0.]):
    angle_rad = np.deg2rad(rotation)

    cos_angle = np.cos(angle_rad) * scale
    sin_angle = np.sin(angle_rad) * scale
    
    # Build the affine transformation matrix
    affine_matrix = np.array([
        [(1. - 2. * float(mirror_x)) * cos_angle, -sin_angle, translate[0]],
        [sin_angle, (1. - 2. * float(mirror_y)) * cos_angle, translate[1]],
        [0., 0., 1.]
    ], dtype=np.float32)
    
    return affine_matrix

affine_matrix = build_affine(scale, rotation, mirror_x, mirror_y, translate)

print(affine_matrix)

# Convert an image to grayscale, then apply a colormap.
# Let's also combine it with some coordinate transformers.
pipeline = Pipeline(
    transformers=[MinMaxNormalizationTransformer(), AffineTransformer(affine_matrix=affine_matrix), ],
)
result = pipeline(file_name="./data/Koala.jpg", accessor=ImageAccessor(x=access_frame[0], y=access_frame[1]))

print(result.image.shape)
plt.imshow(result.image)
plt.show()
