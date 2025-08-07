import matplotlib.pyplot as plt
import numpy as np
from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline
from tiamat.transformers.access import FractionTransformer
from tiamat.transformers.affine import AffineTransformer
from tiamat.transformers.color import GrayscaleTransformer

# Basic pipeline with FractionTransformer
pipeline = Pipeline(
    access_transformers=[FractionTransformer()],
    image_transformers=[GrayscaleTransformer()],
)

# Pipeline with AffineTransformer and FractionTransformer
pipeline_affine = Pipeline(
    access_transformers=[FractionTransformer()],
    image_transformers=[
        GrayscaleTransformer(),
        AffineTransformer(affine_matrix=np.eye(3)),  # Identity transformation
    ],
)

# Pipeline with AffineTransformer but without FractionTransformer
pipeline_affine_wo_fraction = Pipeline(
    image_transformers=[
        GrayscaleTransformer(),
        AffineTransformer(affine_matrix=np.eye(3)),  # Same identity transformation
    ],
)

# example image
file_name = "./data/Koala.jpg"

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Results from different pipeline configurations
result_pipeline = pipeline(
    file_name=file_name,
    accessor=ImageAccessor(x=(0, 1), y=(0, 1)),
)

axs[0, 0].imshow(result_pipeline.image)
axs[0, 0].set_title("Original")
axs[0, 1].axis('off') 

result_pipeline_affine = pipeline_affine(
    file_name=file_name,
    accessor=ImageAccessor(x=(0, 1), y=(0, 1)),
)

axs[1, 0].imshow(result_pipeline_affine.image)
axs[1, 0].set_title("Affine w FractionTransformer")

result_pipeline_affine_wo_fraction = pipeline_affine_wo_fraction(
    file_name=file_name,
    accessor=ImageAccessor(
        x=(0, (result_pipeline.image).shape[1]),
        y=(0, (result_pipeline.image).shape[0])
    ),
)

axs[1, 1].imshow(result_pipeline_affine_wo_fraction.image)
axs[1, 1].set_title("Affine w/o FractionTransformer")
plt.show()
