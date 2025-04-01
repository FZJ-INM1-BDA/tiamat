"""
Using pipeline as readers.
"""

from tiamat.transformers.color import GrayscaleTransformer, GrayscaleToRGBTransformer
from tiamat.pipeline import Pipeline
from tiamat.io import ImageAccessor
from tiamat.readers.pipeline import PipelineReader
from functools import partial

pipeline_a = Pipeline(transformers=[GrayscaleTransformer()])

result_a = pipeline_a(
    file_name="./data/Koala.jpg", accessor=ImageAccessor(), image_spacing=1.0
)
print(result_a.image.shape)

pipeline_b = Pipeline(
    transformers=[GrayscaleToRGBTransformer()],
    reader_factory=partial(PipelineReader, pipeline=pipeline_a),
)
result_b = pipeline_b(file_name="./data/Koala.jpg", accessor=ImageAccessor())
print(result_b.image.shape)
