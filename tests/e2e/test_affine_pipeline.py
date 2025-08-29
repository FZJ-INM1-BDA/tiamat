from collections import namedtuple
from pathlib import Path
import gzip
import shutil

import pytest
import numpy as np
from imageio.v3 import imwrite

from tiamat.transformers.affine import AffineTransformer
from tiamat.transformers.normalization import MinMaxNormalizationTransformer
from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline


image_width, image_height = 2560, 1600

access_frame = np.array([
    [-image_width, image_width],
    [-image_height, image_height],
], dtype=int)

def build_affine(scale=1.0, rotation=0, mirror_x=False, mirror_y=False, translate=[0., 0.]):
    angle_rad = np.deg2rad(rotation)

    cos_angle = np.cos(angle_rad) * scale
    sin_angle = np.sin(angle_rad) * scale
    
    # Build the affine transformation matrix
    rot_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0., 0., 1.]
    ], dtype=np.float32)
    mirror_matrix = np.array([
        [1 - 2 * float(mirror_x), 0, 0],
        [0, 1 - 2 * float(mirror_y), 0],
        [0., 0., 1.]
    ], dtype=np.float32)
    affine_matrix = rot_matrix @ mirror_matrix
    affine_matrix[:2, -1] = translate
    
    return affine_matrix

# this produces gigs 
# @pytest.mark.parametrize('scale', [0.75, 1, 1.5])
# @pytest.mark.parametrize('rotation', [0, 45, -45])
# @pytest.mark.parametrize('mirror_x', [True, False])
# @pytest.mark.parametrize('mirror_y', [True, False])
# @pytest.mark.parametrize('translate_x', [0, -500, 500])
# @pytest.mark.parametrize('translate_y', [0, -500, 500])

TestArg = namedtuple("TestArg",
                     ["scale", "rotation", "mirror_x", "mirror_y", "translate_x", "translate_y"],
                     defaults=[1, 0, False, False, 0, 0])

test_args = {
    "iden": TestArg(),
    
    "scale_down": TestArg(scale=0.75),
    "scale_up": TestArg(scale=1.25),

    "rot_cw": TestArg(rotation=45),
    "rot_ccw": TestArg(rotation=-45),
    "rot_ccw_alt": TestArg(rotation=315),

    "mirror_x": TestArg(mirror_x=True),
    "mirror_y": TestArg(mirror_y=True),
    "mirror_xy": TestArg(mirror_x=True, mirror_y=True),

    "translate_-x": TestArg(translate_x=-500),
    "translate_+x": TestArg(translate_x=500),
    "translate_-y": TestArg(translate_y=-500),
    "translate_+y": TestArg(translate_y=500),

    "mix_1": TestArg(mirror_x=True, translate_x=2560),
    "mix_2": TestArg(mirror_y=True, translate_y=1600),
}

@pytest.mark.parametrize("scale, rotation, mirror_x, mirror_y, translate_x, translate_y", 
                         test_args.values(),
                         ids=test_args.keys(),)
def test_affine_pipeline(
    scale, rotation, mirror_x, mirror_y, translate_x, translate_y, request
):
    
    affine_matrix = build_affine(scale, rotation, mirror_x, mirror_y, (translate_x, translate_y))
    pipeline = Pipeline(
        transformers=[MinMaxNormalizationTransformer(), AffineTransformer(affine_matrix=affine_matrix), ],
    )
    result = pipeline(file_name="./examples/data/Koala.jpg", accessor=ImageAccessor(x=access_frame[0], y=access_frame[1], fill_value=0))

    filename = f"test_affine_s{scale}_r{rotation}"
    if mirror_x:
        filename += f"_mirrorX"
    if mirror_y:
        filename += f"_mirrorY"
    filename += f"_t{translate_x}_{translate_y}"

    png_filename = f"tests/e2e/references/{request.node.callspec.id}--{filename}.png"
    npygz_filename = f"tests/e2e/references/{request.node.callspec.id}--{filename}.npy.gz"
    
    dst_png_filename = f"artefacts/tests/e2e/references/{request.node.callspec.id}--{filename}.png"
    dst_npy_filename = f"artefacts/tests/e2e/references/{request.node.callspec.id}--{filename}.npy"
    dst_npygz_filename = f"artefacts/tests/e2e/references/{request.node.callspec.id}--{filename}.npy.gz"
    
    uint8_img = np.array(result.image * 255, dtype=np.uint8)
    expected_arr = np.load(npygz_filename)
    np.testing.assert_equal(uint8_img, expected_arr)

    Path(dst_png_filename).parent.mkdir(exist_ok=True, parents=True)
    np.save(dst_npy_filename, uint8_img, False)
    with gzip.open(dst_npygz_filename, "wb") as gzip_file:
        with open(dst_npy_filename, "rb") as npy_file:
            shutil.copyfileobj(npy_file, gzip_file)
    Path(dst_npy_filename).unlink()
    imwrite(dst_png_filename, uint8_img)

