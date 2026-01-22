import gzip
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from collections import namedtuple
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from imageio.v3 import imwrite

from tiamat.io import ImageAccessor
from tiamat.pipeline import Pipeline
from tiamat.transformers.affine import AffineTransformer
from tiamat.transformers.normalization import MinMaxNormalizationTransformer

# -----------------------------------------------------------------------------
# External reference data handling (pure Python download + extract into temp dir)
# -----------------------------------------------------------------------------

# The test file specifies the commit to use.
# (Start with the example commit you provided.)
TIAMAT_TEST_DATA_COMMIT = "674007f608add0cb061a294fcd82ee24931c2173"

# Public GitLab archive URLs (try a few common formats to be robust).
_TEST_DATA_PROJECT_BASE = "https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat-test-data"
_TEST_DATA_ARCHIVE_URLS = [
    f"{_TEST_DATA_PROJECT_BASE}/-/archive/{TIAMAT_TEST_DATA_COMMIT}/tiamat-test-data-{TIAMAT_TEST_DATA_COMMIT}.tar.gz",
    f"{_TEST_DATA_PROJECT_BASE}/-/archive/{TIAMAT_TEST_DATA_COMMIT}/tiamat-test-data-{TIAMAT_TEST_DATA_COMMIT}.tar",
    f"{_TEST_DATA_PROJECT_BASE}/-/archive/{TIAMAT_TEST_DATA_COMMIT}/tiamat-test-data-{TIAMAT_TEST_DATA_COMMIT}.zip",
]

_TESTDATA_CACHE = {}  # commit -> (TemporaryDirectory, extracted_root_path)


def _http_download(url: str, dst: Path, timeout_s: int = 60) -> None:
    """
    Download URL to dst using urllib (pure Python).
    Raises on HTTP errors.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    req = Request(
        url,
        headers={
            # A UA helps with some proxies / WAFs.
            "User-Agent": "tiamat-tests/1.0 (python urllib)",
        },
        method="GET",
    )

    with urlopen(req, timeout=timeout_s) as resp:
        # urlopen raises for many non-2xx statuses, but be explicit.
        status = getattr(resp, "status", 200)
        if status and status >= 400:
            raise HTTPError(url, status, f"HTTP {status}", hdrs=resp.headers, fp=None)

        with open(dst, "wb") as f:
            shutil.copyfileobj(resp, f)


def _extract_archive(archive_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
        return

    # tarfile can auto-detect compression with mode "r:*"
    with tarfile.open(archive_path, mode="r:*") as tf:
        tf.extractall(extract_to)


def _get_testdata_root(commit: str) -> Path:
    cached = _TESTDATA_CACHE.get(commit)
    if cached is not None:
        return cached[1]

    tmp = tempfile.TemporaryDirectory(prefix=f"tiamat-testdata-{commit[:8]}-")
    tmp_path = Path(tmp.name)

    archive_path = tmp_path / "archive"
    extracted_path = tmp_path / "extracted"
    extracted_path.mkdir(parents=True, exist_ok=True)

    last_err = None
    for url in _TEST_DATA_ARCHIVE_URLS:
        try:
            _http_download(url, archive_path)
            _extract_archive(archive_path, extracted_path)
            last_err = None
            break
        except (HTTPError, URLError, tarfile.TarError, zipfile.BadZipFile, OSError) as e:
            last_err = e
            continue

    if last_err is not None:
        raise RuntimeError(
            "Failed to download/extract tiamat test-data archive " f"for commit {commit}. Last error: {last_err}"
        )

    # GitLab archives typically extract into a single top-level directory.
    extracted_dirs = [p for p in extracted_path.iterdir() if p.is_dir()]
    if len(extracted_dirs) != 1:
        raise RuntimeError(
            f"Unexpected archive layout in {extracted_path}: "
            f"expected 1 top-level directory, found {len(extracted_dirs)}."
        )

    root_dir = extracted_dirs[0]
    _TESTDATA_CACHE[commit] = (tmp, root_dir)  # keep tempdir alive for the session
    return root_dir


def _get_reference_dir(commit: str) -> Path:
    root_dir = _get_testdata_root(commit)
    ref_dir = root_dir / "tiamat" / "tests" / "e2e" / "references"
    if not ref_dir.exists():
        raise RuntimeError(
            f"Reference directory not found at expected path: {ref_dir}. "
            "Please verify the archive structure and/or adjust the path."
        )
    return ref_dir


# -----------------------------------------------------------------------------
# Original test code below (unchanged except reference-path plumbing)
# -----------------------------------------------------------------------------

image_width, image_height = 2560, 1600

access_frame = np.array(
    [
        [-image_width, image_width],
        [-image_height, image_height],
    ],
    dtype=int,
)


def build_affine(scale=1.0, rotation=0, mirror_x=False, mirror_y=False, translate=[0.0, 0.0]):
    angle_rad = np.deg2rad(rotation)

    cos_angle = np.cos(angle_rad) * scale
    sin_angle = np.sin(angle_rad) * scale

    # Build the affine transformation matrix
    rot_matrix = np.array([[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0.0, 0.0, 1.0]], dtype=np.float32)
    mirror_matrix = np.array(
        [[1 - 2 * float(mirror_x), 0, 0], [0, 1 - 2 * float(mirror_y), 0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    affine_matrix = rot_matrix @ mirror_matrix
    affine_matrix[:2, -1] = translate

    return affine_matrix


Arguments = namedtuple(
    "Arguments",
    ["scale", "rotation", "mirror_x", "mirror_y", "translate_x", "translate_y"],
    defaults=[1, 0, False, False, 0, 0],
)

test_args = {
    "iden": Arguments(),
    "scale_down": Arguments(scale=0.75),
    "scale_up": Arguments(scale=1.25),
    "rot_cw": Arguments(rotation=45),
    "rot_ccw": Arguments(rotation=-45),
    "rot_ccw_alt": Arguments(rotation=315),
    "mirror_x": Arguments(mirror_x=True),
    "mirror_y": Arguments(mirror_y=True),
    "mirror_xy": Arguments(mirror_x=True, mirror_y=True),
    "translate_-x": Arguments(translate_x=-500),
    "translate_+x": Arguments(translate_x=500),
    "translate_-y": Arguments(translate_y=-500),
    "translate_+y": Arguments(translate_y=500),
    "mix_1": Arguments(mirror_x=True, translate_x=2560),
    "mix_2": Arguments(mirror_y=True, translate_y=1600),
}


@pytest.mark.parametrize(
    "scale, rotation, mirror_x, mirror_y, translate_x, translate_y",
    test_args.values(),
    ids=test_args.keys(),
)
def test_affine_pipeline(scale, rotation, mirror_x, mirror_y, translate_x, translate_y, request):

    affine_matrix = build_affine(scale, rotation, mirror_x, mirror_y, (translate_x, translate_y))
    pipeline = Pipeline(
        transformers=[
            MinMaxNormalizationTransformer(),
            AffineTransformer(affine_matrix=affine_matrix),
        ],
    )
    result = pipeline(
        file_name="./examples/data/Koala.jpg",
        accessor=ImageAccessor(x=access_frame[0], y=access_frame[1], fill_value=0),
    )

    filename = f"test_affine_s{scale}_r{rotation}"
    if mirror_x:
        filename += "_mirrorX"
    if mirror_y:
        filename += "_mirrorY"
    filename += f"_t{translate_x}_{translate_y}"

    # Read expected reference from downloaded test-data repo (pinned by commit).
    reference_dir = _get_reference_dir(TIAMAT_TEST_DATA_COMMIT)
    npygz_filename = reference_dir / f"{request.node.callspec.id}--{filename}.npy.gz"

    # Use a temp file for the intermediate .npy to avoid writing into the working tree.
    with tempfile.TemporaryDirectory(prefix="tiamat-expected-npy-") as _tmp_expected:
        npy_filename = Path(_tmp_expected) / f"{request.node.callspec.id}--{filename}.npy"

        dst_png_filename = f"artefacts/tests/e2e/references/{request.node.callspec.id}--{filename}.png"
        dst_npy_filename = f"artefacts/tests/e2e/references/{request.node.callspec.id}--{filename}.npy"
        dst_npygz_filename = f"artefacts/tests/e2e/references/{request.node.callspec.id}--{filename}.npy.gz"

        uint8_img = np.array(result.image * 255, dtype=np.uint8)

        Path(dst_png_filename).parent.mkdir(exist_ok=True, parents=True)
        np.save(dst_npy_filename, uint8_img, False)
        with gzip.open(dst_npygz_filename, "wb") as gzip_file:
            with open(dst_npy_filename, "rb") as npy_file:
                shutil.copyfileobj(npy_file, gzip_file)
        Path(dst_npy_filename).unlink()
        imwrite(dst_png_filename, uint8_img)

        with gzip.open(npygz_filename, "rb") as gzip_file:
            with open(npy_filename, "wb") as npy_file:
                shutil.copyfileobj(gzip_file, npy_file)
        expected_arr = np.load(npy_filename)
        np.testing.assert_equal(uint8_img, expected_arr)
        Path(npy_filename).unlink()
