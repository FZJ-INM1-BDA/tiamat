import json
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import tiamat.readers.zarr as zarr_reader
from tiamat.readers.zarr import OmeZarrReader

zarrcontent = {
    "attributes": {
        "ome": {
            "multiscales": [
                {
                    "axes": [
                        {"name": "x", "type": "space", "unit": "nanometer"},
                        {"name": "y", "type": "space", "unit": "nanometer"},
                        {"name": "z", "type": "space", "unit": "nanometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [{"scale": [1000.0, 1000.0, 1000.0], "type": "scale"}],
                        },
                        {
                            "path": "1",
                            "coordinateTransformations": [{"scale": [2000.0, 2000.0, 2000.0], "type": "scale"}],
                        },
                    ],
                    "coordinateTransformations": [],
                    "name": "multiresolution",
                    "type": "unknown",
                    "metadata": {},
                }
            ],
            "version": "0.5",
        }
    },
    "zarr_format": 3,
    "node_type": "group",
}

z0content = {
    "shape": [6000, 6000, 6000],
    "data_type": "uint8",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [6016, 6016, 6016]}},
    "chunk_key_encoding": {"configuration": {"separator": "."}, "name": "default"},
    "fill_value": 0,
    "codecs": [
        {
            "configuration": {
                "chunk_shape": [64, 64, 64],
                "codecs": [
                    {"configuration": {"endian": "little"}, "name": "bytes"},
                    {"configuration": {"level": 9}, "name": "gzip"},
                ],
                "index_codecs": [{"configuration": {"endian": "little"}, "name": "bytes"}],
                "index_location": "start",
            },
            "name": "sharding_indexed",
        }
    ],
    "dimension_names": ["x", "y", "z"],
    "zarr_format": 3,
    "node_type": "array",
}

z1content = {
    "shape": [3000, 3000, 3000],
    "data_type": "uint8",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3008, 3008, 3008]}},
    "chunk_key_encoding": {"configuration": {"separator": "."}, "name": "default"},
    "fill_value": 0,
    "codecs": [
        {
            "configuration": {
                "chunk_shape": [64, 64, 64],
                "codecs": [
                    {"configuration": {"endian": "little"}, "name": "bytes"},
                    {"configuration": {"level": 9}, "name": "gzip"},
                ],
                "index_codecs": [{"configuration": {"endian": "little"}, "name": "bytes"}],
                "index_location": "start",
            },
            "name": "sharding_indexed",
        }
    ],
    "dimension_names": ["x", "y", "z"],
    "zarr_format": 3,
    "node_type": "array",
}


@pytest.fixture
def micron_zarrfile():
    with TemporaryDirectory(suffix=".ome.zarr") as dir:
        _zarrcontent = deepcopy(zarrcontent)
        for ms in _zarrcontent["attributes"]["ome"]["multiscales"]:
            for axis in ms["axes"]:
                axis["unit"] = "micrometer"
        (Path(dir) / "zarr.json").write_text(json.dumps(_zarrcontent))
        (Path(dir) / "0").mkdir()
        (Path(dir) / "1").mkdir()
        (Path(dir) / "0" / "zarr.json").write_text(json.dumps(z0content))
        (Path(dir) / "1" / "zarr.json").write_text(json.dumps(z1content))
        yield dir


@pytest.fixture
def pico_zarrfile():
    with TemporaryDirectory(suffix=".ome.zarr") as dir:
        _zarrcontent = deepcopy(zarrcontent)
        for ms in _zarrcontent["attributes"]["ome"]["multiscales"]:
            for axis in ms["axes"]:
                axis["unit"] = "picometer"
        (Path(dir) / "zarr.json").write_text(json.dumps(_zarrcontent))
        (Path(dir) / "0").mkdir()
        (Path(dir) / "1").mkdir()
        (Path(dir) / "0" / "zarr.json").write_text(json.dumps(z0content))
        (Path(dir) / "1" / "zarr.json").write_text(json.dumps(z1content))
        yield dir


@pytest.fixture
def nano_zarrfile():
    with TemporaryDirectory(suffix=".ome.zarr") as dir:
        (Path(dir) / "zarr.json").write_text(json.dumps(zarrcontent))
        (Path(dir) / "0").mkdir()
        (Path(dir) / "1").mkdir()
        (Path(dir) / "0" / "zarr.json").write_text(json.dumps(z0content))
        (Path(dir) / "1" / "zarr.json").write_text(json.dumps(z1content))
        yield dir


@pytest.mark.skipif(sys.version_info < (3, 11), reason="omezarr require 3.11 or above")
def test_nanospacing(nano_zarrfile):
    reader = OmeZarrReader(nano_zarrfile)
    metadata = reader.read_metadata()
    assert metadata.spacing == (1, 1, 1)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="omezarr require 3.11 or above")
def test_microspacing(micron_zarrfile):
    reader = OmeZarrReader(micron_zarrfile)
    metadata = reader.read_metadata()
    assert metadata.spacing == (1e3, 1e3, 1e3)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="omezarr require 3.11 or above")
def test_picospacing(pico_zarrfile):
    reader = OmeZarrReader(pico_zarrfile)
    metadata = reader.read_metadata()
    assert metadata.spacing == (1e-3, 1e-3, 1e-3)


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="check < 3.11 results in import error")
def test_importerr(nano_zarrfile):
    with pytest.raises(ImportError):
        reader = OmeZarrReader(nano_zarrfile)
        metadata = reader.read_metadata()


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakeReadResult:
    def __init__(self, value: bytes, state: str = "value"):
        self.value = value
        self.state = state


class _FakeView:
    def __init__(self, data):
        self._data = data

    def read(self):
        return _FakeFuture(self._data)


class _FakeDType:
    def __init__(self, dtype):
        self.numpy_dtype = np.dtype(dtype)


class _FakeTensorStore:
    def __init__(self, data):
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = _FakeDType(self._data.dtype)

    def __getitem__(self, item):
        return _FakeView(self._data[item])


class _FakeKvStore:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self, key: str):
        if key != "zarr.json":
            return _FakeFuture(_FakeReadResult(b"", state="missing"))
        return _FakeFuture(_FakeReadResult(json.dumps(self._payload).encode("utf-8")))


class _FakeTensorStoreModule:
    def __init__(self, payload: dict, arrays: dict[str, np.ndarray]):
        self._payload = payload
        self._arrays = arrays
        self.kvstore_specs = []
        self.open_specs = []
        self.KvStore = self._make_kvstore_api()

    def _make_kvstore_api(self):
        parent = self

        class _FakeKvStoreApi:
            @staticmethod
            def open(spec):
                parent.kvstore_specs.append(spec)
                return _FakeFuture(_FakeKvStore(parent._payload))

        return _FakeKvStoreApi

    def open(self, spec, read=True):
        assert read is True
        self.open_specs.append(spec)
        return _FakeFuture(_FakeTensorStore(self._arrays[spec["path"]]))


def _write_ome_zarr_json(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "multiscales": [
                    {
                        "axes": ["y", "x"],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [1.0, 1.0],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        },
    }
    (path / "zarr.json").write_text(json.dumps(payload), encoding="utf-8")


def test_check_file_accepts_ome_zarr_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(zarr_reader, "_TENSORSTORE_AVAILABLE", True)
    root = tmp_path / "sample.ome.zarr"
    _write_ome_zarr_json(root)

    assert OmeZarrReader.check_file(str(root)) == 10


def test_check_file_accepts_ome_zarr_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(zarr_reader, "_TENSORSTORE_AVAILABLE", True)
    source = tmp_path / "source.ome.zarr"
    _write_ome_zarr_json(source)
    archive = tmp_path / "sample.ome.zarr.zip"

    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(source / "zarr.json", arcname="zarr.json")

    assert OmeZarrReader.check_file(str(archive)) == 10


def test_reader_uses_tensorstore_for_metadata_and_slice_access(monkeypatch):
    payload = {
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "axes": [
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [1.0, 1.0],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    }
    fake_ts = _FakeTensorStoreModule(
        payload=payload,
        arrays={"0": np.arange(20, dtype=np.uint16).reshape(4, 5)},
    )

    monkeypatch.setattr(zarr_reader, "ts", fake_ts)
    monkeypatch.setattr(zarr_reader, "_TENSORSTORE_AVAILABLE", True)
    monkeypatch.setattr(zarr_reader.os.path, "exists", lambda _: True)
    monkeypatch.setattr(zarr_reader.os.path, "isdir", lambda _: True)

    reader = OmeZarrReader("/tmp/sample.ome.zarr")
    metadata = reader.read_metadata()

    assert metadata.shape == (4, 5)
    assert metadata.dtype == np.dtype(np.uint16)
    assert metadata.dimensions == ["y", "x"]
    assert metadata.spacing == (1.0, 1.0)
    assert metadata.scales == [(1.0, 1.0)]
    assert metadata.additional_metadata == payload["attributes"]

    level0 = reader._get_level_array(0)
    assert level0.shape == (4, 5)
    assert level0.ndim == 2
    assert level0.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(level0[1:3, 2:5], np.array([[7, 8, 9], [12, 13, 14]], dtype=np.uint16))

    assert fake_ts.kvstore_specs == [{"driver": "file", "path": "/tmp/sample.ome.zarr/"}]
    assert fake_ts.open_specs == [
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": "/tmp/sample.ome.zarr/"},
            "path": "0",
        }
    ]


def test_reader_uses_multiscales_as_scale_source_of_truth(monkeypatch):
    payload = {
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "axes": [
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [1.0, 1.0],
                                    }
                                ],
                            },
                            {
                                "path": "1",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [4.0, 2.0],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        }
    }
    fake_ts = _FakeTensorStoreModule(
        payload=payload,
        arrays={
            "0": np.arange(24, dtype=np.uint16).reshape(4, 6),
            "1": np.arange(16, dtype=np.uint16).reshape(2, 8),
        },
    )

    monkeypatch.setattr(zarr_reader, "ts", fake_ts)
    monkeypatch.setattr(zarr_reader, "_TENSORSTORE_AVAILABLE", True)
    monkeypatch.setattr(zarr_reader.os.path, "exists", lambda _: True)
    monkeypatch.setattr(zarr_reader.os.path, "isdir", lambda _: True)

    reader = OmeZarrReader("/tmp/sample.ome.zarr")
    metadata = reader.read_metadata()

    assert metadata.spacing == (1.0, 1.0)
    assert metadata.scales == [(1.0, 1.0), (0.5, 0.25)]
    assert reader._level_factors(1) == [0.25, 0.5]
