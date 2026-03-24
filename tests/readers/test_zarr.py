import json
import zipfile
from pathlib import Path

import numpy as np

from tiamat.readers.zarr import OmeZarrReader
import tiamat.readers.zarr as zarr_reader


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
    assert metadata.additional_metadata == payload["attributes"]

    level0 = reader._get_level_array(0)
    assert level0.shape == (4, 5)
    assert level0.ndim == 2
    assert level0.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(level0[1:3, 2:5], np.array([[7, 8, 9], [12, 13, 14]], dtype=np.uint16))

    assert fake_ts.kvstore_specs == [{"driver": "file", "path": "/tmp/sample.ome.zarr"}]
    assert fake_ts.open_specs == [
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": "/tmp/sample.ome.zarr"},
            "path": "0",
        }
    ]
