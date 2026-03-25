"""
Reader for OME-Zarr (Zarr v3 / OME-NGFF >=0.5).
"""

from __future__ import annotations

import json
import os
import zipfile
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np

try:
    import tensorstore as ts  # pylint: disable=import-error

    _TENSORSTORE_AVAILABLE = True
except Exception:  # pragma: no cover
    ts = None  # pylint: disable=import-error
    _TENSORSTORE_AVAILABLE = False

from tiamat.cache import instance_cache
from tiamat.io import ImageAccessor
from tiamat.metadata import ImageMetadata
from tiamat.readers.protocol import ImageReader


# retrieved from https://ngff.openmicroscopy.org/specifications/0.5/index.html#axes-metadata
# on 2026-03-05

class SiUnit(NamedTuple):
    symbol: str
    exponent: int
    prefix: str


base_to_si_units = [
    SiUnit(**d)
    for d in [
        {"symbol": "Y", "exponent": 24, "prefix": "yotta"},
        {"symbol": "Z", "exponent": 21, "prefix": "zetta"},
        {"symbol": "E", "exponent": 18, "prefix": "exa"},
        {"symbol": "P", "exponent": 15, "prefix": "peta"},
        {"symbol": "T", "exponent": 12, "prefix": "tera"},
        {"symbol": "G", "exponent": 9, "prefix": "giga"},
        {"symbol": "M", "exponent": 6, "prefix": "mega"},
        {"symbol": "k", "exponent": 3, "prefix": "kilo"},
        {"symbol": "", "exponent": 0, "prefix": ""},
        {"symbol": "m", "exponent": -3, "prefix": "milli"},
        {"symbol": "µ", "exponent": -6, "prefix": "micro"},
        {"symbol": "u", "exponent": -6, "prefix": "micro"},
        {"symbol": "n", "exponent": -9, "prefix": "nano"},
        {"symbol": "p", "exponent": -12, "prefix": "pico"},
        {"symbol": "f", "exponent": -15, "prefix": "femto"},
        {"symbol": "a", "exponent": -18, "prefix": "atto"},
        {"symbol": "z", "exponent": -21, "prefix": "zepto"},
        {"symbol": "y", "exponent": -24, "prefix": "yocto"},
        {"symbol": "h", "exponent": 2, "prefix": "hecto"},
        {"symbol": "d", "exponent": -1, "prefix": "deci"},
        {"symbol": "c", "exponent": -2, "prefix": "centi"},
    ]
]

meter_to_spatial_units = {
    "angstrom": 1e-10,
    "foot": 0.3048,
    "inch": 0.0254,
    "mile": 1609.34,
    "parsec": 3.0856775814913673e16,
    "yard": 0.9144,
}

spatial_units_to_micron = {
    **{unit: 1e6 / value for unit, value in meter_to_spatial_units.items()},
    **{f"{v.prefix}meter": 10 ** (6 + v.exponent) for v in base_to_si_units},
}


def _axis_to_dim(axis_name: str, md) -> Any:
    """
    Map ome-zarr specific axis labels to tiamat dimensions.
    """
    n = axis_name.lower()
    return {
        "x": md.dimensions.X,
        "y": md.dimensions.Y,
        "z": md.dimensions.Z,
        "c": md.dimensions.C,
        "t": md.dimensions.T,
    }.get(n, md.dimensions.C)


class _TensorStoreArrayAdapter:
    """
    Small ndarray-like wrapper around a TensorStore array handle.
    """

    def __init__(self, store):
        self._store = store

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._store.dtype.numpy_dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self._store.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, item):
        view = self._store[item]
        if hasattr(view, "read"):
            return np.asarray(view.read().result())
        return np.asarray(view)


class OmeZarrReader(ImageReader):
    def __init__(self, fname: str):
        self.fname = fname

    @classmethod
    def check_file(cls, fname) -> bool | int | float:
        if not _TENSORSTORE_AVAILABLE:
            return False

        lower = str(fname).lower()
        is_zip = lower.endswith(".zarr.zip") or lower.endswith(".ome.zarr.zip")
        is_dir_like = lower.endswith(".zarr") or lower.endswith(".ome.zarr")

        if not (is_zip or is_dir_like):
            return False

        def load_zarr_json(json_bytes: bytes) -> dict | None:
            try:
                return json.loads(json_bytes)
            except Exception:
                return None

        zarr_data = None

        if is_zip:
            try:
                with zipfile.ZipFile(fname, "r") as zf:
                    if "zarr.json" not in zf.namelist():
                        return False
                    with zf.open("zarr.json") as f:
                        zarr_data = load_zarr_json(f.read())
            except Exception:
                return False

        elif is_dir_like:
            zarr_json_path = os.path.join(fname, "zarr.json")
            if not os.path.exists(zarr_json_path):
                return False
            try:
                with open(zarr_json_path, "r", encoding="utf-8") as f:
                    zarr_data = json.load(f)
            except Exception:
                return False

        if not isinstance(zarr_data, dict):
            return False

        attributes = zarr_data.get("attributes")
        if not isinstance(attributes, dict):
            return False

        if "ome" in attributes:
            return 10
        return False

    def read_image(self, accessor: ImageAccessor) -> np.ndarray:
        from tiamat.readers.processing import access_and_rescale_image

        axes = self.axes_names
        raw_scale = accessor.scale
        if isinstance(raw_scale, (float, int)):
            target_xy = (float(raw_scale), float(raw_scale))
            target_z = float(raw_scale)
        else:
            seq = list(raw_scale)
            if len(seq) >= 3:
                target_z = float(seq[0])
                target_xy = (float(seq[1]), float(seq[2]))
            elif len(seq) == 2:
                target_z = float(min(seq))
                target_xy = (float(seq[0]), float(seq[1]))
            else:
                target_xy = (float(seq[0]), float(seq[0]))
                target_z = float(seq[0])

        per_axis_target = []
        for axis_name in axes:
            if axis_name == "x":
                per_axis_target.append(target_xy[1])
            elif axis_name == "y":
                per_axis_target.append(target_xy[0])
            elif axis_name == "z":
                per_axis_target.append(target_z)
            else:
                per_axis_target.append(1.0)

        chosen_level = self._select_level_from_target(per_axis_target)
        level_factors = self._level_factors(chosen_level)
        arr = self._get_level_array(chosen_level)

        return access_and_rescale_image(
            image=arr,
            metadata=self.read_metadata(),
            accessor=accessor,
            image_scale=tuple(level_factors),
        )

    @instance_cache
    def read_metadata(self) -> ImageMetadata:
        from tiamat import metadata as md

        dims = [_axis_to_dim(ax, md) for ax in self.axes_names]
        vmin, vmax = self.value_range

        return md.ImageMetadata(
            image_type=md.IMAGE_TYPE_IMAGE,
            shape=self.shape,
            dtype=self.dtype,
            file_path=self.fname,
            value_range=(vmin, vmax),
            spacing=self.pixel_spacing,
            dimensions=dims,
            scales=self.scales,
            additional_metadata=self._root_attrs,
        )

    @cached_property
    def _kvstore_spec(self) -> dict[str, Any]:
        lower = self.fname.lower()
        if lower.endswith(".zip"):
            return {
                "driver": "zip",
                "base": {
                    "driver": "file",
                    "path": self.fname,
                },
            }

        path = self.fname
        if os.path.isdir(path) and not path.endswith(os.sep):
            path += os.sep

        return {
            "driver": "file",
            "path": path,
        }

    @cached_property
    def _kvstore(self):
        if not _TENSORSTORE_AVAILABLE:
            raise ImportError(
                "tensorstore is not installed. Install `tensorstore` to read OME-Zarr data."
            )
        return ts.KvStore.open(self._kvstore_spec).result()

    @cached_property
    def _root_attrs(self) -> dict[str, Any]:
        result = self._kvstore.read("zarr.json").result()
        if getattr(result, "state", None) != "value":
            raise ValueError("OME-Zarr root metadata 'zarr.json' is missing.")

        try:
            root_meta = json.loads(result.value)
        except Exception as exc:  # pragma: no cover
            raise ValueError("OME-Zarr root metadata 'zarr.json' is invalid JSON.") from exc

        attrs = root_meta.get("attributes")
        if not isinstance(attrs, dict):
            raise ValueError("OME-Zarr root metadata is missing 'attributes'.")
        return attrs

    @cached_property
    def _ome_meta(self) -> dict[str, Any]:
        ome = self._root_attrs.get("ome")
        if isinstance(ome, dict) and ome:
            return ome
        return self._root_attrs

    @cached_property
    def _multiscales(self) -> list[dict[str, Any]]:
        multiscales = self._ome_meta.get("multiscales")
        if not isinstance(multiscales, list) or not multiscales:
            raise ValueError("OME-Zarr attributes lack a valid 'multiscales' list.")
        return multiscales

    @cached_property
    def _pyramid(self) -> dict[str, Any]:
        return self._multiscales[0]

    @cached_property
    def axes_names(self) -> list[str]:
        axes = self._pyramid.get("axes")
        if isinstance(axes, list) and axes:
            if isinstance(axes[0], str):
                return [axis_name.lower() for axis_name in axes]
            if isinstance(axes[0], dict):
                return [str(axis.get("name", "")).lower() for axis in axes]

        a0 = self._get_level_array(0)
        rank = a0.ndim
        if rank == 2:
            return ["y", "x"]
        if rank == 3:
            return ["c", "y", "x"]
        if rank == 4:
            return ["z", "c", "y", "x"]
        if rank >= 5:
            return ["t", "z", "c", "y", "x"][-rank:]
        return ["y", "x"]

    @cached_property
    def _datasets(self) -> list[dict[str, Any]]:
        datasets = self._pyramid.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("OME-Zarr multiscales.datasets is missing or empty.")
        return datasets

    @cached_property
    def dtype(self) -> np.dtype:
        return self._get_level_array(0).dtype

    @cached_property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self._get_level_array(0).shape)

    @cached_property
    def value_range(self) -> tuple[float | int, float | int]:
        dtype = self.dtype
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
        elif np.issubdtype(dtype, np.floating):
            info = np.finfo(dtype)
        else:
            raise TypeError(f"Cannot determine value range for dtype {dtype}")
        return info.min, info.max

    @cached_property
    def pixel_spacing(self) -> tuple[float, ...]:
        axes = self.axes_names

        mul_factor: list[float] = []
        for axis in self._pyramid.get("axes", []):
            unit = axis.get("unit")
            if factor := spatial_units_to_micron.get(unit):
                mul_factor.append(factor)
            else:
                raise RuntimeError(f"zarr unit {unit} either not recognized or factor not yet encoded.")

        transforms = self._datasets[0].get("coordinateTransformations", [])
        scale_vec = None
        for transform in transforms or []:
            if isinstance(transform, dict) and transform.get("type") == "scale":
                scale = transform.get("scale")
                if isinstance(scale, list) and len(scale) == len(axes) == len(mul_factor):
                    scale_vec = [float(x) * factor for x, factor in zip(scale, mul_factor)]
                    break
        if scale_vec is None:
            return tuple([1.0] * len(axes))
        return tuple(scale_vec)

    @instance_cache
    def _level_factors(self, level: int) -> list[float]:
        base = self._get_level_array(0).shape
        cur = self._get_level_array(level).shape
        factors = []
        spatial = {"z", "y", "x"}
        for i, axis_name in enumerate(self.axes_names):
            if axis_name in spatial:
                factors.append(float(cur[i]) / float(base[i]))
            else:
                factors.append(1.0)
        return factors

    @cached_property
    def scales(self) -> list[tuple[float, ...]]:
        factors = []
        for lvl in range(len(self._datasets)):
            factors.append(tuple(self._level_factors(lvl)[::-1]))
        return factors

    @instance_cache
    def _get_level_array(self, level: int):
        ds = self._datasets[level]
        rel = ds.get("path")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"Invalid dataset path at level {level}.")

        store = ts.open(
            {
                "driver": "zarr3",
                "kvstore": self._kvstore_spec,
                "path": rel,
            },
            read=True,
        ).result()
        return _TensorStoreArrayAdapter(store)

    @cached_property
    def _spatial_indices(self) -> list[int]:
        order = []
        for axis_name in ("z", "y", "x"):
            if axis_name in self.axes_names:
                order.append(self.axes_names.index(axis_name))
        return order

    def _select_level_from_target(self, per_axis_target: list[float]) -> int:
        spatial_idx = self._spatial_indices

        def meets_target(factors: list[float]) -> bool:
            return all(factors[i] >= per_axis_target[i] for i in spatial_idx)

        candidates = []
        for lvl in range(len(self._datasets)):
            factors = self._level_factors(lvl)
            if meets_target(factors):
                score = max((factors[i] for i in spatial_idx), default=1.0)
                candidates.append((score, lvl))

        if candidates:
            candidates.sort(key=lambda t: t[0])
            return candidates[0][1]

        coarsest = 0
        coarsest_score = -np.inf
        for lvl in range(len(self._datasets)):
            factors = self._level_factors(lvl)
            score = max((factors[i] for i in spatial_idx), default=1.0)
            if score > coarsest_score:
                coarsest_score = score
                coarsest = lvl
        return coarsest
