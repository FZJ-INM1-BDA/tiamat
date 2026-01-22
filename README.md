<p align="center">
  <img src="https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat/-/raw/develop/assets/logo512.png" width="128" />
</p>

<h1 align="center">tiamat</h1>
<h3 align="center">Tiled Image Access, Manipulation, and Analysis Toolkit</h3>

---

**tiamat** is a modular Python toolkit for accessing, transforming, and exposing large scientific image datasets.
It provides a flexible, pluggable pipeline model that separates data access (readers), transformation (transformers), and delivery (interfaces) — allowing on-the-fly, tool-agnostic image workflows without data duplication or format conversion.

Supported outputs include NumPy arrays, Napari, Neuroglancer, OpenSeadragon, and FUSE-mounted virtual filesystems.

---

## 📑 Table of Contents

1. [Quick Start](#-quick-start)
2. [Installation](#-installation)
3. [Core Concepts](#-core-concepts)
4. [Examples](#-examples)
5. [Development Guidelines](#-development-guidelines)
6. [Contributing](#-contributing)
7. [Acknowledgements](#-acknowledgements)
8. [License](#-license)

---

## 🚀 Quick Start

```python
from tiamat import Pipeline
from tiamat.io import ImageAccessor
from tiamat.transformers import FractionalTransformer, LUTTransformer

# Create a pipeline with fractional coordinate access and a rainbow colormap
pipeline = Pipeline(
    access_transformers=[FractionalTransformer()],
    image_transformers=[LUTTransformer(colormap="rainbow")]
)

# Request the central 50% of the image
accessor = ImageAccessor(x=(0.25, 0.75), y=(0.25, 0.75))
result = pipeline("example_image.tif", accessor=accessor)

# Get the transformed NumPy image and metadata
image = result.image
metadata = result.metadata
```

---

## 📦 Installation

Install the latest release from pypi:

```bash
pip install tiamat-python
```

Install the latest development version directly from GitLab:

```bash
pip install git+https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat
```

---

## 🧠 Core Concepts

Tiamat defines a modular pipeline composed of:

- **Readers**: Load image data from formats like TIFF, NIfTI, HDF5, or memory arrays.
- **Transformers**: Apply dynamic, on-the-fly transformations (e.g., colormaps, axis reordering, tiling).
- **Interfaces**: Serve data to tools like Napari, Neuroglancer, OpenSeadragon, or directly as arrays.

This decoupled architecture allows you to:

- Build pipelines from reusable components
- Extend with custom readers or transformers
- Avoid costly format conversions

### Interfaces

Interfaces use `tiamat` to expose data to various client applications.

- **[tiamat-openseadragon](https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat-openseadragon)**: Interface compatible with [OpenSeadragon](https://openseadragon.github.io/).
- **[tiamat-ng](https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat-ng)**: Interface compatible with [Neuroglancer](https://github.com/google/neuroglancer).
- **[tiamat-fuse-zarr](https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat-fuse-zarr)**: Interface for exposing `tiamat` pipelines as zarr files through FUSE (experimental).
- **[tiamat-napari](https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat-napari)**: [Napari](https://napari.org/stable/) plugin interface for `tiamat` (experimental).

### Extension transormers and readers

- **[tiamat-justice](https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat-justice)**: Specialized readers and transformers used at INM-1, Forschungszentrum Jülich.
- **[tiamat-celldetection](https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat-celldetection)**: AI-based transformers for live [cell segmentation](https://celldetection.org).

---

## 📁 Examples

See the [`examples/`](./examples) directory for usage demonstrations and pipeline configurations.

---

## 🛠️ Development Guidelines

- Follow [PEP 561](https://peps.python.org/pep-0561/) type hinting
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Formatting: `flake8` with line length 120
- Tests: `pytest` unit tests
- Feature development follows `git-flow`

### Releases

Releases to [pypi](https://pypi.org/project/tiamat-python/) are automatically performed on semantic versioning tags on the master branch.

---

### 🤝 Contributing

We welcome contributions!

- Fork the repository and work on a feature branch.
- Submit a **Merge Request (MR) into `develop`**.
- All contributions are reviewed and tested before merging.

This project follows the [git-flow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
Releases are merged into `master` from `develop` on a regular basis.

---

## 📄 License

Apache 2.0 – see [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgements

See [ACKNOWLEDGEMENTS](./ACKNOWLEDGEMENTS).
