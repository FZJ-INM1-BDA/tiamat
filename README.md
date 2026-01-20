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
7. [Data](#-data)
8. [Acknowledgements](#-acknowledgements)
9. [Contributors](#-contributors)
10. [License](#-license)

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

### pypi publication

**Note:** Preliminary instructions until publication is automated.

1. Make sure the repository is clean (no unstaged changes).
2. Make sure your pypi or test.pypi credentials are configured in `~/.pypirc`.
3. Make sure the current commit has a version tag, e.g. `git tag v0.1.8`
4. `python -m build`
5. `twine check dist/*`
6. `python -m venv .venv-test; source .venv-test/bin/activate; pip install dist/*.whl; python -c "import tiamat; print(tiamat.__version__)"; deactive; rm -rf .venv-test`
7. `twine upload --repository testpypi dist/*`

---

## 🤝 Contributing

We welcome contributions!

- Fork the repository and work on a feature branch.
- Submit a **Merge Request (MR) into `develop`**.
- All contributions are reviewed and tested before merging.

This project follows the [git-flow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
Releases are merged into `master` from `develop` on a regular basis.

---

## 📂 Data

Some test and example datasets require `git-lfs` for download.

---

## 👥 Contributors

- Forschungszentrum Jülich, Institute of Neuroscience and Medicine (INM-1)
- Community contributors via GitLab merge requests

---

## 📄 License

Apache 2.0 – see [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgements

This project has received funding from the Helmholtz Association’s Initiative and Networking Fund through the Helmholtz International BigBrain Analytics and Learning Laboratory (HIBALL) under the Helmholtz International Lab grant agreement InterLabs-0015.
