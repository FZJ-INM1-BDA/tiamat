<p align="center">
  <img src="https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat/-/raw/develop/assets/logo512.png" />
</p>

# tiamat - Tiled Image Access, Manipulation, and Analysis Toolkit

**Author:** Christian Schiffer (<c.schiffer@fz-juelich.de>), Institute for Neuroscience and Medicine (INM-1), Forschungszentrum Jülich

**Acknowledgement:**
This project has received funding from the Helmholtz Association’s Initiative and Networking Fund through the Helmholtz International BigBrain Analytics and Learning Laboratory (HIBALL) under the Helmholtz International Lab grant agreement InterLabs-0015.

_The future of image service is now._

## Concepts

`tiamat` uses a pipeline to model the flow of reading images and transformations.
A pipeline consists of `readers` for reading images and their metadata, and `transformers`, which apply transformations to images or the access (e.g., coordinates).
`readers` are implemented in `tiamat.readers`, according to a protocl defined in `tiamat.readers.protocol`.
`transformers` are implemented in `tiamat.transformers`, according to a protocol defined in `tiamat.transformers.protocl`.

## Examples

- Examples on how to use `tiamat` are given in [examples](./examples).

## Coding Style

- Use type hinting
- Use google docstring style
- flake8 formatting (120)
- unit tests (pytest)
- main branch is locked, modifications through MRs from feature branches

## Data

Data included in this repository (e.g., example or test data) requires `git-lfs`.

## Contributing

`tiamat` is developed according to [git-flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
New features are merged into developed and released regularly to master.
