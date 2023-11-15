# tiamat - Tiled Image Access, Manipulation, and Analysis Toolkit

__Author:__ Christian Schiffer (c.schiffer@fz-juelich.de), Institute for Neuroscience and Medicine (INM-1), Forschungszentrum Jülich

__Acknowledgement:__
This project has received funding from the Helmholtz Association’s Initiative and Networking Fund through the Helmholtz International BigBrain Analytics and Learning Laboratory (HIBALL) under the Helmholtz International Lab grant agreement InterLabs-0015.

The future of image service is now.

## MVP

- tiamat
    - readers for: nifti, tiff, png/jpeg/..., hdf5 pyramid, precomputed
    - transformers: LUTTransformer 
- tiamat-ng 
- tiamat-dzi
- docker: tiamat-ng, tiamat-dzi + reverse proxy

### Showcase

```bash
docker-compose up -d tiamat-dzi
# navigate to ime262.ime.kfa-juelich.de:3000/microdraw.html?source=http://ime262.ime.kfa-juelich.de/legacy_json/tiamat.json
# http://ime262.ime.kfa-juelich.de/legacy_json/tiamat.json points to tiamat-dzi
```

## Coding Style

- Use type hinting
- Use google docstring style
- flake8 formatting (120)
- unit tests (pytest)
- main branch is locked, modifications through MRs from feature branches
