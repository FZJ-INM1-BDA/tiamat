.. title:: tiamat

======================================================
Tiled Image Access, Manipulation, and Analysis Toolkit
======================================================


.. grid::

    .. grid-item-card:: :fas:`lightbulb` Core Concepts
        :link: core-concepts
        :link-type: ref
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Explanations on how tiamat works

    .. grid-item-card:: :fas:`rocket` Quickstart
        :link: quickstart
        :link-type: ref
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Get started with tiamat!
        
    .. grid-item-card:: :fas:`book` Examples :octicon:`link-external`
        :link: https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat/tiamat/-/tree/master/examples?ref_type=heads
        :link-type: url
        :columns: 12 12 4 4
        :class-card: sd-shadow-md
        :class-title: sd-text-primary
        :margin: 2 2 0 0

        Find and interactively run tiamat examples here!

Imagine a gigantic high-resolution image — a full human brain scanned at micrometer resolution, or a multi-band 
satellite image spanning terabytes. You want to view a tiny region, pan and zoom in a web viewer, or run a segmentation 
model on a handful of patches. Traditionally you’d convert the original dataset into the viewer’s favorite layout 
(tile pyramids, precomputed volumes) and copy terabytes of data around. That costs time, storage, and patience.

**tiamat** is the “middle-layer magician” that says: don’t copy the data; transform it on demand. It’s a Python toolkit 
that composes three things — readers, transformers, and interfaces — into pipelines that lazily read only the pixels 
you ask for, apply transformations (color maps, normalizations, affine re-projections, even model inference), and 
stream the result to whatever client you use (web viewers like Neuroglancer/OpenSeadragon, Napari, FUSE mounts, 
or plain Python scripts). This lets you serve and explore enormous datasets directly from their native storage while 
keeping transformations reproducible and versionable.

Why this matters:

* **No copies** — you avoid generating extra terabytes just so a tool can be happy.
* **On-demand transformations** — only compute what the user requests (tile, pyramid level, channel subset).
* **Multiple clients** — the same backend can serve a Napari plugin, Neuroglancer, or an OpenSeadragon web app.

tiamat is intentionally not a viewer; it is the flexible data layer beneath viewers and analysis tools. Think of it as 
the friendly translator between your storage and your tools.

.. toctree::
    :hidden:
    :caption: Overview
    
    core-concepts
    installation

.. toctree::
    :hidden:
    :caption: How to Guides

    quickstart
    performance
    contribute
    faq

.. toctree::
    :hidden:
    :caption: Reference

    JuGit Repository <https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat>
    acknowledgements

