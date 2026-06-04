Installation
============

PyPI (recommended)
------------------

Install the latest stable release with pip::

    pip install songs

Source install (developer mode)
--------------------------------

Clone the repository and install in editable mode::

    git clone https://github.com/arnablahiry/GalCubeCraft.git
    cd GalCubeCraft
    pip install -e .

Dependencies
------------

The following packages are required and will be installed automatically by pip:

- ``numpy`` — array operations and cube storage
- ``scipy`` — interpolation, Gaussian convolution helpers
- ``matplotlib`` — plotting and the interactive GUI backend
- ``astropy`` — ``Gaussian2DKernel`` for beam convolution, FITS I/O utilities
- ``torch`` — tensor operations for GPU-accelerated generation (optional but recommended)

Optional extras::

    pip install astrodendro   # dendrogram source finding

For GPU-accelerated PyTorch install the matching build for your CUDA version;
see https://pytorch.org for instructions.
