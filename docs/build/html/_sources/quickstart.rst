Quick Start
===========

The snippet below generates one synthetic spectral cube, inspects its shape,
and visualises the results — all in roughly ten lines of Python.

.. code-block:: python

    import songs

    # Initialise the generator: one cube, 125×125 spatial grid, 40 spectral channels
    g = songs.SONGS(n_gals=None, n_cubes=1, grid_size=125, n_spectral_slices=40, seed=42)

    # Run the pipeline
    results = g.generate_cubes()

    # Unpack the first result
    cube, params = results[0]
    print("cube shape (n_vel, ny, nx):", cube.shape)   # e.g. (40, 125, 125)
    print("parameter keys:", list(params.keys()))

    # Visualise moment-0, moment-1, and the integrated spectrum
    g.visualise(results, idx=0, save=False)

Expected output::

    cube shape (n_vel, ny, nx): (40, 125, 125)
    parameter keys: ['average_vels', 'beam_info', 'pix_spatial_scale', ...]

The ``visualise`` call opens three interactive Matplotlib windows: a moment-0
(integrated intensity) map, a moment-1 (intensity-weighted velocity) map, and
the line-of-sight spectrum.

Cubes are also written to ``data/raw_data/<nz>x<ny>x<nx>/cube_*.npy`` by
default.  Set ``save=True`` in ``visualise`` to write PDF figures to
``figures/<shape>/``.
