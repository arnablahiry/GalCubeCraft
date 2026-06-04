Visualisation
=============

.. automodule:: songs.visualise
   :members:
   :undoc-members:
   :show-inheritance:

Helper Functions
----------------

The ``songs.visualise`` module provides lightweight wrappers around Matplotlib
for the most common cube inspection tasks.

moment0
~~~~~~~

.. code-block:: python

    from songs.visualise import moment0
    moment0(cube, save=False, fname=None)

Computes and displays the moment-0 (integrated intensity) map by summing the
cube along the spectral axis.  Optionally saves the figure to *fname* as a PDF.

moment1
~~~~~~~

.. code-block:: python

    from songs.visualise import moment1
    moment1(cube, vels, beam_info=None, save=False, fname=None)

Computes the intensity-weighted velocity map:

.. math::

   M_1(x, y) = \frac{\sum_v v \cdot S(v, x, y)}{\sum_v S(v, x, y)}

An optional beam marker (ellipse) is overlaid when ``beam_info`` is provided.

spectrum
~~~~~~~~

.. code-block:: python

    from songs.visualise import spectrum
    spectrum(cube, vels, save=False, fname=None)

Plots the spatially-integrated line-of-sight spectrum: flux summed over the
spatial axes as a function of velocity channel.

SliceViewer Class
-----------------

``SliceViewer`` provides an interactive channel-by-channel viewer launched
either from the GUI or standalone:

.. code-block:: python

    from songs.visualise import SliceViewer
    viewer = SliceViewer(cube, sources=sources_list, vels=vels)
    viewer.show()

**sources mode**: when ``sources`` is a list of per-source component cubes,
the viewer activates the per-source sidebar so that contours, bounding boxes
and threshold masks are computed independently for each source component.

Key interactive controls:

- Channel slider — step through spectral slices.
- Source selector sidebar — toggle which source's overlays are shown.
- Contour level spinner — set contour threshold as a fraction of the channel peak.
- Threshold slider — mask voxels below a chosen intensity fraction.
- Bounding-box toggle — show/hide axis-aligned bounding boxes per source.
