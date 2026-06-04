GUI Reference
=============

.. automodule:: songs.gui
    :members:
    :undoc-members:
    :show-inheritance:

Overview
--------

The SONGS GUI (``src/songs/gui.py``) is a Tkinter-based interactive front end
for generating and inspecting synthetic spectral cubes.  It features a
**4-column dark-themed card layout** with the SONGS banner across the top.

Layout
------

Controls are grouped into four card columns:

1. **Initialisation Parameters** — grid size, number of spectral channels, number
   of galaxies, random seed, output directory.
2. **Central Galaxy Properties** — Sérsic index :math:`n`, effective radius
   :math:`R_e` (via resolution parameter *r*), scale height :math:`h_z`,
   central flux density :math:`S_e`, inclination, position angle.
3. **Satellite Properties** — satellite offset (pixels), satellite systemic
   velocity offset, beam parameters (bmin, bmaj, BPA).
4. **Diffuse Features** — interactive sliders and entry fields that map directly
   to ``DEFAULT_DIFFUSE_PARAMS``: halo amplitude (``halo_Se_factor``), bridge
   amplitude (``bridge_Se_factor``), streamer/tail amplitude
   (``tail_Se_factor``), tail velocity gradient (``tail_vel_gradient``), tail
   decay scale (``tail_decay_scale``), and the master ``enabled`` toggle.
   See :doc:`generator` for the full parameter reference.

All parameter changes take effect immediately when the user clicks **Generate**.

Launch Instructions
-------------------

From the package root::

    python -m songs.gui

Or directly::

    cd src/songs
    python gui.py

Generation runs in a background thread so the interface remains responsive.
Progress is streamed to the built-in Log window.

SliceViewer
-----------

After a cube has been generated the **Open SliceViewer** button launches the
built-in IFU SliceViewer:

- **Per-source sidebar** — select which galaxy component (central, satellite 1,
  satellite 2, …) to highlight; switching sources updates all overlays instantly.
- **Contour overlays** — per-source emission contours drawn at a configurable
  fraction of the peak channel intensity.
- **Bounding boxes** — axis-aligned bounding boxes around each detected source
  region for quick spatial extent assessment.
- **Intensity threshold masking** — a threshold slider suppresses low-surface-
  brightness residuals below a chosen fraction of the peak without re-generating
  the cube.

Visualisation Buttons
---------------------

Once generation has completed the following buttons become active:

- **Moment-0** — integrated intensity map (sum along spectral axis).
- **Moment-1** — intensity-weighted velocity map with beam marker overlay.
- **Spectrum** — integrated line-of-sight flux vs velocity.

These open interactive Matplotlib figures; use the standard toolbar to pan,
zoom, and save.

Saving
------

The **Save** button writes the current cube and parameter dictionary to disk
without re-running generation.  Supported formats: ``.npz`` (compressed NumPy
archive) and ``.pkl`` (Python pickle).  Saved contents include the spectral cube
array, beam information, pixel spatial scale, and average velocity values.
