Generator Reference
===================

This page documents the primary generator classes and the diffuse-parameter
interface.

SONGS Class
-----------

.. autoclass:: songs.core.SONGS
   :members:
   :undoc-members:
   :show-inheritance:

SONGSPhy Class
--------------

.. autoclass:: songs.core.SONGSPhy
   :members:
   :undoc-members:
   :show-inheritance:

DEFAULT_DIFFUSE_PARAMS
----------------------

All diffuse emission components are controlled through ``DEFAULT_DIFFUSE_PARAMS``
in ``src/songs/core.py``.  Any key can be overridden at construction time by
passing a ``diffuse_params`` dict; it is merged with (not replacing) the defaults.

**Halo** (around the central galaxy)

.. list-table::
   :header-rows: 1
   :widths: 32 15 53

   * - Key
     - Default
     - Meaning
   * - ``enabled``
     - ``True``
     - Master switch.  Set to ``False`` to suppress all diffuse components.
   * - ``halo_Re_factor``
     - ``3.0``
     - Effective radius of the halo as a multiple of the central galaxy's :math:`R_e`.
   * - ``halo_Se_factor``
     - ``0.065``
     - Peak amplitude of the halo as a fraction of the central disk's :math:`S_e`.
   * - ``halo_hz_factor``
     - ``2.0``
     - Scale height of the halo as a multiple of the central galaxy's :math:`h_z`.
   * - ``halo_n``
     - ``0.5``
     - Sérsic index for the halo radial profile (Gaussian-like at 0.5).
   * - ``halo_sigma_vz``
     - ``70.0``
     - Per-voxel LOS velocity dispersion of the halo (km/s).

**Bridges** (between central and each satellite)

.. list-table::
   :header-rows: 1
   :widths: 32 15 53

   * - Key
     - Default
     - Meaning
   * - ``bridge_start_frac``
     - ``0.2``
     - Fractional position along the central→satellite link where the bridge
       emerges (avoids overlap with the central core).
   * - ``bridge_stop_frac``
     - ``0.0``
     - Fractional distance from the satellite end at which the bridge stops
       (0 = reaches the satellite).
   * - ``bridge_width_start_factor``
     - ``3.0``
     - Gaussian :math:`\sigma` at the halo end, in multiples of the central
       galaxy's :math:`R_e`.
   * - ``bridge_width_end_factor``
     - ``2.0``
     - Gaussian :math:`\sigma` at the satellite end, in multiples of the
       satellite's :math:`R_e`.
   * - ``bridge_edge_fade``
     - ``0.1``
     - Fraction of the active bridge length over which a smooth taper is
       applied at each end.
   * - ``bridge_Se_factor``
     - ``0.03``
     - Peak on-axis amplitude as a fraction of the fainter of the two galaxy
       :math:`S_e` values.

**Streamers / Tidal Tails** (Bézier arcs extending from each satellite)

Streamers are modelled as curved 3-D Gaussian tubes along a random Bézier
curve, giving each tidal tail a physically motivated arc shape.  The velocity
gradient along the tail produces a pronounced kinematic signature in
channel-by-channel viewers.

.. list-table::
   :header-rows: 1
   :widths: 32 15 53

   * - Key
     - Default
     - Meaning
   * - ``tail_length_factor``
     - ``0.25``
     - Arc length of the tail as a fraction of the central-to-satellite
       separation.
   * - ``tail_curvature``
     - ``0.15``
     - Overall perpendicular offset of the tail tip, in multiples of the
       separation distance.
   * - ``tail_width_factor``
     - ``1.0``
     - Base Gaussian :math:`\sigma` of the tube, in multiples of the
       satellite's :math:`R_e`.
   * - ``tail_Se_factor``
     - ``0.15``
     - Peak amplitude at the tail root as a fraction of the satellite's
       :math:`S_e`.
   * - ``tail_vel_gradient``
     - ``0.5``
     - Dimensionless velocity gradient along the tail:
       :math:`v(u) = v_\mathrm{sat} + u \cdot (v_\mathrm{sat} - v_\mathrm{central}) \cdot f`.
   * - ``tail_sigma_vz``
     - ``100.0``
     - Per-voxel LOS velocity dispersion along the tail (km/s).
   * - ``tail_n_samples``
     - ``40``
     - Number of Gaussian blobs sampled along the Bézier curve.
   * - ``tail_n_control_points``
     - ``4``
     - Number of Bézier control points defining the tail shape.
   * - ``tail_jitter``
     - ``0.1``
     - Perpendicular jitter of interior control points, in multiples of the
       separation distance, giving each tail a unique random shape.
   * - ``tail_decay_scale``
     - ``1.5``
     - Exponential brightness decay scale in arc-length units
       (:math:`\propto \exp(-u/\lambda)`).  Smaller values concentrate
       emission near the satellite.

Multi-Galaxy Placement
-----------------------

When ``n_gals`` is set to an integer greater than one, SONGS generates one central
galaxy and ``n_gals - 1`` satellites.

- **Separation constraint**: satellite centres are placed at least
  ``offset_gals`` pixels from the central galaxy and from one another to avoid
  unphysical overlap.
- **Velocity offset**: each satellite receives an independent systemic velocity
  drawn from a uniform distribution so that kinematic separation is visible in
  the spectral axis.
- The diffuse components (halo, bridges, tails) are constructed *after* all disk
  components have been generated so that they can reference the finalised
  per-galaxy positions and velocities.
