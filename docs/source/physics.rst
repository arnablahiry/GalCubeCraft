Physics & Mathematics
=====================

This page documents the mathematical models implemented in SONGS.

Sérsic Radial Profile (Disk Plane)
-----------------------------------

The surface brightness in the disk plane follows the Sérsic profile:

.. math::

   S_r(r) = S_e \exp\!\left[-b_n\!\left(\left(\frac{r}{R_e}\right)^{1/n} - 1\right)\right]

where

- :math:`S_e` is the flux density at the effective radius :math:`R_e`,
- :math:`n` is the Sérsic index (controls concentration),
- :math:`b_n` is a series-expansion constant that depends on :math:`n`:

.. math::

   b_n(n) \approx 2n - \tfrac{1}{3} + \frac{4}{405n} + \frac{46}{25515n^2} + \cdots

Vertical Exponential Profile
------------------------------

The vertical structure follows an exponential fall-off from the mid-plane:

.. math::

   S_z(z) = \exp\!\left(-\frac{|z|}{h_z}\right)

Combining the radial and vertical profiles gives the 3D flux density field:

.. math::

   S(x, y, z) = S_e
     \exp\!\left[-b_n\!\left(\left(\frac{r}{R_e}\right)^{1/n} - 1\right)\right]
     \exp\!\left(-\frac{|z|}{h_z}\right)

with :math:`r = \sqrt{x^2 + y^2}` evaluated in the tilted disk plane after the
viewing-angle rotation has been applied.

Analytical Rotation Curve
--------------------------

Tangential velocities are assigned from a compact empirical approximation
(``milky_way_rot_curve_analytical``):

.. math::

   v(R) = v_0 \times 1.022 \times \left(\frac{R}{R_0}\right)^{0.0803}

where :math:`v_0` is a characteristic velocity scale and :math:`R_0` is derived
from :math:`R_e` and :math:`n`.  This form reproduces the gently rising / flat
behaviour of typical disk-galaxy rotation curves at IFU-relevant scales.

Beam Convolution
-----------------

Each 2D channel map is convolved with an elliptical Gaussian to simulate finite
telescope resolution.  The FWHM-to-sigma conversion is:

.. math::

   \sigma = \frac{\mathrm{FWHM}}{2\sqrt{2\ln 2}} \approx \frac{\mathrm{FWHM}}{2.355}

This relation is used when constructing an ``astropy.convolution.Gaussian2DKernel``.

Diffuse Emission Physics
-------------------------

SONGS adds low-surface-brightness diffuse structure to the disk components
during cube assembly.  The model contains three additive pieces controlled by
``DEFAULT_DIFFUSE_PARAMS``.

Halo
~~~~

The extended halo around the central galaxy uses a spherical Sérsic-like radial
profile in the disk plane combined with a vertical exponential:

.. math::

   D_\mathrm{halo}(r, z)
     = A_\mathrm{halo}
       \exp\!\left[-b_n\!\left(\left(\frac{r}{R_\mathrm{halo}}\right)^{1/n_\mathrm{halo}} - 1\right)\right]
       \exp\!\left(-\frac{|z|}{h_{z,\mathrm{halo}}}\right)

The halo velocity field shares the same kinematic noise realisation as the
bridge components so that halo and bridges appear and disappear coherently in
the same spectral channels.

Bridges
~~~~~~~

Bridges are Gaussian tubes connecting the central galaxy to each satellite.
A bridge is parameterised by a unit vector :math:`\hat{u}` pointing from the
central to the satellite, with arc-length coordinate :math:`s \in [0, S]`:

.. math::

   D_\mathrm{bridge}(s, \rho)
     = A_\mathrm{bridge}
       \exp\!\left(-\frac{\rho^2}{2\sigma_\mathrm{bridge}^2}\right)
       \cdot w(s)

where :math:`\rho` is the perpendicular distance from the bridge axis and
:math:`w(s)` is a taper function that smoothly approaches zero at both
endpoints.  The velocity along the bridge is linearly interpolated between the
central systemic velocity :math:`v_\mathrm{cen}` and the satellite systemic
velocity :math:`v_\mathrm{sat}`:

.. math::

   v_\mathrm{bridge}(s) = v_\mathrm{cen} + \frac{s}{S}\,(v_\mathrm{sat} - v_\mathrm{cen})

Tidal Tails
~~~~~~~~~~~

Tidal tails are physically-motivated Bézier arcs extending from each satellite
*away* from the central galaxy.  The control points are constructed so that the
arc direction at the root is constrained to the outward hemisphere (opposite to
the central-to-satellite vector), preventing unphysical back-folding.

Let :math:`u \in [0, 1]` be the normalised arc-length parameter.  The tail
amplitude decays exponentially from root to tip:

.. math::

   A_\mathrm{tail}(u) = A_0 \exp\!\left(-\frac{u}{\lambda_\mathrm{decay}}\right)

so emission is densest near the progenitor satellite and thins toward the tip.

The velocity gradient along the tail is derived from the systemic velocity
difference between satellite and central:

.. math::

   v_\mathrm{tail}(u) = v_\mathrm{sat} + u \cdot \Delta v_\mathrm{sys} \cdot f_\mathrm{scale}

where :math:`\Delta v_\mathrm{sys} = v_\mathrm{sat} - v_\mathrm{cen}` and
:math:`f_\mathrm{scale}` is the ``vel_scale_factor`` diffuse parameter.

The cross-sectional width tapers linearly from root to tip:

.. math::

   \sigma_\mathrm{tail}(u) = \sigma_0 \,(1 - \alpha\, u), \quad 0 \le \alpha < 1

so the stream narrows as it stretches away from the galaxy.
