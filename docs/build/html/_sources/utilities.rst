Utilities
=========

.. automodule:: songs.utils
   :members:
   :undoc-members:
   :show-inheritance:

Beam Convolution
----------------

The ``add_beam`` helper convolves a 2D channel map with an elliptical Gaussian
beam specified by its minor axis (bmin), major axis (bmaj), and position angle
(BPA):

.. code-block:: python

    from songs.utils import add_beam
    convolved_channel = add_beam(channel_2d, bmin=4, bmaj=4, bpa=0)

The FWHM-to-sigma conversion used internally is:

.. math::

   \sigma = \frac{\mathrm{FWHM}}{2\sqrt{2\ln 2}} \approx \frac{\mathrm{FWHM}}{2.355}

This matches the standard relation for Gaussian kernels.

Mask Helpers
------------

Utility functions for generating spatial masks from moment maps or
signal-to-noise thresholds are also located in ``songs.utils``.  These are
used internally by the generator when saving cubes and by the GUI SliceViewer
for bounding-box detection.
