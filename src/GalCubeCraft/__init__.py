"""GalCubeCraft package public API.

This module exposes the primary :class:`GalCubeCraft` generator and a small
helper ``init`` wrapper for convenience when importing the package in
interactive sessions. The helper mirrors the constructor signature of
``GalCubeCraft`` and returns an instantiated generator.

Public
------
- GalCubeCraft: main generator class (imported from :mod:`.core`)
- init(...): convenience wrapper that returns a configured GalCubeCraft

Example
-------
>>> from GalCubeCraft import init
>>> g = init(n_cubes=1, seed=42)
>>> g.generate_cubes()
"""

from .core import GalCubeCraft

__all__ = ["GalCubeCraft", "init"]


def init(n_gals=None, n_cubes=1, resolution='all', offset_gals=5, beam_info = [4,4,0], grid_size=125, n_spectral_slices=40, n_sersic=None, fname=None, verbose=True, seed=None):
    """Create and return a configured :class:`GalCubeCraft` instance.

    This thin convenience wrapper mirrors the constructor signature of
    :class:`GalCubeCraft` and is intended for quick interactive use where
    callers prefer a short import path such as ``from GalCubeCraft import init``.

    See :class:`GalCubeCraft` for detailed parameter descriptions.
    """
    return GalCubeCraft(n_gals, n_cubes, resolution, offset_gals, beam_info, grid_size, n_spectral_slices, n_sersic, fname, verbose, seed)
