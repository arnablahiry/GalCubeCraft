from .core import GalCubeCraft

__all__ = ["GalCubeCraft"]


def init(n_gals=None, n_cubes=1, resolution='all', offset_gals=5, beam_info = [4,4,0], final_grid_size=125, n_spectral_slices=40, fname=None, verbose=True, seed=None):
    return GalCubeCraft(n_gals, n_cubes, resolution, offset_gals, beam_info, final_grid_size, n_spectral_slices, fname, verbose, seed)
