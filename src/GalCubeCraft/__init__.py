"""GalCubeCraft public API.

This module defines the package's public-facing interface and exposes two
complementary generators for different use cases:

- Population-based generator in pixel units (``GalCubeCraft``): designed for
    statistical sampling and batch generation of synthetic galaxy cubes.
- Physically parameterised generator in explicit units (``GalCubeCraft_Phy``):
    designed for interactive construction of single systems, e.g. via the GUI.

Thin convenience wrappers (``init`` and ``init_phy``) are provided.
"""

from .core import GalCubeCraft, GalCubeCraft_Phy

__all__ = [
    "GalCubeCraft",
    "GalCubeCraft_Phy",
    "init",
    "init_phy",
]


def init_phy(
    n_gals=None,
    n_cubes=1,
    spatial_resolution=4.5,
    spectral_resolution=10,
    offset_gals=20,
    beam_info=[18, 18, 0],
    fov=125,
    n_sersic=None,
    save=False,
    fname=None,
    verbose=True,
    seed=None,
):
    """
    Create and return a physically parameterised :class:`GalCubeCraft` instance.

    This entry point is designed for **explicit physical-unit parametrisation**
    (e.g. kpc, km s⁻¹) and is primarily intended for interactive usage through
    the graphical user interface. In this mode, galaxies are specified via
    directly interpretable physical quantities, enabling the construction of
    highly controlled, single systems.

    Unlike the population-based generator, parameters are not drawn from
    statistical distributions but are instead provided explicitly, making this
    interface well suited for exploratory modelling, demonstrations, and
    pedagogical use.

    Parameters
    ----------
    All parameters are forwarded directly to the :class:`GalCubeCraft`
    constructor. Refer to the class documentation for detailed descriptions
    of each physical quantity.

    Returns
    -------
    GalCubeCraft
        An initialized GalCubeCraft generator configured for physical-unit
        parametrisation.
    """
    return GalCubeCraft_Phy(
        n_gals,
        n_cubes,
        spatial_resolution,
        spectral_resolution,
        offset_gals,
        beam_info,
        fov,
        n_sersic,
        save,
        fname,
        verbose,
        seed,
    )


def init(
    n_gals=None,
    n_cubes=1,
    resolution="all",
    offset_gals=5,
    beam_info=[4, 4, 0],
    grid_size=125,
    n_spectral_slices=40,
    n_sersic=None,
    fname=None,
    verbose=True,
    seed=None,
):
    """
    Create and return a population-based :class:`GalCubeCraft` instance.

    This entry point corresponds to the **standard GalCubeCraft generator**,
    operating primarily in **pixel units**. It is designed to generate
    ensembles of synthetic galaxy cubes in which parameter values are drawn
    from uniform distributions over physically viable ranges.

    This mode is intended for statistical studies, machine-learning training
    set generation, and large-scale benchmarking, where diversity across
    many realisations is more important than fine control over individual
    systems.

    Parameters
    ----------
    All parameters are forwarded directly to the :class:`GalCubeCraft`
    constructor. Refer to the class documentation for details on the sampling
    strategy and parameter ranges.

    Returns
    -------
    GalCubeCraft
        An initialized GalCubeCraft generator configured for population-based
        sampling in pixel space.
    """
    return GalCubeCraft(
        n_gals,
        n_cubes,
        resolution,
        offset_gals,
        beam_info,
        grid_size,
        n_spectral_slices,
        n_sersic,
        fname,
        verbose,
        seed,
    )
