"""Plotting helpers for GalCubeCraft results.

This module provides small, focused plotting utilities that operate on the
``results`` list produced by :meth:`~GalCubeCraft.core.GalCubeCraft.generate_cubes`.
Each public helper accepts the ``results`` container and an index selecting
which generated cube to visualise. The functions are intentionally lightweight
and return a Matplotlib ``(fig, ax)`` pair so callers (GUIs, scripts, tests)
can further customise or save figures.

Dependencies
------------
- matplotlib (this module sets the ``TkAgg`` backend by default)
- astrodendro (used to compute a crude mask for visual guides)

Notes
-----
- These helpers call :func:`_prepare_cube` to extract cube / metadata and to
    compute a simple dendrogram-based mask used for moment maps. The dendrogram
    parameters are deliberately conservative and may be tuned for different
    signal-to-noise regimes.
- The plotting functions attempt to save to ``figures/<shape>/`` when
    ``save=True`` is passed; save failures are intentionally ignored to keep
    UI flows robust.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt installed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astrodendro import Dendrogram
from .utils import convolve_beam, add_beam
import os

def _prepare_cube(data, idx):
    """Internal helper: extract cube and derived metadata for plotting.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``GalCubeCraft.generate_cubes``.
        Each element should be a ``(cube, meta)`` tuple where ``cube`` is a
        NumPy array of shape ``(n_vel, ny, nx)`` and ``meta`` is a dict with
        keys including ``'beam_info'``, ``'average_vels'``, and
        ``'pix_spatial_scale'``.
    idx : int
        Index of the cube to extract.

    Returns
    -------
    cube : ndarray
        The spectral cube selected (shape ``n_vel x ny x nx``).
    meta : dict
        The metadata dictionary stored alongside the cube.
    beam_info : sequence
        Beam description (bmin_px, bmaj_px, bpa) as provided in ``meta``.
    vels : ndarray
        Velocity axis (km/s) corresponding to the spectral channels.
    pix_spatial_scale : float
        Physical scale per pixel (kpc/pixel).
    del_V : float
        Mean width of a spectral channel in km/s (used when computing
        moment0 units).
    moment_cube : ndarray
        The cube multiplied by the velocity axis (useful when computing
        first moment / intensity-weighted velocity).
    mask : ndarray (bool)
        A conservative mask derived from a dendrogram computed on the cube;
        intended to highlight contiguous emission regions for plotting.

    Notes
    -----
    - The dendrogram mask is produced with a heuristic threshold (0.25 x
      cube.std()) and may be noisy for very low S/N cubes. The mask is used
      to focus moment computations and to identify significant structures.
    """

    cube, meta = data[idx]
    beam_info = meta['beam_info']
    vels = meta['average_vels']
    pix_spatial_scale = meta['pix_spatial_scale']
    del_V = np.diff(vels).mean()
    moment_cube = cube * vels[:, np.newaxis, np.newaxis]

    # Create a conservative mask from a dendrogram to help moment maps
    mask = np.zeros(cube.shape, dtype=bool)
    dendro = Dendrogram.compute(cube,
                                min_value=0.25 * cube.std(),
                                min_delta=cube.std(),
                                verbose=False)
    for trunk in dendro.trunk:
        mask |= trunk.get_mask()

    return cube, meta, beam_info, vels, pix_spatial_scale, del_V, moment_cube, mask

def moment0(data, idx, save=False, fname_save=None):
    """Plot the zeroth moment (integrated intensity) of a spectral cube.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``GalCubeCraft.generate_cubes``.
    idx : int
        Index selecting which cube to plot.
    save : bool, optional
        If True, attempt to save the figure to ``figures/<shape>/moment0.pdf``.
    fname_save : str or None, optional
        Optional directory to save the figure. If None a path under the
        current working directory is chosen automatically.

    Returns
    -------
    fig, ax : (Figure, Axes)
        Matplotlib figure and axes objects containing the rendered moment map.
    """

    cube, meta, beam_info, vels, pix_spatial_scale, del_V, moment_cube, mask = _prepare_cube(data, idx)
    ny, nx = cube.shape[1], cube.shape[2]
    extent = [0, nx, 0, ny]

    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cube.sum(axis=0)*del_V, cmap='RdBu_r', origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.2)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', label=r'$\rm Jy/beam \cdot km\;s^{-1}$', format='%.2f')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.label.set_size(14)
    cb.ax.tick_params(labelsize=12)
    cb.ax.xaxis.labelpad = 10
    ax.text(nx*0.05, ny*0.89, 'Moment 0', color='white', fontsize=13, weight='bold')
    add_beam(ax, beam_info[0], beam_info[1], beam_info[2], xy_offset=(6,6), color='white')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Scalebar
    scalebar = (25/72)*cube.shape[2]
    x0, y0 = nx*0.6, ny*0.07
    ax.plot([x0, x0+scalebar], [y0, y0], color='white', lw=2)
    ax.text(x0+scalebar/2, y0 + ny*0.03, f'{scalebar*pix_spatial_scale:.1f} kpc',
            color='white', ha='center', va='bottom', fontsize=12, weight='bold')

    plt.tight_layout()

    if save:
        if fname_save is None:
            fname_save = os.path.join(os.getcwd(), 'figures', f'{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}')
        os.makedirs(fname_save, exist_ok=True)
        try:
            fig.savefig(os.path.join(fname_save, 'moment0.pdf'), bbox_inches='tight')
        except Exception:
            pass

    fig.show()
    return fig, ax


def moment1(data, idx, save=False, fname_save=None):
    """Plot the first moment (intensity-weighted velocity) of a spectral cube.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``GalCubeCraft.generate_cubes``.
    idx : int
        Index selecting which cube to plot.
    save : bool, optional
        If True, attempt to save the figure to ``figures/<shape>/moment1.pdf``.
    fname_save : str or None, optional
        Optional directory to save the figure. If None a path under the
        current working directory is chosen automatically.

    Returns
    -------
    fig, ax : (Figure, Axes)
        Matplotlib figure and axes objects containing the rendered moment map.
    """

    cube, meta, beam_info, vels, pix_spatial_scale, del_V, moment_cube, mask = _prepare_cube(data, idx)
    ny, nx = cube.shape[1], cube.shape[2]
    extent = [0, nx, 0, ny]

    numerator = (mask * moment_cube).sum(axis=0)
    denominator = (mask * cube).sum(axis=0)
    ratio = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0)
    vmax = np.max([np.abs(np.nanmin(ratio)), np.abs(np.nanmax(ratio))])

    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(ratio, cmap='RdBu_r', origin='lower', extent=extent, vmin=-vmax, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.2)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', label=r'$\rm km\;s^{-1}$', format='%.0f')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.tick_params(labelsize=12)
    cb.ax.xaxis.label.set_size(14)
    cb.ax.xaxis.labelpad = 10
    ax.text(nx*0.05, ny*0.89, 'Moment 1', color='black', fontsize=13, weight='bold')
    add_beam(ax, beam_info[0], beam_info[1], beam_info[2], xy_offset=(6,6), color='black')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Scalebar
    scalebar = (25/72)*cube.shape[2]
    x0, y0 = nx*0.6, ny*0.07
    ax.plot([x0, x0+scalebar], [y0, y0], color='black', lw=2)
    ax.text(x0+scalebar/2, y0 + ny*0.03, f'{scalebar*pix_spatial_scale:.1f} kpc',
            color='black', ha='center', va='bottom', fontsize=12, weight='bold')

    plt.tight_layout()
    if save:
        if fname_save is None:
            fname_save = os.path.join(os.getcwd(), 'figures', f'{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}')
        os.makedirs(fname_save, exist_ok=True)
        try:
            fig.savefig(os.path.join(fname_save, 'moment1.pdf'), bbox_inches='tight')
        except Exception:
            pass

    fig.show()
    return fig, ax


def spectrum(data, idx, save=False, fname_save=None):
    """Plot the integrated spectrum (total flux vs velocity) for a cube.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``GalCubeCraft.generate_cubes``.
    idx : int
        Index selecting which cube to plot.
    save : bool, optional
        If True, save the figure as ``spectrum.pdf`` under
        ``figures/<shape>/`` unless ``fname_save`` overrides the path.
    fname_save : str or None, optional
        Optional directory to save the figure.

    Returns
    -------
    fig, ax : (Figure, Axes)
        The Matplotlib figure and axes containing the spectrum.
    """

    cube, meta, beam_info, vels, pix_spatial_scale, del_V, moment_cube, mask = _prepare_cube(data, idx)

    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(vels, np.sum(cube, axis=(1,2)), color='xkcd:blue', linewidth=1.2)
    ax.set_ylabel(r'Flux Density ($\rm Jy/beam$)', fontsize=15, labelpad=10)
    ax.set_xlabel(r'(Line-of-sight) Velocity ($\rm km\;s^{-1}   $)', fontsize=15, labelpad=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True)
    plt.tight_layout()

    if save:
        if fname_save is None:
            fname_save = os.path.join(os.getcwd(), 'figures', f'{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}')
        os.makedirs(fname_save, exist_ok=True)
        try:
            fig.savefig(os.path.join(fname_save, 'spectrum.pdf'), bbox_inches='tight')
        except Exception:
            pass

    fig.show()
    return fig, ax
