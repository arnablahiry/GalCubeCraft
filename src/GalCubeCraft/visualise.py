"""Visualization helpers for spectral cubes.

This module contains a simple plotting helper `visualise` which renders
moment-0 and moment-1 maps alongside a velocity spectrum plot.

Notes:
- The function uses utilities from :mod:`.utils` such as `convolve_beam`
    and `add_beam` to mimic telescope beam effects.
- When `save=True`, two files will be written with tight bounding boxes:
    ``mom0_mom1.pdf`` and ``vel_spectum.pdf`` inside the directory
    specified by `fname_save` (or a default `figures/<shape>` folder).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d
from astrodendro import Dendrogram
from .utils import convolve_beam, add_beam  # your utility functions
import os

def visualise(data, idx, save=False, fname_save=None):
    """Render moment maps and a velocity spectrum for a cube.

    Parameters
    ----------
    data : sequence
        Iterable containing tuples of (cube, metadata). The function will use
        ``data[idx]`` to obtain the cube and its metadata.
    idx : int
        Index of the cube to visualise within ``data``.
    save : bool, optional
        If True, save the two generated figures to disk. Default: False.
    fname_save : str or None, optional
        Directory path to save the figures into. If None, a default
        ``figures/<nz>x<ny>x<nx>`` directory under the current working
        directory will be created and used.

    The function creates two files when ``save=True``:
    - ``mom0_mom1.pdf`` : the side-by-side moment-0 and moment-1 maps
    - ``vel_spectum.pdf`` : the velocity spectrum plot

    The existing plotting logic and layout are preserved; this function
    only adds optional saving behaviour and explanatory docstrings.
    """

    # Extract cube and metadata for the requested index
    cube, meta = data[idx]

    # Prepare save directory: if user requested saving, compute a default
    # directory when none is provided and ensure it exists on disk.
    if save:
        if fname_save is None:
            # default folder path under the current working directory
            fname_save = os.path.join(os.getcwd(), 'figures',
                                      f'{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}')
        # Create the directory if it doesn't exist (idempotent)
        os.makedirs(fname_save, exist_ok=True)
    beam_info = meta['beam_info']
    vels = meta['average_vels']
    pix_spatial_scale = meta['pix_spatial_scale']
    del_V = np.diff(vels).mean()
    moment_cube = cube * vels[:, np.newaxis, np.newaxis]

    # Mask
    mask = np.zeros(cube.shape, dtype=bool)
    dendro = Dendrogram.compute(cube,
                                min_value=0.25*cube.std(),
                                min_delta=cube.std(),
                                verbose=False)
    for trunk in dendro.trunk:
        mask |= trunk.get_mask()

    # Plot moment maps
    ny, nx = cube.shape[1], cube.shape[2]
    extent = [0, nx, 0, ny]
    fig, ax = plt.subplots(1, 2, figsize=(7,5), sharex=True, sharey=True)

    # Moment 0
    im0 = ax[0].imshow(cube.sum(axis=0)*del_V, cmap='RdBu_r', origin='lower', extent=extent)
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("top", size="5%", pad=0.2)
    cb0 = fig.colorbar(im0, cax=cax0, orientation='horizontal', label=r'$\rm Jy/beam \cdot km/s$', format='%.2f')
    cb0.ax.xaxis.set_label_position('top')
    cb0.ax.xaxis.set_ticks_position('top')
    cb0.ax.xaxis.label.set_size(14)
    cb0.ax.tick_params(labelsize=12)
    cb0.ax.xaxis.labelpad = 10
    ax[0].text(nx*0.05, ny*0.89, 'Moment 0', color='white', fontsize=13, weight='bold')
    add_beam(ax[0], beam_info[0], beam_info[1], beam_info[2], xy_offset=(6,6), color='white')

    # Moment 1
    numerator = (mask * moment_cube).sum(axis=0)
    denominator = (mask * cube).sum(axis=0)
    ratio = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0)
    vmax = np.max([np.abs(np.nanmin(ratio)), np.abs(np.nanmax(ratio))])
    im1 = ax[1].imshow(ratio, cmap='RdBu_r', origin='lower', extent=extent, vmin=-vmax, vmax=vmax)
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("top", size="5%", pad=0.2)
    cb1 = fig.colorbar(im1, cax=cax1, orientation='horizontal', label=r'$\rm km/s$', format='%.0f')
    cb1.ax.xaxis.set_label_position('top')
    cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=12)
    cb1.ax.xaxis.label.set_size(14)
    cb1.ax.xaxis.labelpad = 10
    ax[1].text(nx*0.05, ny*0.89, 'Moment 1', color='black', fontsize=13, weight='bold')
    add_beam(ax[1], beam_info[0], beam_info[1], beam_info[2], xy_offset=(6,6), color='black')

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel('')
        a.set_ylabel('')
        a.set_aspect('equal')

    # Scalebar
    scalebar = (25/72)*cube.shape[2]
    x0, y0 = nx*0.6, ny*0.07
    for i, color in zip([0,1], ['white','black']):
        ax[i].plot([x0, x0+scalebar], [y0, y0], color=color, lw=2)
        ax[i].text(x0+scalebar/2, y0 + ny*0.03, f'{scalebar*pix_spatial_scale:.1f} kpc',
                   color=color, ha='center', va='bottom', fontsize=12, weight='bold')

    plt.tight_layout()
    # Save the first (moment) figure if requested. Keep tight bounding box.
    if save:
        try:
            fig.savefig(os.path.join(fname_save, 'mom0_mom1.pdf'), bbox_inches='tight')
        except Exception:
            # If saving fails for any reason, don't interrupt the display
            pass

    plt.show()

    # Velocity spectrum
    plt.figure(figsize=(7,4.5))
    plt.plot(vels, np.sum(cube, axis=(1,2)), color='xkcd:blue', linewidth=1.2)
    plt.ylabel(r'Flux Density ($\rm Jy/beam$)', fontsize=15, labelpad=10)
    plt.xlabel(r'(Line-of-sight) Velocity ($\rm km/s$)', fontsize=15, labelpad=12)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.grid(True)
    plt.tight_layout()
    # Save the velocity spectrum if requested
    if save:
        try:
            plt.savefig(os.path.join(fname_save, 'vel_spectum.pdf'), bbox_inches='tight')
        except Exception:
            pass

    plt.show()
