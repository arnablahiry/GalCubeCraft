"""Plotting helpers for SONGS results.

This module provides small, focused plotting utilities that operate on the
``results`` list produced by :meth:`~SONGS.core.SONGS.generate_cubes`.
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
import matplotlib
import matplotlib.pyplot as plt

# Only force TkAgg if we aren't already in an inline environment
if not matplotlib.get_backend().lower().startswith('inline'):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astrodendro import Dendrogram
from .utils import convolve_beam, add_beam
import os
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def _prepare_cube(data, idx):
    """Internal helper: extract cube and derived metadata for plotting.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``SONGS.generate_cubes``.
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

def moment0(data, idx, save=False, fname_save=None, inline=False):
    """Plot the zeroth moment (integrated intensity) of a spectral cube.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``SONGS.generate_cubes``.
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

    # Compute fixed colour limits from the integrated (moment0) map so the
    # slice viewer uses a consistent scale across channels.
    integrated = cube.sum(axis=0) * del_V
    vmin = float(np.nanmin(integrated))
    vmax = float(np.nanmax(integrated))

    fig, ax = plt.subplots(figsize=(5,5))
    # Set a descriptive window title where the backend/window manager
    # exposes a canvas manager (e.g., TkAgg). Wrap in try/except for
    # environments where this attribute is not available.
    if not inline:
        try:
            fig.canvas.manager.set_window_title('Moment 0')
        except Exception:
            try:
                # Older matplotlib versions expose a different attribute
                fig.canvas.set_window_title('Moment 0')
            except Exception:
                pass
        except Exception:
            pass
    im = ax.imshow(cube.sum(axis=0) * del_V, cmap='RdBu_r', origin='lower', extent=extent, vmin=0, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.2)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', label=r'$\rm Jy\;beam^{-1} \cdot km\;s^{-1}$', format='%.2f')
    # Place label and ticks on the top and draw ticks outward from the
    # colorbar so they appear above the bar (consistent with Moment0).
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.label.set_size(14)
    # Make ticks point outwards and add a small pad so label is above ticks
    cb.ax.tick_params(labelsize=12, direction='out', pad=6)
    cb.ax.xaxis.labelpad = 12
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

    # Final interactive trigger
    if not inline:
        fig.show()
    return fig, ax


def moment1(data, idx, save=False, fname_save=None):
    """Plot the first moment (intensity-weighted velocity) of a spectral cube.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``SONGS.generate_cubes``.
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
    try:
        fig.canvas.manager.set_window_title('Moment 1')
    except Exception:
        try:
            fig.canvas.set_window_title('Moment 1')
        except Exception:
            pass
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
    add_beam(ax, beam_info[0], beam_info[1], beam_info[2], xy_offset=(6*cube.shape[1]/72,6*cube.shape[1]/72), color='black')

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
        The ``results`` container produced by ``SONGS.generate_cubes``.
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
    try:
        fig.canvas.manager.set_window_title('Integrated LOS Spectrum')
    except Exception:
        try:
            fig.canvas.set_window_title('Integrated LOS Spectrum')
        except Exception:
            pass
    ax.plot(vels, np.sum(cube, axis=(1,2)), color='xkcd:blue', linewidth=1.2)
    ax.set_ylabel(r'Flux Density ($\rm Jy\;beam^{-1}$)', fontsize=15, labelpad=10)
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


# ---------------------------------------------------------------------------
# Colour constants — SONGS theme (black + faint yellow)
# ---------------------------------------------------------------------------
_BG           = "#0a0a0a"   # window background
_CARD_BG      = "#111111"   # control card / sidebar background
_ACCENT       = "#b8960a"   # faint yellow — headings, active elements
_ACCENT_HOV   = "#f0c040"   # bright yellow on hover
_DIM          = "#2e2000"   # very dark yellow — borders
_DIM_TXT      = "#3a3010"   # dimmed text
_LOG_BG       = "#0a0a0a"   # matplotlib figure background
_STEP_LBL     = "#999999"   # secondary label text

_CMAPS = ["inferno", "viridis", "magma", "plasma", "cividis",
          "gray", "hot", "afmhot", "YlOrRd", "cubehelix"]

# Distinct palette for up to 8 sources (matches matplotlib tab10 first 8)
_SRC_PALETTE = [
    (0.122, 0.467, 0.706),  # blue      — central
    (1.000, 0.498, 0.055),  # orange    — sat 1
    (0.173, 0.627, 0.173),  # green     — sat 2
    (0.839, 0.153, 0.157),  # red       — sat 3
    (0.580, 0.404, 0.741),  # purple    — sat 4
    (0.549, 0.337, 0.294),  # brown     — sat 5
    (0.890, 0.467, 0.761),  # pink      — sat 6
    (0.498, 0.498, 0.498),  # grey      — sat 7
]


def _src_label(i: int) -> str:
    return "Central Galaxy" if i == 0 else f"Satellite {i}"


def _rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def _lighten(rgb, amount=0.3):
    return tuple(min(c + amount, 1.0) for c in rgb)


class SliceViewer(tk.Toplevel):
    """Channel-by-channel IFU slice viewer matching nemo aesthetics.

    Displays the full spectral cube with colormap / normalization controls,
    vmin/vmax sliders, and (when per-galaxy cubes are available) a sources
    sidebar with per-source contours, bounding boxes, and intensity-threshold
    masks whose threshold percentage is tunable via a dedicated slider.
    """

    def __init__(self, master, data, idx: int = 0):
        super().__init__(master)
        self.configure(bg=_BG)
        self.resizable(True, True)
        self.title("SONGS — IFU Slice Viewer")

        # ── Unpack data ──────────────────────────────────────────────────────
        cube, meta = data[idx]
        self._cube        = cube.astype(np.float32)
        self._vels        = np.asarray(meta.get("average_vels", np.arange(cube.shape[0])))
        self._beam        = meta.get("beam_info")
        self._pix_scale   = float(meta.get("pix_spatial_scale", 1.0))
        # per-galaxy ground-truth cubes: shape (n_gals, n_ch, ny, nx)
        pg = meta.get("per_galaxy_cubes")
        self._per_gal     = np.asarray(pg) if pg is not None else None
        self._n_gals      = int(self._per_gal.shape[0]) if self._per_gal is not None else 0

        n_ch, ny, nx = self._cube.shape
        self._channels = list(range(n_ch))
        VW = 500

        flat             = self._cube.ravel()
        self._data_min   = float(np.nanmin(flat))
        self._data_max   = float(np.nanmax(flat))

        # ── Matplotlib figure ────────────────────────────────────────────────
        self._fig   = plt.Figure(figsize=(VW/96, VW/96), dpi=96, facecolor=_LOG_BG)
        self._ax    = self._fig.add_axes([0.01, 0.01, 0.82, 0.98])
        self._ax_cb = self._fig.add_axes([0.86, 0.01, 0.06, 0.98])
        for spine in self._ax.spines.values():
            spine.set_edgecolor(_DIM_TXT)
            spine.set_linewidth(0.8)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._ax_cb.set_facecolor(_LOG_BG)

        # ── Layout: sidebar + canvas ─────────────────────────────────────────
        top = tk.Frame(self, bg=_BG)
        top.pack(fill=tk.BOTH, expand=True)

        _SLIDER_BORDER = "#2e2000"  # low-opacity yellow border matching main GUI

        def _scale(parent, from_, to_, default, length, cmd, show=False):
            """Return a (border_wrap, scale) pair; pack the wrap, query the scale."""
            wrap = tk.Frame(parent, bg=_SLIDER_BORDER, padx=1, pady=1)
            inner = tk.Frame(wrap, bg=_CARD_BG)
            inner.pack(fill='both', expand=True)
            s = tk.Scale(inner, from_=from_, to=to_,
                         resolution=(to_ - from_) / 500 if to_ != from_ else 0.01,
                         orient=tk.HORIZONTAL, command=cmd,
                         bg=_ACCENT, fg=_BG, troughcolor=_CARD_BG,
                         activebackground=_ACCENT_HOV, highlightthickness=0,
                         sliderrelief=tk.FLAT, bd=0, width=6,
                         length=length, showvalue=show)
            s.pack(fill='x', expand=True)
            s.set(default)
            return wrap, s

        # Sources sidebar (only when per-galaxy cubes exist)
        self._src_visible: dict[int, tk.BooleanVar] = {}
        if self._per_gal is not None:
            sb = tk.Frame(top, bg=_BG, width=140)
            sb.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 2), pady=4)
            sb.pack_propagate(False)

            tk.Label(sb, text="Sources", bg=_BG, fg=_ACCENT,
                     font=("Helvetica", 9, "bold")).pack(pady=(4, 6), anchor="w")

            for i in range(self._n_gals):
                col     = _SRC_PALETTE[i % len(_SRC_PALETTE)]
                hex_col = _rgb_to_hex(col)
                var     = tk.BooleanVar(value=True)
                self._src_visible[i] = var
                row = tk.Frame(sb, bg=_BG)
                row.pack(fill=tk.X, pady=1, anchor="w")
                tk.Label(row, text="■", bg=_BG, fg=hex_col,
                         font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 2))
                tk.Checkbutton(row, text=_src_label(i), variable=var,
                               command=self._draw,
                               bg=_BG, fg="white", selectcolor=_ACCENT,
                               activebackground=_BG, activeforeground=_ACCENT_HOV,
                               font=("Helvetica", 8), relief=tk.FLAT,
                               anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Frame(sb, bg=_DIM, height=1).pack(fill=tk.X, pady=(8, 4))
            tk.Label(sb, text="Threshold %", bg=_BG, fg=_STEP_LBL,
                     font=("Helvetica", 7)).pack(anchor="w", pady=(2, 1))
            self._thresh_var = tk.DoubleVar(value=5.0)
            def _thresh_cmd(v):
                try: self._thresh_var.set(float(v))
                except Exception: pass
                self._draw()
            _tw, _ts = _scale(sb, 0.1, 50.0, 5.0, 120, _thresh_cmd, show=True)
            _tw.pack(fill=tk.X)
        else:
            self._thresh_var = tk.DoubleVar(value=5.0)

        self._canvas = FigureCanvasTkAgg(self._fig, master=top)
        self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Control bar: colormap + invert + norm ────────────────────────────
        ctrl = tk.Frame(self, bg=_CARD_BG)
        ctrl.pack(fill=tk.X, padx=0, pady=0)
        tk.Frame(ctrl, bg=_DIM, height=1).pack(fill=tk.X)
        inner_ctrl = tk.Frame(ctrl, bg=_CARD_BG)
        inner_ctrl.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(inner_ctrl, text="Colormap:", bg=_CARD_BG, fg=_STEP_LBL,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(0, 3))
        self._cmap     = tk.StringVar(value="inferno")
        self._inverted = tk.BooleanVar(value=False)
        om = tk.OptionMenu(inner_ctrl, self._cmap, *_CMAPS, command=lambda _v: self._draw())
        om.configure(bg=_CARD_BG, fg=_ACCENT,
                     activebackground=_DIM, activeforeground=_ACCENT_HOV,
                     highlightthickness=1, highlightbackground=_DIM,
                     relief=tk.FLAT, font=("Helvetica", 8), width=8)
        om["menu"].configure(bg=_CARD_BG, fg=_ACCENT,
                             activebackground=_DIM, activeforeground=_ACCENT_HOV,
                             font=("Helvetica", 8))
        om.pack(side=tk.LEFT, padx=(0, 6))

        tk.Checkbutton(inner_ctrl, text="Invert", variable=self._inverted,
                       command=self._draw,
                       bg=_CARD_BG, fg=_STEP_LBL, selectcolor=_ACCENT,
                       activebackground=_CARD_BG, activeforeground=_ACCENT_HOV,
                       font=("Helvetica", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 12))

        self._norm_mode = tk.StringVar(value="linear")
        tk.Label(inner_ctrl, text="Norm:", bg=_CARD_BG, fg=_STEP_LBL,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(0, 3))
        for lbl in ("linear", "log", "power"):
            tk.Radiobutton(inner_ctrl, text=lbl, variable=self._norm_mode, value=lbl,
                           command=self._draw,
                           bg=_CARD_BG, fg="white", selectcolor=_ACCENT,
                           activebackground=_CARD_BG, activeforeground=_ACCENT_HOV,
                           font=("Helvetica", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=2)

        # ── Channel label ─────────────────────────────────────────────────────
        self._ch_lbl = tk.Label(self, text="", bg=_BG, fg=_ACCENT,
                                font=("Helvetica", 8, "italic"), anchor="w")
        self._ch_lbl.pack(fill=tk.X, padx=12, pady=(4, 2))

        # ── vmin / vmax sliders ───────────────────────────────────────────────
        vf = tk.Frame(self, bg=_BG)
        vf.pack(fill=tk.X, padx=10, pady=(2, 2))
        vf.columnconfigure(1, weight=1)

        def _make_slider(parent, label, default, row):
            tk.Label(parent, text=label, bg=_BG, fg=_STEP_LBL,
                     font=("Helvetica", 7), width=5, anchor="e").grid(
                         row=row, column=0, padx=(0, 4), pady=1)
            wrap, s = _scale(parent, self._data_min, self._data_max, default,
                             VW - 160, lambda _v: self._draw())
            wrap.grid(row=row, column=1, sticky="ew", pady=1)
            lbl = tk.Label(parent, text="", bg=_BG, fg=_ACCENT,
                           font=("Helvetica", 7), width=10, anchor="w")
            lbl.grid(row=row, column=2, padx=(6, 0), pady=1)
            return s, lbl

        self._vmin_sl, self._vmin_lbl = _make_slider(vf, "vmin", self._data_min, 0)
        self._vmax_sl, self._vmax_lbl = _make_slider(vf, "vmax", self._data_max, 1)

        # ── Channel slider ────────────────────────────────────────────────────
        sf = tk.Frame(self, bg=_BG)
        sf.pack(fill=tk.X, padx=10, pady=(2, 8))
        tk.Label(sf, text="Channel", bg=_BG, fg=_STEP_LBL,
                 font=("Helvetica", 8), width=7, anchor="e").pack(side=tk.LEFT, padx=(0, 6))
        _sw, self._slider = _scale(sf, 0, len(self._channels) - 1,
                                   len(self._channels) // 2,
                                   VW - 120, lambda _v: self._draw())
        _sw.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._draw()
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"{w}x{h}")
        self.minsize(w, h)
        self.maxsize(w, 9999)

    # ── Normalization ─────────────────────────────────────────────────────────
    def _norm(self):
        from matplotlib.colors import Normalize, LogNorm, PowerNorm
        vmin = float(self._vmin_sl.get())
        vmax = float(self._vmax_sl.get())
        if vmin >= vmax:
            vmax = vmin + 1e-9
        mode = self._norm_mode.get()
        if mode == "log":
            vmin = max(vmin, 1e-12)
            vmax = max(vmax, vmin + 1e-12)
            return LogNorm(vmin=vmin, vmax=vmax)
        elif mode == "power":
            return PowerNorm(gamma=0.5, vmin=max(vmin, 0), vmax=vmax)
        return Normalize(vmin=vmin, vmax=vmax)

    def _fmt_val(self, v: float) -> str:
        if self._norm_mode.get() == "log":
            return f"10^{np.log10(max(abs(v), 1e-30)):.2f}"
        return f"{v:.2e}"

    # ── Main draw ─────────────────────────────────────────────────────────────
    def _draw(self):
        idx = int(self._slider.get())
        ch  = self._channels[idx]
        img = self._cube[ch]

        norm = self._norm()
        cmap = self._cmap.get() + ("_r" if self._inverted.get() else "")

        self._ax.clear()
        self._ax.set_xticks([]); self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_edgecolor(_DIM)
            spine.set_linewidth(0.6)
        self._ax.imshow(img, cmap=cmap, norm=norm, origin="lower")

        # Colorbar
        self._fig.set_facecolor(_LOG_BG)
        self._ax_cb.set_facecolor(_LOG_BG)
        self._ax_cb.clear()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = self._fig.colorbar(sm, cax=self._ax_cb)
        cb.ax.tick_params(colors=_ACCENT, labelsize=5, length=2)
        cb.outline.set_edgecolor(_DIM)
        plt.setp(plt.getp(cb.ax, "yticklabels"), color=_ACCENT, fontsize=5)

        # Per-source contours + bboxes
        if self._per_gal is not None:
            thresh_frac = float(self._thresh_var.get()) / 100.0
            from matplotlib.patches import Rectangle as _Rect
            PAD = 4
            for i in range(self._n_gals):
                if not self._src_visible[i].get():
                    continue
                if ch >= self._per_gal.shape[1]:
                    continue
                gal_ch = self._per_gal[i, ch]
                cube_max = float(np.nanmax(self._per_gal[i])) if np.nanmax(self._per_gal[i]) > 0 else 1e-9
                thresh = thresh_frac * cube_max
                mask = gal_ch >= thresh
                if not mask.any():
                    continue
                col  = _SRC_PALETTE[i % len(_SRC_PALETTE)]
                lcol = _lighten(col)
                self._ax.contour(mask.astype(float), [0.5],
                                 colors=[col], linewidths=0.8)
                rows, cols = np.where(mask)
                r0, r1 = int(rows.min()), int(rows.max())
                c0, c1 = int(cols.min()), int(cols.max())
                self._ax.add_patch(_Rect(
                    (c0 - PAD, r0 - PAD),
                    c1 - c0 + 2*PAD, r1 - r0 + 2*PAD,
                    linewidth=0.8, edgecolor=lcol, facecolor="none", zorder=4,
                ))
                label = "C" if i == 0 else f"S{i}"
                self._ax.text(
                    c1 + PAD, r1 + PAD, label,
                    ha="center", va="center", fontsize=7,
                    color="black", fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.22", fc=lcol, ec=lcol, lw=1.2),
                    zorder=6,
                )

        # Beam + scalebar
        if self._beam is not None:
            try:
                add_beam(self._ax, self._beam[0], self._beam[1], self._beam[2],
                         xy_offset=(6*img.shape[0]/72, 6*img.shape[0]/72), color=_ACCENT)
            except Exception:
                pass
        ny, nx = img.shape
        scalebar_px = (25/72) * nx
        x0, y0 = nx * 0.6, ny * 0.07
        self._ax.plot([x0, x0 + scalebar_px], [y0, y0], color=_ACCENT, lw=1.5)
        self._ax.text(x0 + scalebar_px/2, y0 + ny*0.03,
                      f"{scalebar_px * self._pix_scale:.1f} kpc",
                      color=_ACCENT, ha="center", va="bottom",
                      fontsize=7, weight="bold")

        self._canvas.draw()

        # Update value labels
        self._vmin_lbl.configure(text=self._fmt_val(float(self._vmin_sl.get())))
        self._vmax_lbl.configure(text=self._fmt_val(float(self._vmax_sl.get())))

        # Channel label
        v = self._vels[ch] if ch < len(self._vels) else 0.0
        n_active = 0
        if self._per_gal is not None:
            thresh_frac = float(self._thresh_var.get()) / 100.0
            for i in range(self._n_gals):
                if not self._src_visible[i].get():
                    continue
                if ch >= self._per_gal.shape[1]:
                    continue
                cube_max = float(np.nanmax(self._per_gal[i])) if np.nanmax(self._per_gal[i]) > 0 else 1e-9
                if (self._per_gal[i, ch] >= thresh_frac * cube_max).any():
                    n_active += 1
        parts = [f"Channel {ch}  ({idx+1}/{len(self._channels)})  ·  {v:.1f} km/s"]
        if n_active:
            parts.append(f"{n_active} source(s) visible")
        self._ch_lbl.configure(text="  ·  ".join(parts))


def slice_view(data, idx=0, channel=None, cmap='viridis', parent=None):
    """Show a per-channel slice viewer embedded in a Tk window.

    This viewer embeds the Matplotlib figure into a Tk Toplevel and uses a
    ttk-styled slider to step through spectral channels. The viewer keeps the
    lower colour limit fixed at 0 and computes an upper limit per slice so
    contrast adapts to the currently displayed channel.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``SONGS.generate_cubes``.
    idx : int, optional
        Index of the cube within ``data`` to display (default 0).
    channel : int, optional
        Initial spectral channel index to show. If ``None`` (the default)
        the viewer will open on the central spectral channel (``int(n/2)``).
    cmap : str, optional
        Matplotlib colormap to use for imshow.
    parent : tkinter widget, optional
        If provided, the slice viewer will be a child Toplevel of this
        widget. Otherwise a new Toplevel (or root) is used.

    Returns
    -------
    fig, ax : (Figure, Axes)
        The Matplotlib figure and axes used by the embedded viewer.
    """

    cube, meta, beam_info, vels, pix_spatial_scale, del_V, moment_cube, mask = _prepare_cube(data, idx)

    n_chan = int(cube.shape[0])
    # Default spectral index: middle channel
    if channel is None:
        # Use the 1-based middle slice formula int((n_slices+1)/2) then
        # convert to 0-based index by subtracting 1. This matches the
        # user's requested behaviour for odd/even slice counts.
        channel = int((n_chan + 1) / 2) - 1
    channel = int(max(0, min(int(channel), n_chan - 1)))

    # Precompute fixed colour limits from the integrated (moment0) map so the
    # fixed option has a consistent reference scale. vmin is fixed at 0.
    fixed_vmin = 0.0
    fixed_vmax = float(np.nanmax(cube))

    # Create a Tk Toplevel to host the canvas. If there's an existing Tk
    # root, make a Toplevel so we don't create a second main window.
    if parent is not None:
        win = tk.Toplevel(master=parent)
    else:
        if tk._default_root is None:
            win = tk.Tk()
        else:
            win = tk.Toplevel()
    win.title(f"IFU viewer")

    ny, nx = cube.shape[1], cube.shape[2]
    extent = [0, nx, 0, ny]

    from matplotlib.figure import Figure as _Figure
    fig = _Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    # Shift the subplot region slightly up so title, figure and colorbar sit
    # a bit higher in the Toplevel window by default.
    fig.subplots_adjust(top=0.95, bottom=0.12)
    # Use the same colormap and units styling as moment0. Multiply single
    im = ax.imshow(cube[channel, :, :], cmap='RdBu_r', origin='lower', extent=extent, vmin=0.0, vmax=fixed_vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    divider = make_axes_locatable(ax)
    # Put the colorbar below the image
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', label=r'$\rm Jy\;beam^{-1}$', format='%.2f')
    # Place label and ticks on the bottom and draw ticks outward
    cb.ax.xaxis.set_label_position('bottom')
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.ax.xaxis.label.set_size(14)
    cb.ax.tick_params(labelsize=12, direction='out', pad=6)
    cb.ax.xaxis.labelpad = 6

    # Initialize color limits according to the autoscale default (per-slice)
    try:
        sl0 = cube[channel, :, :]
        v1 = float(np.nanmax(sl0))
        im.set_clim(0.0, v1)
        cb.set_clim(0.0, v1)
        cb.draw_all()
    except Exception:
        # fall back to fixed integrated limits
        try:
            im.set_clim(fixed_vmin, fixed_vmax)
            cb.set_clim(fixed_vmin, fixed_vmax)
            cb.draw_all()
        except Exception:
            pass
    # (we will show the channel/velocity description below the figure as
    # LaTeX text; keep the axes area free of a title overlay)
    add_beam(ax, beam_info[0], beam_info[1], beam_info[2], xy_offset=(6*cube.shape[1]/72,6*cube.shape[1]/72), color='white')

    ax.set_aspect('equal')

    # Scalebar (match moment0 style)
    scalebar = (25/72)*cube.shape[2]
    x0, y0 = nx*0.6, ny*0.07
    ax.plot([x0, x0+scalebar], [y0, y0], color='white', lw=2)
    ax.text(x0+scalebar/2, y0 + ny*0.03, f'{scalebar*pix_spatial_scale:.1f} kpc',
        color='white', ha='center', va='bottom', fontsize=12, weight='bold')

    # Embed the Matplotlib figure in the Tk window. We will draw once and
    # compute sizes so we can fix the Toplevel geometry; this prevents the
    # window from resizing when the controls (scale/labels) update.
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas_widget = canvas.get_tk_widget()
    # Pack without expansion so the geometry we set stays stable
    canvas_widget.pack(side=tk.TOP)

    # Optional navigation toolbar
    toolbar = None
    try:
        toolbar = NavigationToolbar2Tk(canvas, win)
        # Place toolbar above the canvas (it will request its own height)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        toolbar.update()
    except Exception:
        toolbar = None

    # Controls frame with a native Tk scale for robust interaction
    ctrl = tk.Frame(win)
    ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)

    # Use a ttk-styled slider that matches the rest of the GUI. We create a
    # small slider row with a right-aligned numeric label like the main app's
    # `make_slider` helper so appearance is consistent.
    label = ttk.Label(ctrl, text=f"Channel: {channel+1} : v = {vels[channel]:.1f} km/s")
    label.pack(side=tk.LEFT, padx=(0, 12))

    slider_row = ttk.Frame(ctrl)
    slider_row.pack(side=tk.LEFT, fill='x', expand=1)
    # Show the displayed channel as 1-based to match user expectation
    val_lbl = ttk.Label(slider_row, text=f"{channel+1}", width=6, anchor="e")
    val_lbl.pack(side='right', padx=(4, 0))

    scale = ttk.Scale(slider_row, from_=0, to=n_chan - 1, orient='horizontal')
    scale.pack(side='left', fill='x', expand=1)

    # No autoscale checkbox: always use vmin=0 and per-slice vmax by default.

    # Snapping/busy guard to avoid recursive updates and ensure integer steps
    busy = {'val': False}
    # Create a LaTeX title as the Axes title (so it appears above the image)
    def _latex_for(ci, v):
        return r"$\rm Channel\ %d\;:\;v=%.1f\;km\;s^{-1}$" % (ci, v)

    # Set the initial title on the axes (matplotlib mathtext will render it)
    ax.set_title(_latex_for(channel+1, vels[channel]), fontsize=13)

    def _on_scale(val):
        if busy['val']:
            return
        busy['val'] = True
        try:
            ci = int(round(float(val)))
        except Exception:
            busy['val'] = False
            return
        # update widgets and image
        # Display channel number as 1-based
        val_lbl.config(text=str(ci + 1))
        sl = cube[ci, :, :]
        im.set_data(sl)
        # Update the Axes title (LaTeX) and the left-side label
        ax.set_title(_latex_for(ci+1, vels[ci]))
        label.config(text=f"Channel: {ci+1} : v = {vels[ci]:.1f} km/s")
        # Always update the displayed data and compute a per-slice vmax.
        try:
            try:
                v1 = float(np.nanmax(sl))
            except Exception:
                v1 = fixed_vmax
            try:
                im.set_clim(0.0, v1)
                cb.set_clim(0.0, v1)
                cb.draw_all()
            except Exception:
                try:
                    im.set_clim(0.0, fixed_vmax)
                    cb.set_clim(0.0, fixed_vmax)
                    cb.draw_all()
                except Exception:
                    pass
            try:
                canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            # Keep function robust: if anything unexpected fails, continue
            # without crashing the UI.
            pass
        busy['val'] = False

    scale.configure(command=_on_scale)
    try:
        scale.set(channel)
    except Exception:
        pass

    # Force an initial draw so geometry measurements are reliable
    try:
        canvas.draw()
        win.update_idletasks()
    except Exception:
        pass

    # Measure sizes
    try:
        c_w, c_h = canvas.get_width_height()
    except Exception:
        # Fallback to widget requested size
        c_w = canvas_widget.winfo_reqwidth()
        c_h = canvas_widget.winfo_reqheight()

    toolbar_h = toolbar.winfo_height() if toolbar is not None else 0
    ctrl_h = ctrl.winfo_reqheight()

    total_w = max(c_w, 480)
    total_h = c_h + toolbar_h + ctrl_h + 10

    # Set fixed geometry and prevent resizing to keep the window stable
    try:
        win.geometry(f"{total_w}x{total_h}")
        win.minsize(total_w, total_h)
        win.maxsize(total_w, total_h)
        win.resizable(False, False)
    except Exception:
        # If any of the geometry calls fail, continue without locking
        pass

    return fig, ax
