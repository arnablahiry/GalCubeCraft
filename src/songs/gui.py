"""SONGS GUI

Compact Tkinter-based GUI to interactively configure and run the
``SONGS`` generator. Provides a three-column layout of parameter
frames, crisp LaTeX-rendered labels, convenience sliders, and utility
buttons (Generate, Slice, Moments, Spectrum, Save, New). Plotting and file
I/O are intentionally kept out of the generator core; the GUI imports
top-level visualisation helpers (``moment0``, ``moment1``, ``spectrum``,
``slice_view``) to display results.

Design notes
------------
- Lightweight: the GUI focuses on inspection and quick interactive
    experimentation, not production batch runs.
- Threading: generation runs in a background thread so the UI remains
    responsive; generated figures are produced by the visualise helpers.
- Cleanup: LaTeX labels are rendered to temporary PNG files (via
    matplotlib) and tracked in ``_MATH_TEMPFILES`` for removal when the
    application exits.

Usage
-----
Run the module as a script to display the GUI::

    python -m songs.gui

Or instantiate :class:`SONGSGUI` and call ``mainloop()``. The GUI
expects the package to be importable (it will try a fallback path insertion
when executed as a script)."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pickle
import threading
import numpy as np
import matplotlib
# Use Agg backend to avoid Tkinter threading issues
# Figures will still display properly when show() is called
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os
import sys
from PIL import Image, ImageTk

# Track latex PNG tempfiles for cleanup
_MATH_TEMPFILES = []

import warnings

# Or suppress ALL UserWarnings if you prefer a cleaner log
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Tweakable parameter frames 
# ---------------------------
def param_frame(parent, padding=8, border_color="#797979", bg="#303030", width=None, height=80, do_pack=True):
    """Create a framed parameter panel used throughout the GUI."""
    outer = tk.Frame(parent, bg=border_color)
    if do_pack:
        outer.pack(padx=4, pady=4)
    inner = tk.Frame(outer, bg=bg, padx=padding, pady=padding)
    if width or height:
        inner.config(width=width, height=height)
        inner.pack_propagate(False)
    inner.pack(fill='both', expand=True)
    return outer, inner




def latex_label(parent, latex, font_size=2, bg=None, color="white"):
    """Render a LaTeX string to a crisp Tkinter ``Label`` using Matplotlib.

    The routine renders the supplied LaTeX expression using Matplotlib's
    mathtext renderer to a high-DPI temporary PNG, crops the image tightly
    around the rendered text and returns a Tk ``Label`` containing the
    resulting image. This approach yields sharp text on high-DPI displays
    without requiring a full TeX installation.

    Important behaviour and performance notes
    -----------------------------------------
    - Each call creates a temporary PNG file; filenames are appended to the
      module-level ``_MATH_TEMPFILES`` list so they can be removed when the
      application exits. Callers should ensure the GUI's cleanup routine
      calls ``os.remove`` on these files (the main GUI does this in
      ``_on_close``).
    - Rendering is moderately expensive (Matplotlib figure creation and
      rasterisation). Cache or reuse labels for static text where possible.
    - The function forces a very high DPI (default 500) and crops the
      image tightly which keeps runtime acceptable while producing crisp
      output.

    Parameters
    ----------
    parent : tk.Widget
        Parent widget to attach the returned ``Label`` to.
    latex : str
        The LaTeX expression (without surrounding dollar signs) to render.
    font_size : int, optional
        Point-size used for rendering text (passed to Matplotlib).

    Returns
    -------
    tk.Label
        A Tk ``Label`` widget containing the rendered LaTeX as an image.

    Example
    -------
    >>> lbl = latex_label(frame, r"\\alpha + \\beta = \\gamma", font_size=12)
    >>> lbl.pack()

    """
    import tkinter as tk
    import matplotlib.pyplot as plt
    from PIL import Image, ImageTk
    import tempfile

    # Render at high DPI
    DPI = 500

    # Minimal figure; we will crop
    fig = plt.figure(figsize=(1, 1), dpi=DPI)
    fig.patch.set_alpha(0.0)

    text = fig.text(0.5, 0.5, f"${latex}$",
                    fontsize=font_size,
                    ha="center", va="center",
                    color=color)

    # Draw and compute tight bounding box
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer).expanded(1.1, 1.2)

    # Save tightly-cropped
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=DPI, transparent=True,
                bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()),
                pad_inches=0.0)
    plt.close(fig)

    # Load image → convert to RGBA
    img = Image.open(tmp.name).convert("RGBA")
    
    # Keep the PIL image in memory to avoid file access issues
    img.load()

    # Direct Tk image (no scaling)
    photo = ImageTk.PhotoImage(img)

    lbl_kw = dict(image=photo, borderwidth=0)
    if bg:
        lbl_kw['bg'] = bg
    label = tk.Label(parent, **lbl_kw)
    # Store both the PhotoImage AND the PIL image to prevent premature GC
    label.image = photo
    label._pil_image = img
    _MATH_TEMPFILES.append(tmp.name)

    label.pack()

    return label


# Import core
try:
    from .core import SONGSPhy, DEFAULT_DIFFUSE_PARAMS
except Exception:
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from songs.core import SONGSPhy, DEFAULT_DIFFUSE_PARAMS

# Import visualise helpers (module provides moment0, moment1, spectrum)
try:
    from .visualise import *
except Exception:
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from songs.visualise import *

import sys
import tkinter as tk
from tkinter import ttk

class TextRedirector:
    """Redirect writes into a Tk ``Text`` widget behaving like a stream.

    Use this helper to capture and display program output inside the GUI
    (for example, to show progress logs, exceptions, or print() output).
    ``TextRedirector`` implements a minimal stream interface (``write`` and
    ``flush``) so it can be assigned directly to ``sys.stdout`` or
    ``sys.stderr``; written text is inserted into the provided Tk Text
    widget and scrolled to the end so the latest output is visible.

    Threading note
    --------------
    - The class itself is not thread-safe: writes coming from background
      threads should be marshalled to the Tk mainloop (e.g. via
      ``widget.after(...)``) if there is a risk of concurrent access.

    Parameters
    ----------
    widget : tk.Text
        The Tk Text widget where text will be appended.
    tag : str, optional
        Optional text tag name to apply to inserted text (default ``'stdout'``).

    Example
    -------
    Redirect stdout into a Text widget::

        txt = tk.Text(root)
        txt.pack()
        sys.stdout = TextRedirector(txt, tag='log')

    """

    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, string):
        self.widget.configure(state="normal")
        self.widget.insert("end", string, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def flush(self):
        pass  # Needed for compatibility with sys.stdout

class LogWindow(tk.Toplevel):
    """Top-level log window that captures and displays stdout/stderr.

    ``LogWindow`` creates a simple resizable Toplevel containing a Tk
    ``Text`` widget and installs ``TextRedirector`` instances on
    ``sys.stdout`` and ``sys.stderr`` so that all subsequent ``print``
    output and uncaught exception tracebacks are visible in the GUI. The
    window restores the original streams when closed.

    Behaviour
    ---------
    - Creating an instance replaces ``sys.stdout`` and ``sys.stderr`` in
        the running interpreter until the window is closed (``on_close``).
    - The window configures a separate text tag for ``stderr`` so error
        messages are coloured differently.

    Example
    -------
    >>> log = LogWindow(root)
    >>> log.deiconify()  # show the window

    """

    def __init__(self, master):
        super().__init__(master)
        self.title("Logs")
        self.text = tk.Text(self)
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("stderr", foreground="#e55b5b")
        # Redirect stdout and stderr
        sys.stdout = TextRedirector(self.text, "stdout")
        sys.stderr = TextRedirector(self.text, "stderr")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        # Optionally restore stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.destroy()


class SONGSGUI(tk.Tk):
    """Main GUI application for interactively configuring and running
    SONGS simulations.

    This class implements a compact, self-contained Tk application that
    exposes the most commonly-used parameters of the generator via a
    three-column layout of parameter panels. Controls include numeric
    sliders, textual inputs and convenience buttons that invoke high-level
    visualisation helpers (``moment0``, ``moment1``, ``spectrum``) or
    persist generated results to disk.

    Key behaviour
    --------------
    - The generator is constructed from the current UI values and stored
        on ``self.generator``. Calling ``Generate`` runs the generator in a
        background daemon thread so the UI remains responsive; generated
        results become available via ``self.generator.results``.
    - Visualisation buttons call into functions defined in
        :mod:`songs.visualise` which create Matplotlib figures; these
        functions are intentionally separate from the generator core so the
        GUI remains a thin orchestration layer.
    - Temporary files created by :func:`latex_label` are tracked in the
        module-level ``_MATH_TEMPFILES`` list and cleaned up when the GUI is
        closed via ``_on_close``.

    Threading and shutdown
    ----------------------
    - Generation and save operations spawn background daemon threads. The
        UI schedules finalisation callbacks back on the main thread using
        ``self.after(...)`` when worker threads complete.
    - Closing the main window triggers a cleanup of temporary files and
        forces process termination to avoid orphaned interpreters. If you
        prefer a softer shutdown that joins worker threads, modify
        ``_on_close`` accordingly.

    Usage example
    -------------
    Run the GUI as a script::

            python -m songs.gui

    Or instantiate from Python::

            from songs.gui import SONGSGUI
            app = SONGSGUI()
            app.mainloop()

    """

    def __init__(self):
        super().__init__()
        self.title('SONGS GUI')
        self.WINDOW_HEIGHT = 550
        self.configure(bg='#0a0a0a')
        self.resizable(False, False)

        # Window icon
        try:
            _icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'songs_icon.png'))
            if not os.path.exists(_icon_path):
                _icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets', 'songs_icon.png'))
            _icon_img = Image.open(_icon_path).convert('RGBA')
            self._icon_photo = ImageTk.PhotoImage(_icon_img)
            self.iconphoto(True, self._icon_photo)
        except Exception:
            pass

        # Create a hidden log window immediately
        self.log_window = LogWindow(self)
        self.log_window.withdraw()
        self._is_closing = False

        # Horizontal root layout: logo strip on left, right column (cards + buttons) on right
        self._root_frame = tk.Frame(self, bg='#0a0a0a')
        self._root_frame.pack(fill='both', expand=True)

        # Left: vertical logo strip — spans full window height
        logo_strip = tk.Frame(self._root_frame, bg='#0a0a0a')
        logo_strip.pack(side='left', fill='y')
        try:
            logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'songs_dark_vertical.png'))
            if not os.path.exists(logo_path):
                logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets', 'songs_dark_vertical.png'))
            _logo_img = Image.open(logo_path).convert('RGBA')
            _logo_w = int(self.WINDOW_HEIGHT * _logo_img.width / _logo_img.height)
            _logo_img = _logo_img.resize((_logo_w, self.WINDOW_HEIGHT), Image.LANCZOS)
            self._logo_photo = ImageTk.PhotoImage(_logo_img)
            tk.Label(logo_strip, image=self._logo_photo, bg='#0a0a0a', borderwidth=0).pack()
        except Exception:
            tk.Label(logo_strip, text='SONGS', bg='#0a0a0a', fg='#b8960a',
                     font=('Helvetica', 11, 'bold'), wraplength=80).pack(pady=20)

        # Right column: cards on top, buttons at bottom (logo not included)
        right_col = tk.Frame(self._root_frame, bg='#0a0a0a')
        right_col.pack(side='left', fill='both', expand=True)

        # Button bar at the bottom of the right column only
        self._btn_area = tk.Frame(right_col, bg='#0a0a0a')
        self._btn_area.pack(side='bottom', fill='x')

        # Cards container fills remaining space above buttons
        self.container = tk.Frame(right_col, bg='#0a0a0a')
        self.container.pack(side='top', fill='both', expand=True, padx=4, pady=4)

        self.generator = None

        self._build_widgets()

        # Fix height to WINDOW_HEIGHT; width = logo + content (measured after layout).
        self.update_idletasks()
        total_w = self.winfo_reqwidth()
        self.geometry(f"{total_w}x{self.WINDOW_HEIGHT}")
        self.resizable(False, False)

        self.protocol('WM_DELETE_WINDOW', self._on_close)



    # ---------------------------
    # Slider helper
    # ---------------------------
    def make_slider(self, parent, label, var, from_, to,
                    resolution=0.01, fmt="{:.2f}", integer=False):
        """Create a labelled slider widget with snapping and a value label."""
        # Colours — fall back to safe defaults before _build_widgets sets them.
        bg      = getattr(self, '_slider_bg',     "#111111")
        fg      = getattr(self, '_slider_fg',     "#999999")
        acc     = getattr(self, '_slider_accent',  "#b8960a")
        trough  = getattr(self, '_slider_trough',  "#111111")

        _border_col = getattr(self, '_slider_border', "#785605FF")
        _wrap = tk.Frame(parent, bg=_border_col, padx=1, pady=1)
        fr = tk.Frame(_wrap, bg=bg)
        fr.pack(fill='both', expand=True)
        if label:
            tk.Label(fr, text=label, bg=bg, fg=fg,
                     font=("Helvetica", 8)).pack(anchor='w', pady=(0,2))
        slider_row = tk.Frame(fr, bg=bg)
        slider_row.pack(fill='x')
        val_lbl = tk.Label(slider_row, text=fmt.format(var.get()),
                           width=6, anchor="e", bg=bg, fg=acc,
                           font=("Helvetica", 8))
        val_lbl.pack(side='right', padx=(4, 0))
        _thumb  = getattr(self, '_slider_thumb',  "#b8960a")   # yellow thumb
        _thumbh = getattr(self, '_slider_thumbhover', "#f0c040")  # bright on hover
        scale = tk.Scale(slider_row, from_=from_, to=to, orient='horizontal',
                         resolution=resolution,
                         bg=_thumb, fg=fg, troughcolor=trough,
                         activebackground=_thumbh, highlightthickness=0,
                         sliderrelief='flat', bd=0, showvalue=False,
                         relief='flat', width=6)
        scale.pack(side='left', fill='x', expand=True)
        step = resolution if resolution else 0.01
        busy = {'val': False}

        def snap(v):
            if integer:
                return int(round(float(v)))
            nsteps = round((float(v) - from_) / step)
            return from_ + nsteps * step

        def update(v):
            if busy['val']: return
            busy['val'] = True
            v_snap = snap(v)
            try: var.set(v_snap)
            except Exception: pass
            try: val_lbl.config(text=fmt.format(v_snap))
            except Exception: val_lbl.config(text=str(v_snap))
            try: scale.set(v_snap)
            except Exception: pass
            busy['val'] = False

        scale.configure(command=update)
        try: scale.set(var.get())
        except Exception: scale.set(from_)

        try:
            def _var_trace(*_):
                if busy['val']: return
                busy['val'] = True
                v = var.get()
                try: val_lbl.config(text=fmt.format(v))
                except Exception: val_lbl.config(text=str(v))
                try: scale.set(v)
                except Exception: pass
                busy['val'] = False
            if hasattr(var, 'trace_add'):
                var.trace_add('write', _var_trace)
            else:
                var.trace('w', _var_trace)
        except Exception: pass
        return _wrap


    # ---------------------------
    # Button callback methods
    # ---------------------------
    def show_logs(self):
        if hasattr(self, 'log_window') and self.log_window.winfo_exists():
            self.log_window.lift()
        else:
            self.log_window = LogWindow(self)



    def _popup_figure(self, title, fig):
        """Utility to put a matplotlib figure into a new popup window"""
        new_win = tk.Toplevel(self)
        new_win.title(title)
        
        # Use the FigureCanvasTkAgg to embed the plot
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=new_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def show_moments(self):
        if not self.generator:
            return
        try:
            # Generate the figures using the 'Agg' backend (already set)
            fig0, _ = moment0(self.generator.results, idx=0, save=False)
            self._popup_figure("Moment 0", fig0)
            
            fig1, _ = moment1(self.generator.results, idx=0, save=False)
            self._popup_figure("Moment 1", fig1)
        except Exception as e:
            print(f"Error displaying moments: {e}")

    def show_spectra(self):
        if not self.generator:
            return
        try:
            fig, _ = spectrum(self.generator.results, idx=0, save=False)
            self._popup_figure("Integrated Spectrum", fig)
        except Exception as e:
            print(f"Error displaying spectrum: {e}")


    def show_slice(self):
        """Open the SONGS SliceViewer for the first generated cube."""
        if self.generator and self.generator.results:
            try:
                SliceViewer(self, self.generator.results, idx=0)
            except Exception as e:
                messagebox.showerror('Slice viewer error', str(e))

    def show_mom1(self):
        if self.generator:
            fig, ax = moment1(self.generator.results, idx=0, save=False)
            try: 
                import matplotlib
                matplotlib.use('TkAgg')
                plt.figure(fig.number)
                plt.show(block=False)
                matplotlib.use('Agg')
            except Exception: 
                pass

    '''def show_spectra(self):
        if self.generator:
            fig, ax = spectrum(self.generator.results, idx=0, save=False)
            try: 
                import matplotlib
                matplotlib.use('TkAgg')
                plt.figure(fig.number)
                plt.show(block=False)
                matplotlib.use('Agg')
            except Exception: 
                pass'''

    def reset_instance(self):
        """Reset the GUI to a fresh state and disable visualisation/save.

        This clears the in-memory ``self.generator`` reference so that the
        next generate action will create a new instance from current UI
        values. Buttons that depend on generated results are disabled.
        """
        # Disable all result buttons
        for _b in (self.moments_btn, self.spectra_btn, self.slice_btn,
                   self.save_btn, self.new_instance_btn):
            try:
                self._disable_btn(_b)
            except Exception:
                pass
        for child in self.winfo_children():
            if isinstance(child, tk.Toplevel):
                child.destroy()

        self.generator = None

    def _find_scale_in(self, widget):
        """Recursively find a ttk.Scale inside a widget tree.

        Returns the first found Scale or None.
        """
        if isinstance(widget, ttk.Scale):
            return widget
        for c in widget.winfo_children():
            found = self._find_scale_in(c)
            if found is not None:
                return found
        return None

    def _set_sliders_enabled(self, enabled=True):
        """Enable or disable all slider widgets present in the GUI.

        This toggles the internal ttk.Scale widget state for each slider
        frame we create in :meth:`_build_widgets`.
        """
        names = [
            'r_slider', 'n_slider', 'hz_slider', 'sigma_slider',
            'grid_slider', 'spec_slider', 'angle_x_slider', 'angle_y_slider',
            'sat_offset_slider_frame'
        ]
        for name in names:
            w = getattr(self, name, None)
            if w is None:
                continue
            try:
                scale = self._find_scale_in(w)
                if scale is None:
                    continue
                if enabled:
                    try:
                        scale.state(['!disabled'])
                    except Exception:
                        scale.configure(state=tk.NORMAL)
                else:
                    try:
                        scale.state(['disabled'])
                    except Exception:
                        scale.configure(state=tk.DISABLED)
            except Exception:
                # Best-effort: ignore any widget-specific errors
                pass
        

   

    # ---------------------------
    # Build all widgets
    # ---------------------------
    def _build_widgets(self):

        """Build and layout all GUI widgets.

        This method assembles the complete UI inside the scrollable
        container: it defines Tk variables, creates the three-column
        parameter panels (rows 1--6), the slider widgets, and the bottom
        utility buttons (Generate, Moment0, Moment1, Spectra, Save, New).

        The method also hooks variable traces to an auto-update helper so
        that changing parameters in the UI will keep an internal
        ``SONGS`` generator in sync for quick inspection.

        Notes
        -----
        - This method focuses on layout and widget creation; no heavy
            computation is performed here.
        - For clarity we keep layout logic (pack) local to this helper so
            other methods can assume the widgets exist after this call.
        """
        
        # ---------------------------
        # Variables
        # ---------------------------
        self.bmin_var = tk.DoubleVar(value=11.0)
        self.bmaj_var = tk.DoubleVar(value=13.0)
        self.bpa_var = tk.DoubleVar(value=20.0)
        self.spatial_resolution = tk.DoubleVar(value=3.8)
        self.n_var = tk.DoubleVar(value=1.0)
        self.hz_var = tk.DoubleVar(value=0.8)
        self.Se_var = tk.DoubleVar(value=0.1)
        self.sigma_v_var = tk.DoubleVar(value=40.0)
        self.fov = tk.IntVar(value=275)
        self.spectral_resolution = tk.IntVar(value=20)
        self.angle_x_var = tk.IntVar(value=45)
        self.angle_y_var = tk.IntVar(value=30)
        self.n_gals_var = tk.IntVar(value=1)

        # --- Diffuse-emission knobs (defaults pulled from core's DEFAULT_DIFFUSE_PARAMS) ---
        dp = DEFAULT_DIFFUSE_PARAMS
        # Halo
        self.halo_Se_factor_var = tk.DoubleVar(value=float(dp.get('halo_Se_factor', 0.065)))
        self.halo_Re_factor_var = tk.DoubleVar(value=float(dp.get('halo_Re_factor', 3.0)))
        self.halo_sigma_vz_var  = tk.DoubleVar(value=float(dp.get('halo_sigma_vz', 70.0)))
        # Bridges
        self.bridge_Se_factor_var          = tk.DoubleVar(value=float(dp.get('bridge_Se_factor', 0.05)))
        self.bridge_width_start_factor_var = tk.DoubleVar(value=float(dp.get('bridge_width_start_factor', 1.5)))
        self.bridge_width_end_factor_var   = tk.DoubleVar(value=float(dp.get('bridge_width_end_factor', 1.0)))
        # Tails / streamers
        self.tail_Se_factor_var     = tk.DoubleVar(value=float(dp.get('tail_Se_factor', 0.4)))
        # Streamer (channel-traversing trajectory) extras
        self.tail_vel_gradient_var          = tk.DoubleVar(value=float(dp.get('tail_vel_gradient', 0.5)))


        # New: satellite size fraction (max satellite-to-central ratio for Re,
        # hz, Se). Greyed out when only one galaxy is requested.
        self.sat_brightness_frac_var = tk.DoubleVar(value=0.15)

        # ── Colour scheme ──────────────────────────────────────────────────────
        _BG          = "#0a0a0a"   # window / logo strip background
        _CARD_BG     = "#111111"   # small control card background
        _BIG_BG      = "#0d0d0d"   # big card interior background
        _BIG_BORDER  = "#403100"   # dark yellow — visible but subtle border
        _SM_BORDER   = "#201800"   # very low opacity yellow — tiny border tint
        _TEXT        = "#999999"   # normal text
        _ACCENT      = "#b8960a"   # faint yellow accent (headings, values)
        _ENTRY_BG    = "#1a1a1a"
        _FONT_SM     = ("Helvetica", 8)
        _FONT_HDR    = ("Helvetica", 9, "bold")

        # Expose colors to make_slider via instance attrs
        self._slider_bg         = _CARD_BG
        self._slider_fg         = _TEXT
        self._slider_accent     = _ACCENT
        self._slider_trough     = _CARD_BG
        self._slider_thumb      = _ACCENT        # yellow thumb
        self._slider_thumbhover = "#f0c040"      # brighter yellow on hover
        self._slider_border     = "#5e4200"      # low opacity yellow border (slightly visible)

        self.configure(bg=_BG)
        self._root_frame.configure(bg=_BG)

        # Make ttk slider trough match the small card background
        _style = ttk.Style()
        _style.configure('Horizontal.TScale', troughcolor=_CARD_BG, background=_CARD_BG)
        _style.configure('TScale', troughcolor=_CARD_BG, background=_CARD_BG)

        col_width = 200  # small card fixed width

        # Helper used multiple times below to find the underlying ttk.Scale
        # inside a slider frame (so we can grey it out when n_gals == 1).
        def find_scale(widget):
            if isinstance(widget, (tk.Scale, ttk.Scale)):
                return widget
            for child in widget.winfo_children():
                result = find_scale(child)
                if result is not None:
                    return result
            return None

        def big_card(parent, title):
            """Bordered card with thin, low-opacity yellow outline and LaTeX title."""
            outer = tk.Frame(parent, bg=_BIG_BORDER, padx=1, pady=1)
            outer.pack(side='left', fill='both', expand=True, padx=6, pady=6)
            inner = tk.Frame(outer, bg=_BIG_BG)
            inner.pack(fill='both', expand=True)
            tk.Label(inner, text=title, bg=_BIG_BG, fg=_ACCENT,
                     font=_FONT_HDR).pack(anchor='w', padx=8, pady=(6,2))
            sep = tk.Frame(inner, bg=_BIG_BORDER, height=1)
            sep.pack(fill='x', padx=6, pady=(0,6))
            return inner

        def small_card(parent, latex_text=None):
            outer = tk.Frame(parent, bg=_SM_BORDER, padx=1, pady=1)
            outer.pack(fill='x', padx=6, pady=2)
            inner = tk.Frame(outer, bg=_CARD_BG, padx=6, pady=4)
            inner.pack(fill='both', expand=True)
            if latex_text:
                latex_label(inner, latex_text, font_size=1.8,
                            bg=_CARD_BG, color="white").pack(anchor='w')
            return inner

        # ── Horizontal big-card row ─────────────────────────────────────────
        cards_row = tk.Frame(self.container, bg=_BG)
        cards_row.pack(fill='both', expand=True)
        for c in range(4):
            cards_row.columnconfigure(c, weight=1)

        # ──────────────────────────────────────────────────────────────────────
        # BIG CARD 1: Initialisation Parameters
        # ──────────────────────────────────────────────────────────────────────
        bc1 = big_card(cards_row, "Initialisation Parameters")

        sc = small_card(bc1, r"\text{Number of galaxies}")
        rb_frame = tk.Frame(sc, bg=_CARD_BG)
        rb_frame.pack(anchor='w', pady=2)
        for val in range(1, 7):
            rb = tk.Radiobutton(rb_frame, text=str(val), variable=self.n_gals_var, value=val,
                                bg=_CARD_BG, fg="white", selectcolor=_ACCENT,
                                activebackground=_CARD_BG, activeforeground=_ACCENT,
                                font=("Helvetica", 9, "bold"), relief='flat', bd=0,
                                highlightthickness=0, indicatoron=1)
            rb.pack(side='left', padx=3)

        sc = small_card(bc1, r"\text{Spatial Resolution}\;(\Delta_{X,Y})\;[\rm kpc\;px^{-1}]")
        self.pix_scale_var_slider = self.make_slider(sc, "", self.spatial_resolution, 0.72, 9.0, resolution=0.01, fmt="{:.2f}")
        self.pix_scale_var_slider.pack(fill='x')

        sc = small_card(bc1, r"\text{Spectral Resolution}\;(\Delta_{v_z})\;[\rm km\,s^{-1}]")
        self.spec_slider = self.make_slider(sc, "", self.spectral_resolution, 5, 40, resolution=5, fmt="{:d}", integer=True)
        self.spec_slider.pack(fill='x')

        sc = small_card(bc1, r"\text{Field of View [px]}")
        self.fov_slider = self.make_slider(sc, "", self.fov, 64, 512, resolution=1, fmt="{:d}", integer=True)
        self.fov_slider.pack(fill='x')

        sc = small_card(bc1, r"\text{Beam}\ [\rm kpc,\,kpc,\,deg]")
        for latex_lbl, var, lo, hi, res in [
            (r"B_{\rm min}", self.bmin_var, 1.0, 30.0, 0.1),
            (r"B_{\rm maj}", self.bmaj_var, 1.0, 30.0, 0.1),
            (r"\rm BPA",     self.bpa_var,  0.0, 90.0, 1.0),
        ]:
            beam_row = tk.Frame(sc, bg=_CARD_BG)
            beam_row.pack(fill='x', pady=1)
            latex_label(beam_row, latex_lbl, font_size=1.6,
                        bg=_CARD_BG, color=_TEXT).pack(side='left', padx=(0, 4))
            sl = self.make_slider(beam_row, "", var, lo, hi, resolution=res, fmt="{:.1f}")
            sl.pack(side='left', fill='x', expand=True)

        # ──────────────────────────────────────────────────────────────────────
        # BIG CARD 2: Galaxy Properties
        # ──────────────────────────────────────────────────────────────────────
        bc2 = big_card(cards_row, "Central Galaxy Properties")

        sc = small_card(bc2, r"\text{Sérsic index}\;(n)")
        self.n_slider = self.make_slider(sc, "", self.n_var, 0.5, 1.5, resolution=0.01, fmt="{:.3f}")
        self.n_slider.pack(fill='x')

        sc = small_card(bc2, r"\text{Scale height}\;(h_z)\;[\rm kpc]")
        self.hz_slider = self.make_slider(sc, "", self.hz_var, 0.4, 9.0, resolution=0.01, fmt="{:.3f}")
        self.hz_slider.pack(fill='x')

        sc = small_card(bc2, r"\text{Surface brightness}\;(S_e)\;[\rm Jy]")
        self.Se_slider = self.make_slider(sc, "", self.Se_var, 0.01, 1.0, resolution=0.01, fmt="{:.3f}")
        self.Se_slider.pack(fill='x')

        sc = small_card(bc2, r"\text{Velocity dispersion}\;(\sigma_{v_z})\;[\rm km\,s^{-1}]")
        self.sigma_slider = self.make_slider(sc, "", self.sigma_v_var, 30.0, 60.0, resolution=0.1, fmt="{:.1f}")
        self.sigma_slider.pack(fill='x')

        sc = small_card(bc2, r"\text{Inclination angle}\;(\theta_X)\;[\rm deg]")
        self.angle_x_slider = self.make_slider(sc, "", self.angle_x_var, 0, 359, resolution=1, fmt="{:d}", integer=True)
        self.angle_x_slider.pack(fill='x')

        sc = small_card(bc2, r"\text{Azimuthal angle}\;(\phi_Y)\;[\rm deg]")
        self.angle_y_slider = self.make_slider(sc, "", self.angle_y_var, 0, 359, resolution=1, fmt="{:d}", integer=True)
        self.angle_y_slider.pack(fill='x')

        # ──────────────────────────────────────────────────────────────────────
        # BIG CARD 3: Satellite & Halo
        # ──────────────────────────────────────────────────────────────────────
        bc3 = big_card(cards_row, "Satellite Properties")
        self._bc3 = bc3  # kept for greyout

        sc = small_card(bc3, r"\text{Satellite flux fraction of central}")
        self.sat_frac_slider_frame = self.make_slider(sc, "", self.sat_brightness_frac_var, 0.0, 0.5, resolution=0.01, fmt="{:.2f}")
        self.sat_frac_slider_frame.pack(fill='x')
        self.sat_frac_scale = find_scale(self.sat_frac_slider_frame)

        self.sat_offset_max_var = tk.DoubleVar(value=180.0)
        self.sat_offset_min_var = tk.DoubleVar(value=80.0)

        sc = small_card(bc3, r"\text{Satellite offset max [kpc]}")
        self.sat_offset_max_frame = self.make_slider(sc, "", self.sat_offset_max_var, 10.0, 500.0, resolution=5.0, fmt="{:.0f}")
        self.sat_offset_max_frame.pack(fill='x')
        self.sat_offset_max_scale = find_scale(self.sat_offset_max_frame)

        sc = small_card(bc3, r"\text{Satellite offset min [kpc]}")
        self.sat_offset_min_frame = self.make_slider(sc, "", self.sat_offset_min_var, 0.0, 333.0, resolution=5.0, fmt="{:.0f}")
        self.sat_offset_min_frame.pack(fill='x')
        self.sat_offset_min_scale = find_scale(self.sat_offset_min_frame)

        # ──────────────────────────────────────────────────────────────────────
        # BIG CARD 4: Diffuse Features (halo + bridge + streamers)
        # ──────────────────────────────────────────────────────────────────────
        bc4 = big_card(cards_row, "Diffuse Features")

        sc = small_card(bc4, r"S_{e,\rm halo}\,/\,S_{e,c}")
        self.halo_Se_slider = self.make_slider(sc, "", self.halo_Se_factor_var, 0.0, 0.3, resolution=0.005, fmt="{:.3f}")
        self.halo_Se_slider.pack(fill='x')

        sc = small_card(bc4, r"R_{e,\rm halo}\,/\,R_{e,c}")
        self.halo_Re_slider = self.make_slider(sc, "", self.halo_Re_factor_var, 1.0, 5.0, resolution=0.1, fmt="{:.1f}")
        self.halo_Re_slider.pack(fill='x')

        sc = small_card(bc4, r"\sigma_{v_z,\rm halo}\;[\rm km\,s^{-1}]")
        self.halo_sigma_slider = self.make_slider(sc, "", self.halo_sigma_vz_var, 0.0, 150.0, resolution=5.0, fmt="{:.0f}")
        self.halo_sigma_slider.pack(fill='x')

        sc = small_card(bc4, r"S_{e,\rm br}\,/\,\min(S_{e,c},\,S_{e,s})")
        self.bridge_Se_slider = self.make_slider(sc, "", self.bridge_Se_factor_var, 0.0, 0.3, resolution=0.005, fmt="{:.3f}")
        self.bridge_Se_slider.pack(fill='x')

        sc = small_card(bc4, r"\sigma_{\rm bridge,\,halo}\,/\,R_{e,c}")
        self.bridge_w0_slider = self.make_slider(sc, "", self.bridge_width_start_factor_var, 0.5, 4.0, resolution=0.1, fmt="{:.1f}")
        self.bridge_w0_slider.pack(fill='x')

        sc = small_card(bc4, r"\sigma_{\rm bridge,\,sat}\,/\,R_{e,s}")
        self.bridge_w1_slider = self.make_slider(sc, "", self.bridge_width_end_factor_var, 0.3, 3.0, resolution=0.1, fmt="{:.1f}")
        self.bridge_w1_slider.pack(fill='x')

        sc = small_card(bc4, r"S_{e,\rm tail}\,/\,S_{e,s}")
        self.tail_Se_slider = self.make_slider(sc, "", self.tail_Se_factor_var, 0.0, 1.0, resolution=0.02, fmt="{:.2f}")
        self.tail_Se_slider.pack(fill='x')

        sc = small_card(bc4, r"\text{Streamer vel. scale}\;(\times\,\Delta v_\text{sys})")
        self.tail_vel_grad_slider = self.make_slider(sc, "", self.tail_vel_gradient_var, 0.0, 2.0, resolution=0.05, fmt="{:.2f}")
        self.tail_vel_grad_slider.pack(fill='x')

        # ── Satellite-dependent greyout ──────────────────────────────────────
        def _update_min_range(*args):
            new_upper = max(5.0, float(self.sat_offset_max_var.get()) / 1.5)
            self.sat_offset_min_scale.configure(to=new_upper)
            if float(self.sat_offset_min_var.get()) > new_upper:
                self.sat_offset_min_var.set(round(new_upper / 5) * 5)
        self.sat_offset_max_var.trace_add('write', _update_min_range)
        _update_min_range()

        def _compute_max_offset_kpc():
            fov_px  = int(self.fov.get())
            res_kpc = max(int(self.spatial_resolution.get()), 1)
            gs      = max(fov_px // res_kpc, 4)
            igs     = int((31 / 64) * gs)
            if igs % 2 != 0:
                igs -= 1
            half    = igs // 2
            center  = (gs + 1) // 2
            max_off_px = max(min(gs - half - 1 - center, center - half) - 1, 1)
            return max_off_px * res_kpc

        def _update_offset_range(*_args):
            cap = _compute_max_offset_kpc()
            new_max_upper = float(cap)
            new_min_upper = max(1.0, new_max_upper / 1.5)
            self.sat_offset_max_scale.configure(to=new_max_upper)
            self.sat_offset_min_scale.configure(to=new_min_upper)
            if float(self.sat_offset_max_var.get()) > new_max_upper:
                self.sat_offset_max_var.set(round(new_max_upper * 0.8 / 5) * 5)
            if float(self.sat_offset_min_var.get()) > new_min_upper:
                self.sat_offset_min_var.set(round(new_min_upper * 0.5 / 5) * 5)

        _update_offset_range()
        for _v in (self.fov, self.spatial_resolution):
            if hasattr(_v, 'trace_add'):
                _v.trace_add('write', _update_offset_range)
            else:
                _v.trace('w', _update_offset_range)

        def _update_sat_dependent(*args):
            active = self.n_gals_var.get() > 1
            state  = tk.NORMAL if active else tk.DISABLED
            dim_fg = "#2a2a1a"   # dimmed text — keep bg unchanged so design stays intact

            def _set_state(w):
                # Only dim foreground text when inactive; leave backgrounds alone.
                try:
                    w.configure(state=state)
                except Exception: pass
                if not active:
                    try: w.configure(fg=dim_fg)
                    except Exception: pass
                else:
                    # Restore fg by widget type
                    if isinstance(w, tk.Scale):
                        try: w.configure(fg=_TEXT, bg=_ACCENT, activebackground="#f0c040")
                        except Exception: pass
                    elif isinstance(w, tk.Label):
                        try: w.configure(fg=_ACCENT if getattr(w, '_is_val_lbl', False) else _TEXT)
                        except Exception: pass
                for child in w.winfo_children():
                    _set_state(child)

            _set_state(self._bc3)

        _update_sat_dependent()
        if hasattr(self.n_gals_var, 'trace_add'):
            self.n_gals_var.trace_add('write', _update_sat_dependent)
        else:
            self.n_gals_var.trace('w', _update_sat_dependent)



        # ---------------------------
        # Generate & utility buttons (Generate, Slice, Moments, Spectrum, Save, New)
        # ---------------------------
        btn_frame = self._btn_area
        btn_frame.configure(padx=8, pady=6)

        # tk.Label buttons — Labels always respect bg/fg on macOS unlike tk.Button
        def _mk_btn(parent, text, cmd, bg="#1a1400", fg="#b8960a",
                    hov="#2e2400", disabled=False):
            lbl = tk.Label(parent, text=text,
                           bg=bg, fg=fg,
                           font=("Helvetica", 10, "bold"),
                           padx=14, pady=10, cursor='hand2')
            lbl.pack(side='left', padx=4, expand=True, fill='x')
            if disabled:
                lbl.configure(fg="#333322", cursor='arrow')
            else:
                lbl.bind('<Enter>', lambda e, b=lbl, h=hov: b.configure(bg=h))
                lbl.bind('<Leave>', lambda e, b=lbl, n=bg: b.configure(bg=n))
                lbl.bind('<Button-1>', lambda e: cmd())
            lbl._btn_bg   = bg
            lbl._btn_hov  = hov
            lbl._btn_cmd  = cmd
            lbl._disabled = disabled
            return lbl

        def _enable_btn(lbl, bg=None, hov=None):
            bg  = bg  or lbl._btn_bg
            hov = hov or lbl._btn_hov
            lbl.configure(bg=bg, fg="#b8960a" if bg != "#6b3800" else "white",
                          cursor='hand2')
            lbl._disabled = False
            lbl.bind('<Enter>', lambda e, b=lbl, h=hov: b.configure(bg=h))
            lbl.bind('<Leave>', lambda e, b=lbl, n=bg:  b.configure(bg=n))
            lbl.bind('<Button-1>', lambda e: lbl._btn_cmd())

        def _disable_btn(lbl):
            lbl.configure(fg="#333322", cursor='arrow')
            lbl._disabled = True
            lbl.unbind('<Enter>')
            lbl.unbind('<Leave>')
            lbl.unbind('<Button-1>')

        # Store helpers so enable/disable still work for the rest of the code
        self._enable_btn  = _enable_btn
        self._disable_btn = _disable_btn

        self.generate_btn     = _mk_btn(btn_frame, 'Generate', self.generate,
                                        bg="#6b3800", fg="white", hov="#8b4a00")
        self.slice_btn        = _mk_btn(btn_frame, 'Slice',    self.show_slice,    disabled=True)
        self.moments_btn      = _mk_btn(btn_frame, 'Moments',  self.show_moments,  disabled=True)
        self.spectra_btn      = _mk_btn(btn_frame, 'Spectrum', self.show_spectra,  disabled=True)
        self.save_btn         = _mk_btn(btn_frame, 'Save',     self.save_sim,      disabled=True)
        self.new_instance_btn = _mk_btn(btn_frame, 'Reset',    self.reset_instance, disabled=True)


       

        # Auto-create/refresh generator when variables change (fast preview)
        def _auto_update_generator(*args):
            try:
                self.create_generator()
            except Exception as e:
                print("Auto-create generator failed:", e)

        for var in [self.bmin_var, self.bmaj_var, self.bpa_var, self.spatial_resolution, self.n_var,
                    self.hz_var, self.Se_var, self.sigma_v_var, self.fov,
                    self.spectral_resolution, self.angle_x_var, self.angle_y_var,
                    self.sat_brightness_frac_var, self.sat_offset_min_var, self.sat_offset_max_var,
                    # Diffuse-emission knobs
                    self.halo_Se_factor_var, self.halo_Re_factor_var,
                    self.halo_sigma_vz_var,
                    self.bridge_Se_factor_var, self.bridge_width_start_factor_var,
                    self.bridge_width_end_factor_var,
                    self.tail_Se_factor_var,
                    self.tail_vel_gradient_var]:
            if hasattr(var, 'trace_add'):
                var.trace_add('write', _auto_update_generator)
            else:
                var.trace('w', _auto_update_generator)


    # ---------------------------
    # Parameter collection & generator
    # ---------------------------

    
    def _collect_parameters(self):
        """Read current UI controls and return a parameter dict.

        The returned dictionary mirrors the small set of fields used by the
        :class:`SONGS` constructor and the GUI. Values are converted
        to plain Python / NumPy types where appropriate.

        Returns
        -------
        params : dict
            Dictionary containing keys like ``beam_info``, ``n_gals``,
            ``grid_size``, ``n_spectral_slices``, ``all_Re``, ``all_hz``,
            ``all_Se``, ``all_n``, and ``sigma_v``. This dict is consumed by
            :meth:`create_generator` and used when saving.
        """

        bmin = float(self.bmin_var.get())
        bmaj = float(self.bmaj_var.get())
        bpa = float(self.bpa_var.get())
        n_gals = int(self.n_gals_var.get())
        fov = int(self.fov.get())
        spectral_resolution = int(self.spectral_resolution.get())
        spatial_resolution = int(self.spatial_resolution.get())
        central_n = float(self.n_var.get())
        central_hz = float(self.hz_var.get())
        central_Se = float(self.Se_var.get())
        central_gal_x_angle = int(self.angle_x_var.get())
        central_gal_y_angle = int(self.angle_y_var.get())
        offset_gals = (float(self.sat_offset_min_var.get()), float(self.sat_offset_max_var.get()))
        sigma_v = float(self.sigma_v_var.get())

        # Create per-galaxy lists. For a single galaxy we keep the
        # specified central values. For multiple galaxies we generate
        # satellite properties using simple random draws so the
        # generator receives arrays of length ``n_gals`` (primary + satellites).
        all_Re = [5/spatial_resolution]
        all_hz = [central_hz]
        all_Se = [central_Se]
        all_gal_x_angles = [central_gal_x_angle]
        all_gal_y_angles = [central_gal_y_angle]
        all_n = [central_n]

        if n_gals > 1:
            n_sat = n_gals - 1
            rng = np.random.default_rng()

            # Satellites are physically smaller (Re, hz fixed ratio).
            sat_Re = list(rng.uniform(all_Re[0] / 3, all_Re[0] / 2, n_sat))
            sat_hz = list(rng.uniform(all_hz[0] / 3, all_hz[0] / 2, n_sat))

            # sat_brightness_frac scales Se relative to central, compensating for
            # the smaller satellite Re so that frac=1 ≈ similar surface brightness.
            _b = float(np.clip(self.sat_brightness_frac_var.get(), 0.0, 2.0))

            # Random Sérsic indices for satellites
            sat_n = list(rng.uniform(0.5, 1.5, n_sat))

            sat_Se = [
                float(all_Se[0] * _b * (all_Re[0] / re_sat) ** 2
                      * rng.uniform(0.85, 1.15))
                for re_sat in sat_Re
            ]

            # Random orientations for satellites (degrees)
            sat_x_angles = list(rng.uniform(-180.0, 180.0, n_sat))
            sat_y_angles = list(rng.uniform(-180.0, 180.0, n_sat))

            all_Re += sat_Re
            all_hz += sat_hz
            all_Se += sat_Se
            all_n += sat_n
            all_gal_x_angles += sat_x_angles
            all_gal_y_angles += sat_y_angles

        # Convert lists to NumPy arrays to match generator expectations
        all_Re = np.array(all_Re)
        all_hz = np.array(all_hz)
        all_Se = np.array(all_Se)
        all_n = np.array(all_n)
        all_gal_x_angles = np.array(all_gal_x_angles)
        all_gal_y_angles = np.array(all_gal_y_angles)
        
        # Compose a `diffuse_params` dict from the GUI controls, layered on
        # top of the package defaults so we never silently drop any key the
        # core helper expects.
        diffuse_params = dict(DEFAULT_DIFFUSE_PARAMS)
        diffuse_params.update({
            'enabled': True,
            'halo_Se_factor': float(self.halo_Se_factor_var.get()),
            'halo_Re_factor': float(self.halo_Re_factor_var.get()),
            'halo_sigma_vz': float(self.halo_sigma_vz_var.get()),
            'bridge_Se_factor': float(self.bridge_Se_factor_var.get()),
            'bridge_width_start_factor': float(self.bridge_width_start_factor_var.get()),
            'bridge_width_end_factor': float(self.bridge_width_end_factor_var.get()),
            'tail_Se_factor': float(self.tail_Se_factor_var.get()),
            # Streamer knobs
            'tail_vel_gradient': float(self.tail_vel_gradient_var.get()),
        })

        params = dict(
                    beam_info=[bmin,bmaj,bpa],
                    n_gals=n_gals,
                    fov=fov,
                    spectral_resolution=spectral_resolution,
                    spatial_resolution=spatial_resolution,
                    all_Re=np.array(all_Re),
                    all_hz=np.array(all_hz),
                    all_Se=np.array(all_Se),
                    all_n=np.array(all_n),
                    all_gal_x_angles=np.array(all_gal_x_angles),
                    all_gal_y_angles=np.array(all_gal_y_angles),
                    sigma_v=sigma_v,
                    offset_gals=offset_gals,
                    diffuse_params=diffuse_params,
                )
        return params

    def create_generator(self):
        """Instantiate a :class:`SONGS` object from current UI values.

        The method calls :meth:`_collect_parameters` to assemble a parameter
        dictionary and then constructs a single-cube generator instance with
        sensible defaults for fields not exposed directly in the GUI. After
        construction the per-galaxy attributes on the generator are filled
        from the collected parameters so the generator is ready to run.
        """

        params = self._collect_parameters()
        _instance_seed = getattr(self, '_pending_seed', None)
        try:
            g = SONGSPhy(
                n_gals=params['n_gals'],
                n_cubes=1,
                spatial_resolution=params['spatial_resolution'],
                spectral_resolution=params['spectral_resolution'],
                offset_gals=params['offset_gals'],
                beam_info=params['beam_info'],
                fov=params['fov'],
                verbose=True,
                seed=_instance_seed,
                diffuse_params=params['diffuse_params'],
            )
        except Exception as e:
            messagebox.showerror('Error', f'Failed to create SONGS: {e}')
            return

        # Fill the galaxy-specific properties
        n_g = params['n_gals']
        g.all_Re = [params['all_Re']]
        g.all_hz = [params['all_hz']]
        g.all_Se = [params['all_Se']]
        g.all_n = [params['all_n']]
        g.all_gal_x_angles = [params['all_gal_x_angles']]
        g.all_gal_y_angles = [params['all_gal_y_angles']]
        g.all_gal_vz_sigmas = [np.full(n_g, params['sigma_v'])]
        #g.all_pix_spatial_scales = [np.full(n_g, params['spatial_resolution'])]
        g.all_gal_v_0 = [np.full(n_g, 200.0)]  # default systemic velocity

        self.generator = g


    def _run_generate(self):
        # Disable garbage collection in this thread to prevent cleanup
        # of Tkinter objects from the wrong thread
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()
        
        try:
            # Check if closing before doing expensive work
            if self._is_closing:
                return
                
            # Auto-show log window
            if hasattr(self, 'log_window') and self.log_window.winfo_exists():
                self.log_window.deiconify()
                self.log_window.lift()
            else:
                self.log_window = LogWindow(self)

            try:
                results = self.generator.generate_cubes()
                # Check again before scheduling UI updates
                if self._is_closing:
                    return
                # Enable buttons on main thread
                def _enable_all():
                    for _b in (self.moments_btn, self.spectra_btn, self.slice_btn,
                               self.save_btn, self.new_instance_btn):
                        try: self._enable_btn(_b)
                        except Exception: pass
                self.after(0, _enable_all)
            except Exception as e:
                if not self._is_closing:
                    self.after(0, lambda e=e: messagebox.showerror('Error during generation', str(e)))
        finally:
            # Re-enable garbage collection if it was enabled
            if gc_was_enabled:
                gc.enable()
    
    
    def generate(self):
        import random as _random
        self._pending_seed = _random.randint(0, 2**31 - 1)
        print(f"[SONGS] Instance seed: {self._pending_seed}")
        self.create_generator()

        if self.generator is None:
            return

        t = threading.Thread(target=self._run_generate, daemon=True)
        t.start()

    # ---------------------------
    # Save simulation (cube + params)
    # ---------------------------
    def save_sim(self):
        """Generate (if needed) and save the sim tuple (cube, params).

        This runs generation in a background thread and then opens a
        Save-As dialog on the main thread to let the user choose where
        to store the result. We support .npz (numpy savez) and .pkl
        (pickle) formats; complex parameter dicts fall back to pickle.
        """
        # If we already have generated results, save them directly without
        # re-running the (potentially expensive) generation. Otherwise,
        # fall back to running generation in background and then prompting
        # the user to save.
        try:
            has_results = bool(self.generator and getattr(self.generator, 'results', None))
        except Exception:
            has_results = False

        if has_results:
            # Use existing results (do not re-run generation)
            results = self.generator.results
            # extract first cube/meta
            cube = None
            meta = None
            if isinstance(results, (list, tuple)) and len(results) > 0:
                first = results[0]
                if isinstance(first, tuple) and len(first) >= 2:
                    cube, meta = first[0], first[1]
                else:
                    cube = first
            else:
                cube = results

            params = self._collect_parameters()
            # Prompt on main thread
            self.after(0, lambda: self._save_sim_prompt(cube, params, meta))
            return

        # No existing results: run generation in background then prompt to save
        if self.generator is None:
            # create generator from current GUI values
            self.create_generator()
            if self.generator is None:
                return

        t = threading.Thread(target=self._save_sim_thread, daemon=True)
        t.start()

    def _save_sim_thread(self):
        """Background worker that runs generation and then prompts to save.

        Runs ``self.generator.generate_cubes()`` in the background thread and
        then schedules :meth:`_save_sim_prompt` on the main thread to show the
        Save-As dialog. Errors are displayed via a messagebox scheduled on
        the main thread.
        """
        # Disable garbage collection in this thread to prevent cleanup
        # of Tkinter objects from the wrong thread
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()
        
        try:
            # Check if closing before doing expensive work
            if self._is_closing:
                return

            try:
                results = self.generator.generate_cubes()
            except Exception as e:
                if not self._is_closing:
                    self.after(0, lambda e=e: messagebox.showerror('Error during generation', str(e)))
                return

            # Check again after generation completes
            if self._is_closing:
                return

            # extract first cube and params
            cube = None
            meta = None
            if isinstance(results, (list, tuple)) and len(results) > 0:
                first = results[0]
                if isinstance(first, tuple) and len(first) >= 2:
                    cube, meta = first[0], first[1]
                else:
                    cube = first
            else:
                cube = results

            params = self._collect_parameters()

            # prompt/save on main thread
            if not self._is_closing:
                self.after(0, lambda: self._save_sim_prompt(cube, params, meta))
        finally:
            # Re-enable garbage collection if it was enabled
            if gc_was_enabled:
                gc.enable()

    def _save_sim_prompt(self, cube, params, meta=None):
        """Prompt the user for a filename and save the provided cube/params.

        Parameters
        ----------
        cube : ndarray
            Spectral cube array to save.
        params : dict
            Parameters dictionary produced by :meth:`_collect_parameters`.
        meta : dict or None
            Optional metadata returned by the generator.
        """

        # Ask for filename
        fname = filedialog.asksaveasfilename(
            defaultextension='.h5',
            filetypes=[
                ('HDF5 file', '.h5'),
                ('NumPy archive', '.npz'),
                ('Pickled Python object', '.pkl'),
            ],
        )
        if not fname:
            return

        try:
            if fname.lower().endswith('.h5') or fname.lower().endswith('.hdf5'):
                import h5py
                with h5py.File(fname, 'w') as f:
                    f.create_dataset('cube', data=cube)
                    g = f.create_group('galaxies')
                    if meta is not None and 'per_galaxy_cubes' in meta:
                        try:
                            g.create_dataset('cubes', data=np.array(meta['per_galaxy_cubes']))
                        except Exception:
                            pass
                    if meta is not None and 'galaxy_centers' in meta:
                        try:
                            g.create_dataset('positions_xyz_px', data=np.array(meta['galaxy_centers']))
                        except Exception:
                            pass
                    n_gals = int(params['n_gals'])
                    types = np.array(['central'] + ['satellite'] * (n_gals - 1), dtype='S10')
                    g.create_dataset('types', data=types)
                    g.create_dataset('Re_px', data=np.asarray(params['all_Re']))
                    g.create_dataset('Se', data=np.asarray(params['all_Se']))
                    g.create_dataset('hz_px', data=np.asarray(params['all_hz']))
                    f.attrs['n_gals'] = n_gals
                    f.attrs['n_satellites'] = n_gals - 1
                    f.attrs['spatial_resolution_kpc_per_px'] = float(params['spatial_resolution'])
                    f.attrs['spectral_resolution_km_s'] = float(params['spectral_resolution'])
                    f.attrs['fov_kpc'] = float(params['fov'])
                    dp_grp = f.create_group('diffuse_params')
                    for k, v in params.get('diffuse_params', {}).items():
                        try:
                            dp_grp.attrs[k] = v
                        except Exception:
                            pass
            elif fname.lower().endswith('.npz'):
                # try to prepare a flat dict for savez
                save_dict = {}
                save_dict['cube'] = cube
                # flatten params into arrays where possible
                for k, v in params.items():
                    try:
                        if isinstance(v, (list, tuple)):
                            save_dict[k] = np.array(v)
                        else:
                            save_dict[k] = v
                    except Exception:
                        save_dict[k] = v
                # include meta if available
                if meta is not None:
                    try:
                        save_dict['meta'] = meta
                    except Exception:
                        pass
                np.savez(fname, **save_dict)
            else:
                with open(fname, 'wb') as fh:
                    pickle.dump((cube, params, meta), fh)
        except Exception as e:
            messagebox.showerror('Save error', f'Failed to save simulation: {e}')
            return

        messagebox.showinfo('Saved', f'Simulation saved to {fname}')

    # ---------------------------
    # Cleanup
    # ---------------------------
    def _on_close(self):
        """Cleanup temporary files created for LaTeX rendering and exit.

        Sets a flag to stop background threads from scheduling UI updates,
        removes any temporary PNG files recorded in ``_MATH_TEMPFILES``,
        and performs a graceful shutdown of the Tkinter application.
        """
        # Signal threads to stop scheduling UI updates
        self._is_closing = True
        
        # Clean up temporary files
        for p in list(_MATH_TEMPFILES):
            try: 
                os.remove(p)
            except: 
                pass
        
        # Graceful Tkinter shutdown
        try:
            self.quit()  # Stop the mainloop
        except Exception:
            pass
        
        try:
            self.destroy()  # Destroy all widgets
        except Exception:
            pass


def main():
    app = SONGSGUI()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure cleanup happens
        try:
            app._is_closing = True
            app.quit()
        except:
            pass
        try:
            app.destroy()
        except:
            pass

if __name__ == '__main__':
    main()