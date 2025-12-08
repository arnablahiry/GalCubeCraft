"""GalCubeCraft GUI

This module implements a compact Tkinter-based GUI used to interactively
configure and run the ``GalCubeCraft`` generator. It provides a three-column
layout of parameter frames, LaTeX-rendered labels, convenience slider
widgets and a small set of utility buttons (Generate, Moment0, Moment1,
Spectra, Save, New). The implementation intentionally keeps plotting and
file IO out of the generator core; the GUI imports the top-level visualisation
helpers (``moment0``, ``moment1``, ``spectrum``) to display results.

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

        python -m GalCubeCraft.gui

Or instantiate :class:`GalCubeCraftGUI` from another script and call
``mainloop()``. The GUI expects the package to be importable (it will try a
fallback path insertion when executed as a script).
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pickle
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os
import sys
from PIL import Image, ImageTk

# Track latex PNG tempfiles for cleanup
_MATH_TEMPFILES = []


# ---------------------------
# Tweakable parameter frames 
# ---------------------------
def param_frame(parent, padding=8, border_color="#797979", bg="#303030", width=None, height=80):
    """Create a framed parameter panel used throughout the GUI.

    The function returns a tuple ``(outer, inner)`` where ``outer`` is a
    thin border frame (useful to provide a coloured outline) and ``inner`` is
    the content frame where widgets should be placed. ``inner`` uses the
    provided padding and can be sized with ``width``/``height`` while
    preserving layout via ``pack_propagate(False)`` when explicit dimensions
    are given.

    Parameters
    ----------
    parent : tk.Widget
        Parent widget to attach the frames to.
    padding : int, optional
        Internal padding inside the inner frame.
    border_color : str, optional
        Background colour used for the outer border frame.
    bg : str, optional
        Background colour for the inner content frame.
    width, height : int or None, optional
        Optional fixed size for the inner frame. When provided, the inner
        frame will not resize to its children.

    Returns
    -------
    (outer, inner) : tuple
        Outer border frame and inner content frame.
    """

    outer = tk.Frame(parent, bg=border_color)
    outer.pack(padx=4, pady=4)  # <--- pack the outer here
    inner = tk.Frame(outer, bg=bg, padx=padding, pady=padding)
    if width or height:
        inner.config(width=width, height=height)
        inner.pack_propagate(False)
    inner.pack(fill='both', expand=True)
    return outer, inner




def latex_label(parent, latex, font_size=5):
    """Render a short LaTeX string to a Tkinter-compatible image label.

    This helper uses Matplotlib to render inline LaTeX to a temporary PNG
    file which is then opened with Pillow and wrapped in a Tk ``Label``.
    Temporary filenames are recorded in ``_MATH_TEMPFILES`` and removed by
    :meth:`GalCubeCraftGUI._on_close` when the application exits.

    Parameters
    ----------
    parent : tk.Widget
        Parent widget for the returned ``Label``.
    latex : str
        LaTeX string (without surrounding $ signs) to render.
    font_size : int, optional
        Font size passed to Matplotlib when rendering.

    Returns
    -------
    tk.Label
        A Tkinter Label containing the rendered image. The PhotoImage is
        stored on the widget as ``label.image`` to avoid garbage collection.
    """

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=font_size, color='white')
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    _MATH_TEMPFILES.append(tmp.name)
    img = Image.open(tmp.name)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(parent, image=photo, borderwidth=0)
    label.image = photo
    return label

# Import core
try:
    from .core import GalCubeCraft
except Exception:
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from GalCubeCraft.core import GalCubeCraft

# Import visualise helpers (module provides moment0, moment1, spectrum)
try:
    from .visualise import *
except Exception:
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from GalCubeCraft.visualise import *

import sys
import tkinter as tk
from tkinter import ttk

class TextRedirector:
    """A tiny stream-like object that redirects writes into a Tk Text widget.

    Instances of this class mimic a text stream by providing ``write`` and
    ``flush`` methods so they can be assigned to ``sys.stdout`` and
    ``sys.stderr``. Written text is inserted into the supplied widget and
    the view is scrolled to the end.
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
    """Simple top-level window that shows redirected stdout/stderr.

    The window installs TextRedirector instances to capture global
    ``sys.stdout`` and ``sys.stderr`` so that print() calls are visible to
    the user. Closing the window restores the original streams.
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


class GalCubeCraftGUI(tk.Tk):
    """Main GUI application window for interactively configuring GalCubeCraft.

    The class provides methods to assemble a :class:`GalCubeCraft` instance
    from UI controls, run generation in a background thread, display simple
    visualisations (via the top-level visualise helpers) and save generated
    results to disk. The UI is split into reusable parameter frames that
    keep the layout compact and consistent.
    """

    def __init__(self):
        super().__init__()
        self.title('GalCubeCraft GUI')
        self.WINDOW_WIDTH = 600
        self.WINDOW_HEIGHT = 800
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.resizable(False, False)
        # Create a hidden log window immediately
        self.log_window = LogWindow(self)
        self.log_window.withdraw()  # Hide it until "Logs" button clicked



        # Banner (same as before)...
        try:
            banner_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'cubecraft.png'))
            if not os.path.exists(banner_path):
                banner_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets', 'cubecraft.png'))
            original_img = Image.open(banner_path).convert("RGBA")
            target_width = self.WINDOW_WIDTH - 0
            aspect = original_img.height / original_img.width
            resized = original_img.resize((target_width, int(target_width * aspect)), Image.LANCZOS)
            self.banner_image = ImageTk.PhotoImage(resized)
            banner_lbl = ttk.Label(self, image=self.banner_image)
            banner_lbl.pack(pady=(8,6))
        except Exception:
            ttk.Label(self, text="GalCubeCraft", font=('Helvetica', 18, 'bold')).pack(pady=(8,6))

        
        # Scrollable canvas setup (same as before)...
        self.main_canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side='right', fill='y')
        self.main_canvas.pack(fill='both', expand=True)
        self.container = ttk.Frame(self.main_canvas)
        self.window = self.main_canvas.create_window((0,0), window=self.container, anchor='nw')
        self.container.bind('<Configure>', lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox('all')))
        self.main_canvas.bind('<Configure>', lambda e: self.main_canvas.itemconfig(self.window, width=e.width))

        

        # Generator
        self.generator = None

        # Build 3-column layout
        self._build_widgets()

        self.protocol('WM_DELETE_WINDOW', self._on_close)



    # ---------------------------
    # Slider helper
    # ---------------------------
    def make_slider(self, parent, label, var, from_, to,
                    resolution=0.01, fmt="{:.2f}", integer=False):
        """Create a labelled slider widget with snapping and a value label.

        Returns a small frame containing a horizontal ``ttk.Scale`` and a
        right-aligned textual value display. The function attaches a trace
        to ``var`` so programmatic updates are reflected in the slider and
        vice versa.
        """

        fr = ttk.Frame(parent)
        if label:
            ttk.Label(fr, text=label).pack(anchor='w', pady=(0,2))
        slider_row = ttk.Frame(fr)
        slider_row.pack(fill='x')
        val_lbl = ttk.Label(slider_row, text=fmt.format(var.get()), width=6, anchor="e")
        val_lbl.pack(side='right', padx=(4,0))
        scale = ttk.Scale(slider_row, from_=from_, to=to, orient='horizontal')
        scale.pack(side='left', fill='x', expand=True)
        step = resolution if resolution else 0.01
        busy = {'val':False}
        def snap(v):
            if integer:
                return int(round(float(v)))
            nsteps = round((float(v)-from_)/step)
            return from_ + nsteps*step
        def update(v):
            if busy['val']: return
            busy['val']=True
            v_snap = snap(v)
            try: var.set(v_snap)
            except Exception: pass
            try: val_lbl.config(text=fmt.format(v_snap))
            except Exception: val_lbl.config(text=str(v_snap))
            try: scale.set(v_snap)
            except Exception: pass
            busy['val']=False
        scale.configure(command=update)
        try: scale.set(var.get())
        except Exception: scale.set(from_)
        try:
            def _var_trace(*_):
                if busy['val']: return
                busy['val']=True
                v = var.get()
                try: val_lbl.config(text=fmt.format(v))
                except Exception: val_lbl.config(text=str(v))
                try: scale.set(v)
                except Exception: pass
                busy['val']=False
            if hasattr(var, 'trace_add'):
                var.trace_add('write', _var_trace)
            else:
                var.trace('w', _var_trace)
        except Exception: pass
        return fr


    # ---------------------------
    # Button callback methods
    # ---------------------------
    def show_logs(self):
        if hasattr(self, 'log_window') and self.log_window.winfo_exists():
            self.log_window.lift()
        else:
            self.log_window = LogWindow(self)


    def show_mom0(self):
        """Display the moment0 (integrated intensity) for the first cube.

        Calls the top-level :func:`moment0` function with the generator's
        results. If no generator has been created yet the method is a no-op.
        """

        if self.generator:
            fig, ax = moment0(self.generator.results, idx=0, save=False)
            try: fig.show()
            except Exception: pass

    def show_mom1(self):
        if self.generator:
            fig, ax = moment1(self.generator.results, idx=0, save=False)
            try: fig.show()
            except Exception: pass

    def show_spectra(self):
        if self.generator:
            fig, ax = spectrum(self.generator.results, idx=0, save=False)
            try: fig.show()
            except Exception: pass

    def reset_instance(self):
        """Reset the GUI to a fresh state and disable visualisation/save.

        This clears the in-memory ``self.generator`` reference so that the
        next generate action will create a new instance from current UI
        values. Buttons that depend on generated results are disabled.
        """
        # Disable all except generate
        self.mom0_btn.config(state='disabled')
        self.mom1_btn.config(state='disabled')
        self.spectra_btn.config(state='disabled')
        # Also disable Save when starting a fresh instance
        try:
            self.save_btn.config(state='disabled')
        except Exception:
            pass
        self.generator = None
        
    # ---------------------------
    # Build all widgets
    # ---------------------------
    def _build_widgets(self):
        # ---------------------------
        # Variables
        # ---------------------------
        self.bmin_var = tk.DoubleVar(value=4.0)
        self.bmaj_var = tk.DoubleVar(value=4.0)
        self.bpa_var = tk.DoubleVar(value=0.0)
        self.r_var = tk.DoubleVar(value=1.0)
        self.n_var = tk.DoubleVar(value=1.0)
        self.hz_var = tk.DoubleVar(value=0.8)
        self.Se_var = tk.DoubleVar(value=0.1)
        self.sigma_v_var = tk.DoubleVar(value=40.0)
        self.grid_size_var = tk.IntVar(value=125)
        self.n_spectral_var = tk.IntVar(value=40)
        self.angle_x_var = tk.IntVar(value=45)
        self.angle_y_var = tk.IntVar(value=30)
        self.n_gals_var = tk.IntVar(value=1)

        col_width = 280  # column width

        # ---------------------------
        # Row 1: Number of galaxies + Satellite offset
        # ---------------------------
        r1 = ttk.Frame(self.container)
        r1.pack(fill='x', pady=6)

        # Number of galaxies frame
        outer1, fr1 = param_frame(r1, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')
        latex_label(fr1, r"\text{Number of galaxies}").pack(anchor='w', pady=(0,6))
        rb_frame = ttk.Frame(fr1)
        rb_frame.pack(anchor='w')
        for val in range(1, 6):
            rb = ttk.Radiobutton(rb_frame, text=str(val), variable=self.n_gals_var, value=val)
            rb.pack(side='left', padx=4)

        # Satellite offset frame
        outer2, fr2 = param_frame(r1, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\text{Satellite offset from centre (px)}").pack(anchor='w', pady=(0,6))
        # When creating the slider, keep a reference to the Scale
        # Create slider
        self.sat_offset_var = tk.DoubleVar(value=5.0)
        self.sat_offset_slider_frame = self.make_slider(
            fr2, "", self.sat_offset_var, 3.0, 10.0, resolution=0.1, fmt="{:.1f}"
        )
        self.sat_offset_slider_frame.pack(fill='x')

        # Find the ttk.Scale inside the frame
        def find_scale(widget):
            if isinstance(widget, ttk.Scale):
                return widget
            for child in widget.winfo_children():
                result = find_scale(child)
                if result is not None:
                    return result
            return None

        self.sat_offset_scale = find_scale(self.sat_offset_slider_frame)

        # Disable initially if n_gals = 1
        if self.n_gals_var.get() == 1:
            self.sat_offset_scale.state(['disabled'])

        # Auto-enable/disable slider based on n_gals
        def _update_sat_offset(*args):
            active = self.n_gals_var.get() > 1
            if active:
                self.sat_offset_scale.state(['!disabled'])
            else:
                self.sat_offset_scale.state(['disabled'])

        if hasattr(self.n_gals_var, 'trace_add'):
            self.n_gals_var.trace_add('write', _update_sat_offset)
        else:
            self.n_gals_var.trace('w', _update_sat_offset)


        # ---------------------------
        # Row 2: Beam + r_var
        # ---------------------------
        r2 = ttk.Frame(self.container)
        r2.pack(fill='x', pady=6)

        # Beam frame
        outer1, fr1 = param_frame(r2, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')

        # LaTeX label for beam
        latex_label(fr1, r"\text{Beam Info [px + deg]}").pack(anchor='w', pady=(0,6))

        beam_row = ttk.Frame(fr1)
        beam_row.pack(anchor='w', pady=2)

        # Smaller width for entry boxes
        entry_width = 3

        # LaTeX labels + entries
        for text, var in [(r"B_{\rm min}", self.bmin_var),
                        (r"B_{\rm maj}", self.bmaj_var),
                        (r"\rm BPA", self.bpa_var)]:
            lbl = latex_label(beam_row, text, font_size=5)
            lbl.pack(side='left', padx=(0,2))
            e = ttk.Entry(beam_row, textvariable=var, width=entry_width)
            e.pack(side='left', padx=(0,6))


        # r_var frame
        outer2, fr2 = param_frame(r2, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\text{Resolution } r").pack(anchor='w')
        self.r_slider = self.make_slider(fr2, "", self.r_var, 0.35, 4.0, resolution=0.01, fmt="{:.2f}")
        self.r_slider.pack(fill='x')

        # ---------------------------
        # Row 3: Sérsic n + Scale height
        # ---------------------------
        r3 = ttk.Frame(self.container)
        r3.pack(fill='x', pady=6)

        outer1, fr1 = param_frame(r3, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')
        latex_label(fr1, r"\text{Sérsic index } n_{\text{Sérsic}}").pack(anchor='w')
        self.n_slider = self.make_slider(fr1, "", self.n_var, 0.5, 1.5, resolution=0.01, fmt="{:.3f}")
        self.n_slider.pack(fill='x')

        outer2, fr2 = param_frame(r3, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\text{Scale height } h_z \ (\text{px})").pack(anchor='w')
        self.hz_slider = self.make_slider(fr2, "", self.hz_var, 0.3, 1.0, resolution=0.01, fmt="{:.3f}")
        self.hz_slider.pack(fill='x')

        # ---------------------------
        # Row 4: Central base S_e + sigma_v
        # ---------------------------
        r4 = ttk.Frame(self.container)
        r4.pack(fill='x', pady=6)

        outer1, fr1 = param_frame(r4, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')
        latex_label(fr1, r"\text{Central base } S_e \ (\text{flux units})").pack(anchor='w')
        ttk.Entry(fr1, textvariable=self.Se_var).pack(fill='x')

        outer2, fr2 = param_frame(r4, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\sigma_{v,z}\ (\text{km/s})").pack(anchor='w')
        self.sigma_slider = self.make_slider(fr2, "", self.sigma_v_var, 30.0, 60.0, resolution=0.1, fmt="{:.1f}")
        self.sigma_slider.pack(fill='x')

        # ---------------------------
        # Row 5: Grid size + Spectral slices
        # ---------------------------
        r5 = ttk.Frame(self.container)
        r5.pack(fill='x', pady=6)

        outer1, fr1 = param_frame(r5, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')
        latex_label(fr1, r"\text{Grid size (}n_x=n_y\text{)}").pack(anchor='w')
        self.grid_slider = self.make_slider(fr1, "", self.grid_size_var, 64, 256, resolution=1, fmt="{:d}", integer=True)
        self.grid_slider.pack(fill='x')

        outer2, fr2 = param_frame(r5, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\text{Spectral slices (channels)}").pack(anchor='w')
        self.spec_slider = self.make_slider(fr2, "", self.n_spectral_var, 32, 128, resolution=1, fmt="{:d}", integer=True)
        self.spec_slider.pack(fill='x')

        # ---------------------------
        # Row 6: Inclination + Position angles
        # ---------------------------
        r6 = ttk.Frame(self.container)
        r6.pack(fill='x', pady=6)

        outer1, fr1 = param_frame(r6, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')
        latex_label(fr1, r"\text{Inclination angle (X) [deg]}").pack(anchor='w')
        self.angle_x_slider = self.make_slider(fr1, "", self.angle_x_var, 0, 359, resolution=1, fmt="{:d}", integer=True)
        self.angle_x_slider.pack(fill='x')

        outer2, fr2 = param_frame(r6, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\text{Position angle (Y) [deg]}").pack(anchor='w')
        self.angle_y_slider = self.make_slider(fr2, "", self.angle_y_var, 0, 359, resolution=1, fmt="{:d}", integer=True)
        self.angle_y_slider.pack(fill='x')

        # ---------------------------
        # Generate & utility buttons
        # ---------------------------
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side='bottom', pady=8, fill='x')

        # Button height (bigger than normal)
        btn_height = 2
        # Create buttons as ttk with a compact dark style so we don't change
        # the global theme but render dark buttons reliably on macOS.
        btn_fg = 'white'
        btn_disabled_fg = '#8c8c8c'
        btn_bg = '#222222'
        btn_active_bg = '#2f2f2f'

        style = ttk.Style()
        # Do not change the global theme; just define a local style
        style.configure('Dark.TButton', background=btn_bg, foreground=btn_fg, height=btn_height, padding=(6,4))
        style.map('Dark.TButton',
                  background=[('active', btn_active_bg), ('disabled', btn_bg), ('!disabled', btn_bg)],
                  foreground=[('disabled', btn_disabled_fg), ('!disabled', btn_fg)])

        # Create as ttk.Button with the dark style (keeps rest of theme intact)
        self.generate_btn = ttk.Button(btn_frame, text='Generate', command=self.generate, style='Dark.TButton')
        self.mom0_btn = ttk.Button(btn_frame, text='Moment0', command=self.show_mom0, state='disabled', style='Dark.TButton')
        self.mom1_btn = ttk.Button(btn_frame, text='Moment1', command=self.show_mom1, state='disabled', style='Dark.TButton')
        self.spectra_btn = ttk.Button(btn_frame, text='Spectra', command=self.show_spectra, state='disabled', style='Dark.TButton')
        self.new_instance_btn = ttk.Button(btn_frame, text='New', command=self.reset_instance, state='disabled', style='Dark.TButton')

        # Pack buttons side by side with padding
        # Place the Save button immediately before the New button (New at the end)
        self.save_btn = ttk.Button(btn_frame, text='Save', command=self.save_sim, state='disabled', style='Dark.TButton')
        for btn in [self.generate_btn, self.mom0_btn, self.mom1_btn, self.spectra_btn, self.save_btn, self.new_instance_btn]:
            btn.pack(side='left', padx=4, pady=2, expand=True, fill='x')


       

        # Auto-update generator when variables change
        def _auto_update_generator(*args):
            try:
                self.create_generator()
            except Exception as e:
                print("Auto-create generator failed:", e)

        for var in [self.bmin_var, self.bmaj_var, self.bpa_var, self.r_var, self.n_var,
                    self.hz_var, self.Se_var, self.sigma_v_var, self.grid_size_var,
                    self.n_spectral_var, self.angle_x_var, self.angle_y_var]:
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
        :class:`GalCubeCraft` constructor and the GUI. Values are converted
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
        r = float(self.r_var.get())
        grid_size = int(self.grid_size_var.get())
        n_spectral = int(self.n_spectral_var.get())
        central_n = float(self.n_var.get())
        central_hz = float(self.hz_var.get())
        central_Se = float(self.Se_var.get())
        central_gal_x_angle = int(self.angle_x_var.get())
        central_gal_y_angle = int(self.angle_y_var.get())
        sigma_v = float(self.sigma_v_var.get())
        # Create per-galaxy lists. For a single galaxy we keep the
        # specified central values. For multiple galaxies we generate
        # satellite properties using simple random draws so the
        # generator receives arrays of length ``n_gals`` (primary +
        # satellites).
        all_Re = [r * bmin]
        all_hz = [central_hz]
        all_Se = [central_Se]
        all_gal_x_angles = [central_gal_x_angle]
        all_gal_y_angles = [central_gal_y_angle]

        if n_gals > 1:
            n_sat = n_gals - 1
            # Use a local RNG for reproducibility control if needed
            rng = np.random.default_rng()

            # Satellites are smaller and fainter than the primary
            all_Re += list(rng.uniform(all_Re[0] / 3.0, all_Re[0] / 2.0, n_sat))
            all_hz += list(rng.uniform(all_hz[0] / 3.0, all_hz[0] / 2.0, n_sat))
            all_Se += list(rng.uniform(all_Se[0] / 3.0, all_Se[0] / 2.0, n_sat))

            # Random orientations for satellites (degrees)
            all_gal_x_angles += list(rng.uniform(-180.0, 180.0, n_sat))
            all_gal_y_angles += list(rng.uniform(-180.0, 180.0, n_sat))

        central_Re_kpc = rng.uniform(4, 6)  # Central Re in kpc
        pix_spatial_scale = central_Re_kpc / all_Re[0]



        params = dict(
            beam_info=[bmin, bmaj, bpa],
            n_gals=n_gals,
            grid_size=grid_size,
            n_spectral_slices=n_spectral,
            all_Re=np.array(all_Re),
            all_hz=np.array(all_hz),
            all_Se=np.array(all_Se),
            central_n=central_n,
            all_gal_x_angles=np.array(all_gal_x_angles),
            all_gal_y_angles=np.array(all_gal_y_angles),
            sigma_v=sigma_v,
            pix_spatial_scale=pix_spatial_scale,
            r=r
        )
        return params

    def create_generator(self):
        """Instantiate a :class:`GalCubeCraft` object from current UI values.

        The method calls :meth:`_collect_parameters` to assemble a parameter
        dictionary and then constructs a single-cube generator instance with
        sensible defaults for fields not exposed directly in the GUI. After
        construction the per-galaxy attributes on the generator are filled
        from the collected parameters so the generator is ready to run.
        """

        params = self._collect_parameters()
        try:
            g = GalCubeCraft(
                n_gals=params['n_gals'],
                n_cubes=1,
                resolution=params['r'],        # use the correct key
                beam_info=params['beam_info'],
                grid_size=params['grid_size'],
                n_spectral_slices=params['n_spectral_slices'],
                n_sersic=params['central_n'],
                verbose=True,
                seed=None
            )
        except Exception as e:
            messagebox.showerror('Error', f'Failed to create GalCubeCraft: {e}')
            return

        # Fill the galaxy-specific properties
        n_g = params['n_gals']
        # pixel spatial scales per galaxy
        g.all_pix_spatial_scales = np.full(n_g, params['pix_spatial_scale'])
        # params already provide per-galaxy NumPy arrays for these
        g.all_Re = params['all_Re']
        g.all_hz = params['all_hz']
        g.all_Se = params['all_Se']
        # Support legacy cases where 'all_n' might be missing: fall back
        # to central_n replicated for all galaxies
        g.all_n = params.get('all_n', np.full(n_g, params.get('central_n', 1.0)))
        g.all_gal_x_angles = params['all_gal_x_angles']
        g.all_gal_y_angles = params['all_gal_y_angles']
        g.all_gal_vz_sigmas = np.full(n_g, params['sigma_v'])

        self.generator = g


    def _run_generate(self):
        # Auto-show log window
        if hasattr(self, 'log_window') and self.log_window.winfo_exists():
            self.log_window.deiconify()
            self.log_window.lift()
        else:
            self.log_window = LogWindow(self)

        try:
            results = self.generator.generate_cubes()
            # Enable buttons on main thread
            self.after(0, lambda: [
                self.mom0_btn.config(state='normal'),
                self.mom1_btn.config(state='normal'),
                self.spectra_btn.config(state='normal'),
                self.new_instance_btn.config(state='normal'),
                self.save_btn.config(state='normal')
            ])
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror('Error during generation', str(e)))
    
    
    def generate(self):
        if self.generator is None:
            self.create_generator()
            if self.generator is None: return
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

        try:
            results = self.generator.generate_cubes()
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror('Error during generation', str(e)))
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
        self.after(0, lambda: self._save_sim_prompt(cube, params, meta))

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
        fname = filedialog.asksaveasfilename(defaultextension='.npz', filetypes=[('NumPy archive', '.npz'), ('Pickled Python object', '.pkl')])
        if not fname:
            return

        try:
            if fname.lower().endswith('.npz'):
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

        Removes any temporary PNG files recorded in ``_MATH_TEMPFILES`` and
        attempts to close the Tk window. If a normal destroy fails the
        process is force-exited to ensure orphaned processes do not remain.
        """

        for p in list(_MATH_TEMPFILES):
            try: os.remove(p)
            except: pass
        try: self.destroy()
        except: os._exit(0)


def main():
    app = GalCubeCraftGUI()
    app.mainloop()

if __name__ == '__main__':
    main()
