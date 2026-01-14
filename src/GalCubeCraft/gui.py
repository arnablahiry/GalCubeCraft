"""GalCubeCraft GUI

Compact Tkinter-based GUI to interactively configure and run the
``GalCubeCraft`` generator. Provides a three-column layout of parameter
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

    python -m GalCubeCraft.gui

Or instantiate :class:`GalCubeCraftGUI` and call ``mainloop()``. The GUI
expects the package to be importable (it will try a fallback path insertion
when executed as a script).
"""

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
def param_frame(parent, padding=8, border_color="#797979", bg="#303030", width=None, height=80):
    """Create a framed parameter panel used throughout the GUI.

    This helper centralises a common visual pattern used across the
    application: a thin outer border (contrasting colour) with an inner
    content frame that holds parameter widgets. The outer frame is packed
    for convenience; callers receive both frames so they may add labels,
    sliders, or more complex layouts into the ``inner`` frame while the
    outer provides a consistent visual outline.

    Notes
    -----
    - When ``width`` or ``height`` are provided the inner frame will have
      its requested size set and ``pack_propagate(False)`` will be used to
      prevent the frame from resizing to its children. This is useful for
      creating compact, fixed-size parameter panels.
    - The helper packs the ``outer`` frame immediately; this simplifies
      call-sites but means callers should not re-pack the ``outer``.

    Parameters
    ----------
    parent : tk.Widget
        Parent widget to attach the frames to (typically a :class:`tk.Frame`).
    padding : int, optional
        Internal padding inside the inner frame (default: 8).
    border_color : str, optional
        Colour used for the outer border area (default: ``"#797979"``).
    bg : str, optional
        Background colour for the inner content frame (default: ``"#303030"``).
    width, height : int or None, optional
        When provided, these set fixed dimensions on the inner frame. Use
        ``None`` to allow the inner frame to size to its children.

    Returns
    -------
    tuple
        ``(outer, inner)`` where ``outer`` is the bordered container Frame
        (already packed) and ``inner`` is the content Frame where widgets
        should be placed.

    Examples
    --------
    >>> outer, inner = param_frame(parent, padding=10, width=300, height=80)
    >>> ttk.Label(inner, text='My parameter').pack(anchor='w')

    """

    outer = tk.Frame(parent, bg=border_color)
    outer.pack(padx=4, pady=4)  # <--- pack the outer here
    inner = tk.Frame(outer, bg=bg, padx=padding, pady=padding)
    if width or height:
        inner.config(width=width, height=height)
        inner.pack_propagate(False)
    inner.pack(fill='both', expand=True)
    return outer, inner




def latex_label(parent, latex, font_size=2):
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
                    color="white")

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

    label = tk.Label(parent, image=photo, borderwidth=0)
    # Store both the PhotoImage AND the PIL image to prevent premature GC
    label.image = photo
    label._pil_image = img
    _MATH_TEMPFILES.append(tmp.name)

    label.pack()

    return label


# Import core
try:
    from .core import GalCubeCraft_Phy
except Exception:
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from GalCubeCraft.core import GalCubeCraft_Phy

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


class GalCubeCraftGUI(tk.Tk):
    """Main GUI application for interactively configuring and running
    ``GalCubeCraft`` simulations.

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
        :mod:`GalCubeCraft.visualise` which create Matplotlib figures; these
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

            python -m GalCubeCraft.gui

    Or instantiate from Python::

            from GalCubeCraft.gui import GalCubeCraftGUI
            app = GalCubeCraftGUI()
            app.mainloop()

    """

    def __init__(self):
        super().__init__()
        self.title('GalCubeCraft GUI')
        self.WINDOW_WIDTH = 650
        self.WINDOW_HEIGHT = 810
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.resizable(False, False)
        # Create a hidden log window immediately
        self.log_window = LogWindow(self)
        self.log_window.withdraw()  # Hide it until "Logs" button clicked
        # Track if we're closing to prevent thread issues
        self._is_closing = False



        # Banner image: load assets/cubecraft.png (fallback to text label)
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

        
        # Scrollable canvas + container frame for a compact, scrollable UI
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

        # Build 3-column layout (parameter panels + controls)
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
        """Display an interactive spectral-slice viewer for the first cube.

        Uses the helper in ``visualise.slice_view`` which provides an
        interactive Matplotlib Slider to step through spectral channels.
        """
        if self.generator:
            try:
                # Temporarily switch to TkAgg for interactive display
                import matplotlib
                matplotlib.use('TkAgg')
                # Pass the main window as parent so the viewer is a child Toplevel
                # Do not force channel=0 here; allow the viewer to choose its
                # default (central slice) when channel is None.
                fig, ax = slice_view(self.generator.results, idx=0, channel=None, parent=self)
                matplotlib.use('Agg')
            except Exception as e:
                import matplotlib
                matplotlib.use('Agg')
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
        # Disable all except generate
        try:
            self.moments_btn.config(state='disabled')
        except Exception:
            # Fallback: older versions may have separate buttons
            try:
                self.mom0_btn.config(state='disabled')
            except Exception:
                pass
            try:
                self.mom1_btn.config(state='disabled')
            except Exception:
                pass
        self.spectra_btn.config(state='disabled')
        try:
            self.slice_btn.config(state='disabled')
        except Exception:
            pass
        # Also disable Save when starting a fresh instance
        try:
            self.save_btn.config(state='disabled')
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
        ``GalCubeCraft`` generator in sync for quick inspection.

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

        col_width = 310  # column width

        # ---------------------------
        # Row 1: Number of galaxies + Satellite offset
        # ---------------------------
        r1 = ttk.Frame(self.container)
        r1.pack(fill='x', pady=4)

        # Number of galaxies frame (radio buttons 1–6)
        outer1, fr1 = param_frame(r1, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')
        latex_label(fr1, r"\text{Number of galaxies}").pack(anchor='w', pady=(0,6))
        rb_frame = ttk.Frame(fr1)
        rb_frame.pack(anchor='w')
        for val in range(1, 7):
            rb = ttk.Radiobutton(rb_frame, text=str(val), variable=self.n_gals_var, value=val)
            rb.pack(side='left', padx=4)


        # Spatial resolution frame (kpc per pixel)
        outer2, fr2 = param_frame(r1, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\text{Spatial Resolution } (\Delta_{X,Y}) \: {\rm [kpc\;px^{-1}]}").pack(anchor='w')
        self.pix_scale_var_slider = self.make_slider(fr2, "", self.spatial_resolution, 0.72, 9.0, resolution=0.01, fmt="{:.2f}")
        self.pix_scale_var_slider.pack(fill='x')




        # ---------------------------
        # Row 2: FOV + Beam
        # ---------------------------
        r2 = ttk.Frame(self.container)
        r2.pack(fill='x', pady=4)

        # --- FOV frame ---
        outer1, fr1 = param_frame(r2, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')

        # LaTeX-style label for the section
        latex_label(fr1, r"\text{Field of View [kpc]}").pack(anchor='w', pady=(0,6))

        # --- Input row (pixel values) ---
        fov_row = ttk.Frame(fr1)
        fov_row.pack(anchor='w', pady=2)

        entry_width = 4  # width for entry boxes

        # Variables: bmin/bmaj (kpc), BPA (deg), spatial resolution already defined

        # Pixel inputs
        for text, var in [
            (r"FOV_{X} \:\:;\:\: FOV_{Y}\:", self.fov),
        ]:
            lbl = latex_label(fov_row, text)
            lbl.pack(side='left', padx=(0,2))
            e = ttk.Entry(fov_row, textvariable=var, width=entry_width)
            e.pack(side='left', padx=(0,6))


        # --- Beam frame ---
        outer2, fr2 = param_frame(r2, width=col_width)
        outer2.pack(side='left', padx=4, fill='y')

        # LaTeX-style label for the section
        latex_label(fr2, r"\text{Beam Information [kpc , kpc , deg]}").pack(anchor='w', pady=(0,6))

        # --- Input row (pixel values) ---
        beam_row = ttk.Frame(fr2)
        beam_row.pack(anchor='w', pady=2)

        entry_width = 3  # width for entry boxes

        # Variables: bmin/bmaj (kpc), BPA (deg)

        # Pixel inputs
        for text, var in [
            (r"B_{\rm min}", self.bmin_var),
            (r"B_{\rm maj}", self.bmaj_var),
            (r"\rm BPA", self.bpa_var)
        ]:
            lbl = latex_label(beam_row, text)
            lbl.pack(side='left', padx=(0,2))
            e = ttk.Entry(beam_row, textvariable=var, width=entry_width)
            e.pack(side='left', padx=(0,6))




        # ---------------------------
        # Row 3: Sérsic n + Scale height
        # ---------------------------
        r3 = ttk.Frame(self.container)
        r3.pack(fill='x', pady=4)

        outer1, fr1 = param_frame(r3, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')
        latex_label(fr1, r"\text{Sérsic index } (n) \: [-]").pack(anchor='w')
        self.n_slider = self.make_slider(fr1, "", self.n_var, 0.5, 1.5, resolution=0.01, fmt="{:.3f}")
        self.n_slider.pack(fill='x')

        outer2, fr2 = param_frame(r3, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\text{Scale height } (h_z) \ [\text{kpc}]").pack(anchor='w')
        self.hz_slider = self.make_slider(fr2, "", self.hz_var, 0.4, 9.0, resolution=0.01, fmt="{:.3f}")
        self.hz_slider.pack(fill='x')

        # ---------------------------
        # Row 4: Central effective flux density (S_e) + Satellite offset
        # ---------------------------
        r4 = ttk.Frame(self.container)
        r4.pack(fill='x', pady=4)

        outer1, fr1 = param_frame(r4, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')
        latex_label(fr1, r"\text{Central effective flux density } (S_e) \ [\text{Jy}]").pack(anchor='w')
        ttk.Entry(fr1, textvariable=self.Se_var).pack(fill='x')

        # Satellite offset frame (distance from primary centre in kpc)
        outer2, fr2 = param_frame(r4, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\text{Satellite offset from centre [kpc]}").pack(anchor='w', pady=(0,6))
        # Create slider and keep a reference to the underlying ttk.Scale
        self.sat_offset_var = tk.DoubleVar(value=5.0)
        self.sat_offset_slider_frame = self.make_slider(
            fr2, "", self.sat_offset_var, 5.0, 100.0, resolution=0.1, fmt="{:.1f}"
        )
        self.sat_offset_slider_frame.pack(fill='x')

        # Find the ttk.Scale inside the composed slider frame
        def find_scale(widget):
            if isinstance(widget, ttk.Scale):
                return widget
            for child in widget.winfo_children():
                result = find_scale(child)
                if result is not None:
                    return result
            return None

        self.sat_offset_scale = find_scale(self.sat_offset_slider_frame)

        # Disable satellite offset when only 1 galaxy is selected
        if self.n_gals_var.get() == 1:
            self.sat_offset_scale.state(['disabled'])

        # Auto-enable/disable satellite offset slider when n_gals changes
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
        # Row 5: Spectral resolution + velocity dispersion
        # ---------------------------
        r5 = ttk.Frame(self.container)
        r5.pack(fill='x', pady=4)

        
        outer1, fr1 = param_frame(r5, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')
        latex_label(fr1, r"\text{Spectral Resolution }(\Delta_{v_z})\ [km\;s^{-1}]").pack(anchor='w')
        self.spec_slider = self.make_slider(fr1, "", self.spectral_resolution, 5, 40, resolution=5, fmt="{:d}", integer=True)
        self.spec_slider.pack(fill='x')

        outer2, fr2 = param_frame(r5, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\text{Velocity dispersion }(\sigma_{v_z})\ [km\;s^{-1}]").pack(anchor='w')
        self.sigma_slider = self.make_slider(fr2, "", self.sigma_v_var, 30.0, 60.0, resolution=0.1, fmt="{:.1f}")
        self.sigma_slider.pack(fill='x')

        # ---------------------------
        # Row 6: Inclination angle (θ_X) + Azimuthal angle (ϕ_Y)
        # ---------------------------
        r6 = ttk.Frame(self.container)
        r6.pack(fill='x', pady=4)

        outer1, fr1 = param_frame(r6, width=col_width)
        outer1.pack(side='left', padx=6, fill='y')
        latex_label(fr1, r"\text{Inclination angle }(\theta_X) \text{ [deg]}").pack(anchor='w')
        self.angle_x_slider = self.make_slider(fr1, "", self.angle_x_var, 0, 359, resolution=1, fmt="{:d}", integer=True)
        self.angle_x_slider.pack(fill='x')

        outer2, fr2 = param_frame(r6, width=col_width)
        outer2.pack(side='left', padx=6, fill='y')
        latex_label(fr2, r"\text{Azimuthal angle }(\phi_Y) \text{ [deg]}").pack(anchor='w')
        self.angle_y_slider = self.make_slider(fr2, "", self.angle_y_var, 0, 359, resolution=1, fmt="{:d}", integer=True)
        self.angle_y_slider.pack(fill='x')

        # ---------------------------
        # Generate & utility buttons (Generate, Slice, Moments, Spectrum, Save, New)
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
        style.configure('Dark.TButton', background=btn_bg, foreground=btn_fg, height=btn_height, padding=(0,0))
        style.map('Dark.TButton',
                  background=[('active', btn_active_bg), ('disabled', btn_bg), ('!disabled', btn_bg)],
                  foreground=[('disabled', btn_disabled_fg), ('!disabled', btn_fg)])

        # Create as ttk.Button with the dark style (keeps rest of theme intact)
        self.generate_btn = ttk.Button(btn_frame, text='Generate', command=self.generate, style='Dark.TButton', width=5)
        self.slice_btn = ttk.Button(btn_frame, text='Slice', command=self.show_slice, state='disabled', style='Dark.TButton', width=5)
        # Combined Moments button: shows both Moment0 and Moment1 windows
        self.moments_btn = ttk.Button(btn_frame, text='Moments', command=self.show_moments, state='disabled', style='Dark.TButton', width=5)
        self.spectra_btn = ttk.Button(btn_frame, text='Spectrum', command=self.show_spectra, state='disabled', style='Dark.TButton', width=5)
        # The "New" button resets the GUI to a fresh instance. Make it
        # visible by default (enabled) so users can quickly clear state.
        # It will be disabled by reset_instance when appropriate.
        self.new_instance_btn = ttk.Button(btn_frame, text='Reset', command=self.reset_instance, state='disabled', style='Dark.TButton', width=5)

        # Pack buttons side by side with padding; Save before New (New last)
        self.save_btn = ttk.Button(btn_frame, text='Save', command=self.save_sim, state='disabled', style='Dark.TButton', width=5)
        for btn in [self.generate_btn, self.slice_btn, self.moments_btn, self.spectra_btn, self.save_btn, self.new_instance_btn]:
            btn.pack(side='left', padx=4, pady=2, expand=True, fill='x')


       

        # Auto-create/refresh generator when variables change (fast preview)
        def _auto_update_generator(*args):
            try:
                self.create_generator()
            except Exception as e:
                print("Auto-create generator failed:", e)

        for var in [self.bmin_var, self.bmaj_var, self.bpa_var, self.spatial_resolution, self.n_var,
                    self.hz_var, self.Se_var, self.sigma_v_var, self.fov,
                    self.spectral_resolution, self.angle_x_var, self.angle_y_var]:
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
        fov = int(self.fov.get())
        spectral_resolution = int(self.spectral_resolution.get())
        spatial_resolution = int(self.spatial_resolution.get())
        central_n = float(self.n_var.get())
        central_hz = float(self.hz_var.get())
        central_Se = float(self.Se_var.get())
        central_gal_x_angle = int(self.angle_x_var.get())
        central_gal_y_angle = int(self.angle_y_var.get())
        offset_gals = float(self.sat_offset_var.get())
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

            # Satellites are smaller and fainter than the primary
            sat_Re = list(rng.uniform(all_Re[0] / 3.0, all_Re[0] / 2.0, n_sat))
            sat_hz = list(rng.uniform(all_hz[0] / 3.0, all_hz[0] / 2.0, n_sat))
            sat_Se = list(rng.uniform(all_Se[0] / 3.0, all_Se[0] / 2.0, n_sat))

            # Random Sérsic indices for satellites
            sat_n = list(rng.uniform(0.5, 1.5, n_sat))

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
            g = GalCubeCraft_Phy(
                n_gals=params['n_gals'],
                n_cubes=1,
                spatial_resolution=params['spatial_resolution'],
                spectral_resolution=params['spectral_resolution'],                
                offset_gals=params['offset_gals'],
                beam_info=params['beam_info'],
                fov=params['fov'],
                verbose=True,
                seed=None
            )
        except Exception as e:
            messagebox.showerror('Error', f'Failed to create GalCubeCraft: {e}')
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
                self.after(0, lambda: [
                    self.moments_btn.config(state='normal'),
                    self.spectra_btn.config(state='normal'),
                    self.slice_btn.config(state='normal'),
                    self.save_btn.config(state='normal'),
                    self.new_instance_btn.config(state='normal'),
                ])
            except Exception as e:
                if not self._is_closing:
                    self.after(0, lambda e=e: messagebox.showerror('Error during generation', str(e)))
        finally:
            # Re-enable garbage collection if it was enabled
            if gc_was_enabled:
                gc.enable()
    
    
    def generate(self):
        # Always create a fresh generator from current UI values 
        # so that changes to n_gals or sliders are captured
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
    app = GalCubeCraftGUI()
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