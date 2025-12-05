# GalCubeCraft

High-fidelity toy generator for synthetic IFU (Integral Field Unit) spectral cubes.

GalCubeCraft provides a compact, well-documented pipeline to build 3D spectral cubes
that mimic observations of disk galaxies. It combines analytic galaxy models (Sérsic
light profiles + exponential vertical structure), simple rotation-curve kinematics,
viewing-angle projections and instrument effects (beam convolution, channel binning)
to produce realistic test data for algorithm development, denoising, and visualization.

This README explains the science and mathematics behind the generator, how to
install the package (assumes the package is published to PyPI), and several
practical examples for quick experimentation.

## Table of contents

- What GalCubeCraft does
- Scientific background & equations
- Installation (PyPI + source)
- Quick start examples
- API reference (minimal)
- Reproducibility, limitations, and troubleshooting
- Credits & citation

## What GalCubeCraft does

GalCubeCraft synthesizes spectral datacubes with dimensions (n_velocity, ny, nx).
Each cube contains one or more galaxy components. For each galaxy component the
generator:

- Builds a 3D flux density field using a Sérsic profile in the disk plane combined
	with an exponential vertical profile.
- Computes an analytic circular velocity field from a compact rotation-curve model
	and assigns tangential velocities to voxels.
- Rotates the 3D flux and velocity fields to a chosen viewing geometry.
- Projects emission into line-of-sight velocity bins to produce a spectral cube.
- Optionally convolves each 2D channel with a telescope beam and saves cubes to
	`data/raw_data/<nz>x<ny>x<nx>/cube_*.npy`.

The package is intentionally clear and inspectable (readable loops, compact
functions), making it suitable for method development and teaching.

## Scientific background & equations

This section summarises the main mathematical building blocks implemented in
the code: the Sérsic flux distribution, vertical exponential profile, and the
analytical rotation curve used to assign tangential velocities.

### Sérsic radial profile (disk plane)

The radial surface brightness (Sérsic) profile is given by
$$
S_r(r) = S_e \exp\left[-b_n\left(\left(\frac{r}{R_e}\right)^{1/n} - 1\right)\right]
$$

where
- $S_e$ is the flux density at the effective radius $R_e$,
- $n$ is the Sérsic index that controls the concentration,
- $b_n$ is a constant that depends on $n$ (approximated by a series expansion).

The package uses the standard series expansion for $b_n$:
$$
b_n(n) \approx 2n - \tfrac{1}{3} + \frac{4}{405n} + \frac{46}{25515n^2} + \cdots
$$

### Vertical exponential profile

Galaxies are modeled with an exponential vertical fall-off:
$$
S_z(z) = \exp\left(-\frac{|z|}{h_z}\right)
$$

Combining radial and vertical profiles gives the 3D flux density used in the
generator:
$$
S(x,y,z) = S_e \; \exp\left[-b_n\left(\left(\frac{r}{R_e}\right)^{1/n} - 1\right)\right]\; \exp\left(-\frac{|z|}{h_z}\right)
$$
with $r = \sqrt{x^2 + y^2}$ in the disk plane.

### Analytical rotation curve

To assign tangential velocities the implementation uses a compact empirical
approximation (implemented as `milky_way_rot_curve_analytical`):
$$
v(R) = v_0 \times 1.022 \times \left(\frac{R}{R_0}\right)^{0.0803}
$$
where $v_0$ is a characteristic velocity scale and $R_0$ is derived from the
effective radius and Sérsic index (see code comments for details). This simple
form reproduces the gently rising/flat behaviour of typical disk-galaxy rotation
curves at the scales of interest for IFU-like synthetic data.

### Beam convolution and FWHM to σ relation

When simulating instrument resolution we convolve 2D channels with an elliptical
Gaussian. The conversion between FWHM and Gaussian sigma used is:
$$
\sigma = \frac{\mathrm{FWHM}}{2\sqrt{2\ln 2}} \approx \frac{\mathrm{FWHM}}{2.355}
$$

This relation is used when creating a `Gaussian2DKernel` for convolution.

## Installation

Assuming you have published the package to PyPI, the simplest install is:

```zsh
pip install GalCubeCraft
```

Installing from source (developer mode):

```zsh
git clone https://github.com/arnablahiry/GalCubeCraft.git
cd GalCubeCraft
pip install -e .
```

Recommended dependencies are listed in `requirements.txt`. A minimal set used by
the package includes:

- numpy
- scipy
- matplotlib
- astropy
- torch

If you rely on plotting or dendrograms, also ensure `astrodendro` is available:

```zsh
pip install astrodendro
```

Note: for environments with GPU-accelerated PyTorch, install a matching `torch`
build according to your CUDA version (see https://pytorch.org).

## Quick start examples

Below are short, runnable examples that demonstrate common workflows. The
examples assume a Python session or script; replace package name with the one
you published to PyPI if different.

### 1) Generate one synthetic cube and inspect shapes

```python
from GalCubeCraft import GalCubeCraft

# Create a generator: one cube, grid 125x125, 40 spectral channels (internally oversampled)
g = GalCubeCraft(n_gals=None, n_cubes=1, resolution='all', grid_size=125, n_spectral_slices=40, seed=42)

# Run the generation pipeline and collect results
results = g.generate_cubes()

# Each element in results is a tuple (spectral_cube, metadata)
cube, meta = results[0]
print('cube shape (nz, ny, nx) =', cube.shape)
print('metadata keys =', list(meta.keys()))
```

Typical output:

- `cube.shape` → (n_velocity, ny, nx) (e.g. (40, 125, 125))
- `meta` contains `average_vels`, `beam_info`, `pix_spatial_scale`, etc.

### 2) Save and visualise

GalCubeCraft saves generated cubes to `data/raw_data/<nz>x<ny>x<nx>/cube_*.npy` by
default. The class also exposes a `visualise` helper that wraps the plotting
helpers in `visualise.py`:

```python
g.visualise(g.results, idx=0, save=False)
```

This will show moment-0 and moment-1 maps and a velocity spectrum using
matplotlib. Set `save=True` to write PDF figures in `figures/<shape>/`.

### 3) Minimal script to generate multiple cubes non-verbosely

```python
from GalCubeCraft import GalCubeCraft

g = GalCubeCraft(n_cubes=5, verbose=False, seed=123)
results = g.generate_cubes()

# Save metadata or run batch analysis over results
for i, (cube, meta) in enumerate(results, start=1):
		print(i, cube.shape, meta['n_gals'])
```

## Minimal API reference

- `GalCubeCraft(n_gals=None, n_cubes=1, resolution='all', offset_gals=5, beam_info=[4,4,0], grid_size=125, n_spectral_slices=40, fname=None, verbose=True, seed=None)`
	- Construct the generator. See code docstrings for parameter meanings.
- `generate_cubes()` → runs the pipeline and returns a list of tuples `(cube, params)`
- `visualise(data, idx, save=False, fname_save=None)` → wrapper for plotting utilities

Files of interest in the repository:

- `src/GalCubeCraft/core.py` — main pipeline and `GalCubeCraft` class
- `src/GalCubeCraft/utils.py` — beam, convolution and mask helpers
- `src/GalCubeCraft/visualise.py` — plotting helpers (moment maps, spectra)

## Reproducibility, limitations and edge cases

Edge cases and behaviour to be aware of:

- Small effective radii (much smaller than the beam) trigger flux-scaling to
	avoid vanishing integrated flux; check `all_Se` and `all_Re` if results look
	unusually bright or faint.
- Very small grids or extremely fine spectral oversampling may increase memory
	use; the code uses modest oversampling (5x) and then bins channels.
- The generator uses a compact analytic rotation curve (not a full mass-model).
	For physically realistic kinematics beyond toy data, replace the rotation
	module with your preferred prescription.

Suggested tests:

- Verify `generate_cubes()` returns arrays with non-negative flux (the code clips
	negative numerical artifacts to zero before convolution).
- Confirm beam convolution preserves integrated flux (up to numerical noise).

## Troubleshooting

- Import error after pip install: check that `PYTHONPATH` is not shadowing the
	installed package and that you're using the same Python interpreter where
	`pip` installed the package (use `python -m pip install ...` to be explicit).
- If plotting fails, ensure GUI backend is available or use a non-interactive
	backend (e.g., `matplotlib.use('Agg')`) when running headless.

## Credits & citation

This package was developed as a compact educational and research tool for IFU
data simulation and denoising algorithm development. If you use GalCubeCraft in
published work, please cite this repository and mention the design choices
(Sérsic + vertical exponential, simple analytic rotation curve).

License: please include your chosen license file in the repository (e.g., `LICENSE`).

---

If you'd like, I can also:

- Add a short example notebook in `examples/` demonstrating generation + plotting.
- Include unit tests for the key numerical functions (Sérsic profile, FWHM→σ).
- Add badges (PyPI, build, license) to the top of this README.

If you want any of those, tell me which and I'll implement them next.
