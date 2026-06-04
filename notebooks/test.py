import songs  # package name (ensure the package is importable in this environment)
import numpy as np

# Initialise the generator using the convenience wrapper in __init__.py.
# Parameters explained:
# - n_gals: number of galaxy components per cube (None -> random 1-3)
# - n_cubes: how many cubes to generate in this run
# - resolution: 'all'|'resolved'|'unresolved' (controls Re/beam ratio sampling)
# - final_grid_size: spatial size of output cube (pixels)

seed = np.random.randint(0,1000)
print('seed = ', seed, '\n')
g = songs.init(n_gals=3, n_cubes=1, resolution='resolved', grid_size = 96, offset_gals=80, n_spectral_slices=40, seed=seed, save=True) #optional fname=.... to specify location

# Generate the cubes. This runs the pipeline and returns a list of tuples (cube, metadata).
# Each `cube` has shape (n_velocity, ny, nx). `metadata` contains keys like 'average_vels', 'beam_info', 'pix_spatial_scale', 'n_gals' etc.
sim = g.generate_cubes()

# Example: inspect the first result (optional)
print('Number of cubes generated:', len(sim))
print('First cube shape:', sim[0][0].shape)


import songs.visualise as visualise

# Now you can assign the functions directly

visualise.slice_view(sim, idx=0)

'''visualise.moment0(sim, idx=0)

visualise.moment1(sim, idx=0)

visualise.spectrum(sim, idx=0)'''