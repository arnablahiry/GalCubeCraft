"""This module implements a compact pipeline to create toy ``n_vel x ny x nx``
spectral cubes that mimic IFU observations of disk galaxies. The implementation
is intentionally self-contained and focuses on the following responsibilities:

- Build a 3D light distribution from a Sérsic radial profile combined with an
    exponential vertical profile (see :meth:`GalCubeCraft.sersic_flux_density_3d`).
- Create a simple analytical rotation curve and assign tangential velocities to
    the 3D grid (see :meth:`GalCubeCraft.milky_way_rot_curve_analytical`).
- Rotate the full 3D flux and velocity fields to simulate arbitrary viewing
    angles and project galaxy emission into velocity bins to form a spectral
    cube (see :meth:`GalCubeCraft.rotated_system` and
    :meth:`GalCubeCraft.make_spectral_cube`).
- Optionally convolve the final cube with a telescope beam and save cubes to
    disk (see :meth:`GalCubeCraft.generate_cubes`).

Design notes
------------
- Coordinates: internal grids are defined in pixels and converted to physical
    units (kpc) using a pixel scale stored per cube.
- Velocities: rotation is computed analytically and random Gaussian scatter is
    added per voxel to mimic dispersion.
- Output: spectral cubes are produced in units of flux per pixel and are
    optionally downsampled/averaged in the spectral axis to simulate channel
    binning/oversampling.

Usage notes and API surface
---------------------------
- The primary user-facing class is :class:`GalCubeCraft`. Call
    ``g.generate_cubes()`` to produce one or more cubes; the method appends the
    generated outputs to ``g.results`` and also returns that list. Each entry in
    ``g.results`` is typically a tuple ``(spectral_cube, params)`` where
    ``spectral_cube`` is a NumPy array with shape ``(n_spectral, ny, nx)`` and
    ``params`` is a dict containing metadata (beam info, pixel scale, velocity
    axes, etc.). The GUI and utilities in this repository expect this layout.

- NOTE: visualisation helpers are provided as top-level functions in
    ``visualise.py`` (moment0, moment1, spectrum). There is intentionally no
    bound ``visualise`` attribute on the ``GalCubeCraft`` instance in this
    implementation; consumers should import and call the helper functions in
    ``GalCubeCraft.visualise`` or use the returned ``g.results`` with those
    helpers. A lightweight wrapper method could be added if callers prefer a
    bound method, but the current design keeps plotting utilities separate from
    the generation core to avoid UI/plotting dependencies in the core module.

This file provides the :class:`GalCubeCraft` helper class which encapsulates
parameters, sampling choices, and the generation pipeline.
"""

import numpy as np
import os
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
from .utils import *
from astropy.cosmology import FlatLambdaCDM
import matplotlib.patches as patches
from astropy import units as u
from scipy.ndimage import gaussian_filter1d

# Flat ΛCDM cosmology used to convert small spatial offsets into
# relative Hubble-flow velocities when placing multiple galaxies in one cube.
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from astrodendro import Dendrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .visualise import *


class GalCubeCraft:
    """Generator class that assembles synthetic IFU spectral cubes.

    High-level behaviour
    --------------------
    - For each requested cube the class samples (or accepts) one or more
      galaxy components. Each component has a 3D Sérsic + exponential
      vertical light distribution and a simple analytical rotation field.
    - The components are rotated to a viewing geometry and placed into a
      larger spatial grid. Voxels are binned by line-of-sight velocity to
      produce a 3D spectral cube (n_spectral x ny x nx).
    - The pipeline supports a resolution parameter that controls the
      effective physical size of galaxies relative to the beam; this is used
      to vary surface brightness and sampling across generated cubes.

    Constructor arguments closely mirror the fields used throughout the
    implementation (see the __init__ signature). Key internal attributes are
    lists storing per-cube parameters (``all_Re``, ``all_Se``,
    ``all_pix_spatial_scales``, etc.) and the final results are appended to
    ``self.results`` as tuples of ``(spectral_cube, params)``.

    Implementation details (what the methods do)
    ---------------------------------------------
    - :meth:`milky_way_rot_curve_analytical` computes an analytic circular
      velocity as a function of radius using a simple power-law approximation
      scaled by a characteristic velocity ``v_0``.
    - :meth:`sersic_flux_density_3d` returns a 3D flux field for a Sérsic
      profile in the disk plane multiplied by an exponential vertical profile.
    - :meth:`rotated_system` constructs the 3D flux and velocity cubes on a
      small grid, assigns tangential velocities, and applies geometric
      rotations (both image rotations and vector rotations) to yield a
      rotated flux cube and a rotated line-of-sight velocity cube.
    - :meth:`make_spectral_cube` places rotated components into the final
      grid, computes velocity bin masks for each spectral channel, projects
      emission along the line of sight, and returns the assembled spectral
      cube together with metadata (average velocities, beam info, pixel
      scale, etc.).

    The class intentionally focuses on clarity and inspectability rather
    than performance; nested Python loops are used to build fields which is
    adequate for moderate grid sizes used in examples.
    """
    
    def __init__(self, n_gals=None, n_cubes=1, resolution='all', offset_gals=5, beam_info = [4,4,0], grid_size=125, n_spectral_slices=40, n_sersic=None, save=False, fname=None, verbose=True, seed=None):
        """
        Initialize the GalCubeCraft generator.

        Parameters
        ----------
        n_gals : int or None
            If an integer, the fixed number of galaxies per cube. If ``None``,
            a random number of galaxies (1--3) is sampled for each cube.
        n_cubes : int
            Number of cubes to generate when ``generate_cubes`` is called.
        resolution : {'all', 'resolved', 'unresolved', 'visualise'}
            Controls sampling of effective radii relative to the beam. 'all'
            samples a broad log-uniform range; 'resolved' and 'unresolved'
            constrain the ratio; 'visualise' uses a fixed set useful for
            producing illustrative figures.
        offset_gals : float
            Typical spatial offset (in pixels) used when placing secondary
            galaxies relative to the primary in a multi-galaxy cube.
        beam_info : sequence
            Telescope beam description [bmin_px, bmaj_px, bpa] in pixels and
            degrees (position angle). bmin_px is used to set effective Re
            scales when sampling resolution.
        grid_size : int
            Final output spatial dimension (square): ny = nx = grid_size.
        n_spectral_slices : int
            Number of spectral channels to produce (internally the code uses
            5x oversampling and bins back to simulate channel binning, so the
            stored value is expanded internally).
        n_sersic : float or None
            If provided, a fixed Sérsic index to use for all galaxies. If
            ``None`` (the default) a per-galaxy Sérsic index is sampled
            from a uniform range (roughly 0.5--1.5) to produce disk-like
            and intermediate profiles.
        save : bool
            If ``True``, generated cubes will be written to disk as part of
            the :meth:`generate_cubes` run. When ``False`` (default) cubes
            are kept in-memory and returned via ``self.results``.
        fname : str or None
            Optional path where generated cubes will be saved. If ``None``
            the default `data/raw_data/<shape>/` directory is used.
        verbose : bool
            Whether to print progress messages during generation.
        seed : int or None
            RNG seed used to make results reproducible across NumPy, PyTorch
            and Python `random`.

        Notes
        -----
        After instantiation the object contains arrays (e.g. ``all_Re``,
        ``all_Se``, ``all_pix_spatial_scales``) describing the sampled
        parameters for each cube. Call :meth:`generate_cubes` to run the
        full pipeline and fill ``self.results``.

        Example
        -------
        >>> g = GalCubeCraft(n_cubes=2, grid_size=125, n_spectral_slices=40, seed=42)
        >>> len(g)
        2
        """

        # Initialize random seeds for reproducible results
        #self.central_Re_kpc = 5 #kpc

        # Store configuration parameters
        self.resolution = resolution
        self.fname = fname
        self.seed = seed
        self.save = save
        if self.seed is not None:
            # Set all random number generators for consistency
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        # Galaxy separation parameter (affects interaction dynamics)
        self.offset_gals = offset_gals

        # Determine number of galaxies per cube
        if not n_gals:
            # Randomly sample 1-3 galaxies per cube for variety
            self.n_gals = np.random.randint(1, 3, n_cubes)
        else:
            # Fixed number of galaxies across all cubes
            self.n_gals = [n_gals for _ in range(n_cubes)]

        # Grid and observational parameters
        self.n_cubes = n_cubes

        init_grid_size = (31/64)*grid_size
        if int(init_grid_size)%2!=0:
            self.init_grid_size = int(init_grid_size)-1
        else:
            self.init_grid_size = int(init_grid_size)

        self.grid_size = grid_size      # Size for combined output cube
        self.n_spectral_slices = 5*n_spectral_slices + 1  # 5x oversampling + 1 for binning
        self.beam_info = beam_info               # Telescope beam info : [bmin_px, bmaj_px, bpa]
        self.verbose = verbose

        # Initialize storage arrays for galaxy and system parameters
        self.results = []                 # Final spectral cube and params tuple
        self.all_gal_vz_sigmas = []       # Velocity dispersion along line of sight
        self.all_gal_x_angles = []        # Rotation angles about X-axis (inclination)
        self.all_gal_y_angles = []        # Rotation angles about Y-axis (position angle)
        self.all_Re = []                  # Effective radii in pixels
        self.all_hz = []                  # Vertical scale heights in pixels
        self.all_Se = []                  # Effective flux density
        self.all_n = []                   # Sérsic indices
        self.all_pix_spatial_scales = []  # Physical scale per pixel in kpc
        self.all_gal_v_0 = []            # Characteristic rotation velocity


        # =======================================================================
        # RESOLUTION PARAMETER SETUP
        # =======================================================================
        # Define the ratio r = Re/beam_radius to control spatial resolution
        # r < 1: Unresolved (galaxy smaller than beam)
        # r > 1: Resolved (galaxy larger than beam)
        

        if isinstance(self.resolution, float) and n_cubes==1:
            # Directly use the float as r
            r = np.full(n_cubes, self.resolution)
        else:
            if self.resolution == 'all':
                r_min, r_max = 0.25, 4
            elif self.resolution == 'unresolved':
                r_min, r_max = 0.25, 1
            elif self.resolution == 'resolved':
                r_min, r_max = 1, 4

            log_r = np.random.uniform(np.log10(r_min), np.log10(r_max), size=n_cubes)
            r = 10 ** log_r


        # Convert resolution ratio to effective radius in pixels
        # Re = r * (beam minor axis / 2) gives effective radius
        Re_central = r * self.beam_info[0] /2


        # =======================================================================
        # GALAXY PARAMETER GENERATION  
        # =======================================================================
        # Generate physical parameters for each galaxy system
        
        # Fixed effective radius in physical units (could be varied)
        central_Re_kpc = np.random.uniform(4, 6, n_cubes)  # Central Re in kpc

        # Generate parameters for each cube
        for i in range(n_cubes):

            # Calculate pixel scale: kpc per pixel
            pix_spatial_scale = central_Re_kpc[i] / Re_central[i]  # Scale in pixels relative to Re

            # Primary galaxy parameters
            Re = [Re_central[i]]                                      # Effective radius in pixels
            hz = [np.random.uniform(0.5, 1) / pix_spatial_scale]      # Scale height (thinner in high-res)
            n_sersics = [n_sersic if n_sersic is not None else np.random.uniform(0.5, 1.5)]  # If specific central sersic index is provided, else random
            
            
            # 1. Define base flux density (intrinsic)
            base_Se = np.random.uniform(0.08, 0.12)
            
            # 2. Define a reference r value (e.g., r=1.0 is the baseline "standard" size)
            # If r < r_ref, the galaxy is smaller, so Se increases to conserve loss of flux due to aggressive binning.
            r_ref = 1.0 
            
            # 3. Calculate scaling factor
            # We constrain the scaling to prevent singular values if r is tiny
            flux_scaling = (r_ref / r[i])**(1.5)
            
            # 4. Apply scaling
            Se = [base_Se * flux_scaling]


            # Orientation angles (disk inclination and position angle)
            gal_x_angles = [np.random.uniform(0, 85)]   # Inclination: 0°=face-on, 90°=edge-on
            gal_y_angles = [np.random.uniform(0, 85)]   # Position angle in sky plane
            
            n_gal = self.n_gals[i]

            # Generate satellite galaxies if multi-galaxy system
            if n_gal > 1:
                # Satellites are smaller and fainter than the primary
                Re += list(np.random.uniform(Re[0]/3, Re[0]/2, n_gal - 1))
                hz += list(np.random.uniform(hz[0]/3, hz[0]/2, n_gal - 1))
                Se += list(np.random.uniform(Se[0]/3, Se[0]/2, n_gal - 1))

                # Satellites have random sersic index always
                n_sersics += list(np.random.uniform(0.5, 1.5, n_gal-1))

                # Random orientations for satellites
                gal_x_angles += list(np.random.uniform(-180, 180, n_gal - 1))
                gal_y_angles += list(np.random.uniform(-180, 180, n_gal - 1))

            # Store parameters for this cube's galaxies
            self.all_pix_spatial_scales.append(np.full(n_gal, pix_spatial_scale))
            self.all_gal_vz_sigmas.append(np.random.uniform(30, 50, n_gal))      # Velocity dispersion 30-50 km/s
            self.all_gal_x_angles.append(np.asarray(gal_x_angles))
            self.all_gal_y_angles.append(np.asarray(gal_y_angles))
            self.all_Re.append(np.asarray(Re))
            self.all_hz.append(np.asarray(hz))
            self.all_Se.append(np.asarray(Se))
            self.all_n.append(np.asarray(n_sersics))                                    # Sérsic index: 0.5-1.5
            self.all_gal_v_0.append(np.random.uniform(200, 200, n_gal))         # Rotation velocity: fixed at 200 km/s

        # Initialize cube generation process
        self.fname = fname


    # ==========================================================================
    # STATIC METHODS FOR GALAXY PHYSICS
    # ==========================================================================

    @staticmethod
    def milky_way_rot_curve_analytical(R,v_0, R_e,n):
        """Analytical rotation-curve approximation.

        Computes an analytic circular velocity approximation used to assign
        tangential velocities to voxels in the toy galaxy model. The form is a
        shallow power-law scaled by a characteristic velocity v_0 and a
        scale radius R_0 derived from the Sérsic effective radius.

        Parameters
        ----------
        R : float
            Galactocentric radius (kpc). Can be a scalar or NumPy array.
        v_0 : float
            Characteristic rotation velocity (km/s). Typical values ~200 km/s.
        R_e : float
            Sérsic effective radius (kpc).
        n : float
            Sérsic index (dimensionless) used to derive the profile shape.

        Returns
        -------
        vel : float or ndarray
            Circular rotation velocity (km/s) evaluated at R.

        Notes
        -----
        - The function first computes the Sérsic constant b_n using a series
          expansion and then derives a scale radius R_0 ~ 2 * R_e / b_n^n.
        - The working formula is: v(R) = v_0 * 1.022 * (R / R_0)**0.0803
        - The exponent 0.0803 produces a gently rising/flat curve typical of
          disk galaxies over the radial range used here.

        Edge cases
        ----------
        - For R == 0 the returned velocity will be 0 (handled naturally by the
          power-law when R is 0).
        - Very small R_e or extreme n values may produce R_0 values that are
          physically unrealistic; validate inputs when using this function.

        Example
        -------
        >>> GalCubeCraft.milky_way_rot_curve_analytical(np.array([0.1,1,10]), 200, 5.0, 1.0)
        array([...])  # velocities in km/s

        References
        ----------
        See discussion in Lahiry et al. and empirical approximations used for
        compact rotation-curve modelling.
        """
        # Sérsic parameter calculation using series expansion
        bn_func = lambda k: 2 * k - 1/3 + 4 / (405 * k) + 46 / (25515 * k**2) + 131 / (1148175 * k**3) - 2194697 / (30690717750 * k**4)
        bn = bn_func(n)

        # Scale radius derived from effective radius and Sérsic index
        R_0 = 2*(R_e/((bn)**n))

        # Analytical rotation curve with empirically-motivated parameters
        vel = v_0 * 1.022 * np.power((R/R_0),0.0803)
        return vel
        
        #ref: https://www.aanda.org/articles/aa/pdf/2017/05/aa30540-17.pdf



    @staticmethod
    def sersic_flux_density_3d(x, y, z, Se, Re, n, hz):
        """Compute the 3D Sérsic + exponential vertical flux density.

        The returned array represents the intrinsic 3D flux distribution of
        a disk galaxy in physical units (same units as the coordinate grids).

        Parameters
        ----------
        x, y, z : ndarray
            Coordinate grids (kpc). These should have identical shapes (for
            example produced by ``np.meshgrid`` with ``indexing='ij'``).
        Se : float
            Flux density at the effective radius (arbitrary flux units).
        Re : float
            Effective (half-light) radius in kpc.
        n : float
            Sérsic index; lower values produce disk-like profiles.
        hz : float
            Vertical exponential scale height in kpc.

        Returns
        -------
        S : ndarray
            3D array with the same shape as the input coordinate grids giving
            flux density at each voxel.

        Notes
        -----
        - The radial Sérsic profile is evaluated using the standard series
          expansion for the constant b_n.
        - The profile assumes circular symmetry in the disk plane (axis ratio
          q = 1). To model elliptical disks, scale one of the axes before
          calling this routine.
        - The vertical structure is a symmetric exponential: exp(-|z|/hz).

        Example
        -------
        >>> nx = ny = nz = 21
        >>> x = np.arange(nx) - (nx-1)/2
        >>> X,Y,Z = np.meshgrid(x,x,x,indexing='ij')
        >>> S = GalCubeCraft.sersic_flux_density_3d(X, Y, Z, Se=0.1, Re=5.0, n=1.0, hz=0.5)
        >>> S.shape
        (21, 21, 21)

        References
        ----------
        Sérsic (1963) and standard approximations for b_n (see Ciotti &
        Bertin, 1999 for derivations and series expansions).
        """
        # Assume circular disk (could be generalized to elliptical)
        q = 1 

        # Calculate Sérsic parameter bn using series expansion
        bn_func = lambda k: 2 * k - 1/3 + 4 / (405 * k) + 46 / (25515 * k**2) + 131 / (1148175 * k**3) - 2194697 / (30690717750 * k**4)
        bn = bn_func(n)
        
        # Compute elliptical radius in disk plane
        r_elliptical = np.sqrt(x**2 + (y / q)**2)
        
        # Sérsic profile in the disk plane
        profile_xy = np.exp(-bn * ((r_elliptical / Re)**(1/n) - 1))
        
        # Exponential profile in vertical direction
        profile_z = np.exp(-np.abs(z) / hz)
        
        # Combined 3D profile
        S = Se * profile_xy * profile_z

        return S




    def rotated_system(self, params_gal_rot):
        """Create a rotated 3D galaxy flux cube and corresponding LOS velocity cube.

        This routine constructs an isolated galaxy on a small cubic grid of
        size ``self.init_grid_size`` and performs the following steps:

        1. Build a 3D flux density using :meth:`sersic_flux_density_3d`.
        2. Compute a tangential velocity magnitude at each (x,y) using
           :meth:`milky_way_rot_curve_analytical` and assign vector components
           in the local tangent direction.
        3. Add a Gaussian random LOS velocity component with standard
           deviation ``gal_vz_sigma`` to mimic dispersion.
        4. Rotate both the scalar flux cube and the velocity vector field to
           the requested viewing angles (inclination and position angle).

        Parameters
        ----------
        params_gal_rot : dict
            Dictionary with the following keys (units in parentheses):
            - 'pix_spatial_scale' (kpc/pixel)
            - 'Re' (pixels)
            - 'hz' (pixels)
            - 'Se' (flux units)
            - 'n' (dimensionless, Sérsic index)
            - 'gal_x_angle' (degrees, inclination)
            - 'gal_y_angle' (degrees, position angle)
            - 'gal_vz_sigma' (km/s, LOS dispersion)
            - 'v_0' (km/s, characteristic rotation velocity)

        Returns
        -------
        rotated_disk_xy : ndarray
            3D flux cube after rotations (shape ``(init_grid_size,)*3``).
        rotated_vel_z_cube_xy : ndarray
            3D line-of-sight velocity cube (same shape) giving the LOS
            velocity (km/s) at each voxel after rotation and projection.

        Performance and memory
        ----------------------
        - The method uses explicit Python loops to assign velocities and to
          rotate vectors voxel-by-voxel; this is clear but not optimal for
          very large grids. ``self.init_grid_size`` is chosen to be small to
          keep runtime reasonable for examples.

        Example
        -------
        >>> params = {'pix_spatial_scale':0.1, 'Re':20, 'hz':2, 'Se':0.1, 'n':1.0,
        ...           'gal_x_angle':45, 'gal_y_angle':30, 'gal_vz_sigma':40, 'v_0':200}
        >>> disk, vel = g.rotated_system(params)
        >>> disk.shape, vel.shape
        ((31,31,31), (31,31,31))

        Notes
        -----
        - The method currently assumes circular disks (axis ratio q=1) and a
          simple form for the rotation curve. Replace parts of the pipeline
          if you need more physical realism.
        """

        # Extract galaxy parameters from input dictionary
        pix_spatial_scale = params_gal_rot['pix_spatial_scale']
        Re_kpc = params_gal_rot['Re']*pix_spatial_scale      # Convert to physical units
        hz_kpc = params_gal_rot['hz']*pix_spatial_scale      # Convert to physical units
        Se = params_gal_rot['Se']
        n = params_gal_rot['n']
        angle_x = params_gal_rot['gal_x_angle']              # Inclination angle
        angle_y = params_gal_rot['gal_y_angle']              # Position angle
        sigma_vz = params_gal_rot['gal_vz_sigma']            # Velocity dispersion
        v_0 = params_gal_rot['v_0']                          # Rotation velocity scale


        #--------------------------------------------------------------------------------------------------------------------------#
        #                                          § GENERATING THE 3D SPATIAL CUBE §                                              # 
        #--------------------------------------------------------------------------------------------------------------------------#

        grid_size = self.init_grid_size
        centre = np.array([(grid_size - 1) / 2] * 3)    # Center of the 3D grid

        # Create 3D coordinate grid centered at origin
        if self.verbose:
            print('Calculating the flux density values at each spatial location')
        x = np.arange(grid_size) - (grid_size - 1) / 2  # Pixel coordinates
        y = np.arange(grid_size) - (grid_size - 1) / 2
        z = np.arange(grid_size) - (grid_size - 1) / 2
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Convert pixel coordinates to physical coordinates (kpc)
        X_kpc = X * pix_spatial_scale
        Y_kpc = Y * pix_spatial_scale
        Z_kpc = Z* pix_spatial_scale

        # Compute 3D galaxy flux density profile
        # Apply cosmological flux density dimming: S ∝ (1+z)^-3
        disk = self.sersic_flux_density_3d(X_kpc, Y_kpc, Z_kpc, Se, Re_kpc, n, hz_kpc)

        #--------------------------------------------------------------------------------------------------------------------------#
        #                                  § Calculating the velocity magnitudes and vectors §                                     # 
        #--------------------------------------------------------------------------------------------------------------------------#


        if self.verbose:
            print('Calculating and assigning velocity vectors...')
        vel_x_cube = np.zeros((grid_size, grid_size, grid_size))
        vel_y_cube = np.zeros((grid_size, grid_size, grid_size))
        vel_z_cube = np.zeros((grid_size, grid_size, grid_size))

        vel_mag_cube = np.zeros((grid_size, grid_size, grid_size))


        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):

                    coords = np.asarray([i,j,k])

                    pos_vect = coords[:2] - centre[:2]

                    tangent_vect = np.cross(pos_vect, [0,0,1])

                    r = np.linalg.norm(pos_vect)*pix_spatial_scale

                    velocity_mag_value = self.milky_way_rot_curve_analytical(r,v_0, Re_kpc, n)

                    if r !=0:   
                        tangent_unit_vect = tangent_vect/np.linalg.norm(tangent_vect)
                    else:
                        tangent_unit_vect = np.array([0,0,0])

                    vel_x_cube[i,j,k], vel_y_cube[i,j,k], vel_z_cube[i,j,k] = (velocity_mag_value * tangent_unit_vect[0]), (velocity_mag_value * tangent_unit_vect[1]), np.random.normal(0, sigma_vz)

                    vel_mag_cube[i,j,k] = velocity_mag_value



        #--------------------------------------------------------------------------------------------------------------------------#
        #                                                       § Rotations §                                                      # 
        #--------------------------------------------------------------------------------------------------------------------------#



        axes = [(0,2), (1,2)]


        rotation_angles = np.asarray([angle_x, angle_y, 0])



        #------------------------------------------- § Rotating/transforming the system § ---------------------------------------- # 

        if self.verbose:
            print('Rotating {:.2f} degrees about X axis and {:.2f} degrees about Y axis:'.format(rotation_angles[0], rotation_angles[1]))
            print('1. Rotating/transforming the whole system...')

        rotated_disk_x = rotate(disk, rotation_angles[0], axes=axes[0], reshape=False,)
        rotated_disk_xy = rotate(rotated_disk_x, rotation_angles[1], axes=axes[1], reshape=False)

        transformed_vel_z_cube_x = rotate(vel_z_cube, rotation_angles[0], axes=axes[0], reshape=False)
        transformed_vel_z_cube_xy = rotate(transformed_vel_z_cube_x, rotation_angles[1], axes=axes[1], reshape=False)

        transformed_vel_y_cube_x = rotate(vel_y_cube, rotation_angles[0], axes=axes[0], reshape=False)
        transformed_vel_y_cube_xy = rotate(transformed_vel_y_cube_x, rotation_angles[1], axes=axes[1], reshape=False)

        transformed_vel_x_cube_x = rotate(vel_x_cube, rotation_angles[0], axes=axes[0], reshape=False)
        transformed_vel_x_cube_xy = rotate(transformed_vel_x_cube_x, rotation_angles[1], axes=axes[1], reshape=False)

        
        #------------------------------------------ § Rotating the velocity vectors § ---------------------------------------- # 

        if self.verbose:
            print('2: Rotating the individual velocity vectors...')

        rotated_vel_z_cube_xy = np.zeros((grid_size,grid_size,grid_size))

        rotation = R.from_euler('yxz', rotation_angles, degrees=True)
        rotation_matrix = rotation.as_matrix()

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):

                    vel_vector = np.asarray([transformed_vel_x_cube_xy[i,j,k], transformed_vel_y_cube_xy[i,j,k], transformed_vel_z_cube_xy[i,j,k]])
                    rotated_vel_vector_xy = rotation_matrix @ vel_vector
                    rotated_vel_z_cube_xy[i,j,k] = rotated_vel_vector_xy[2]

        return rotated_disk_xy, rotated_vel_z_cube_xy


    def make_spectral_cube(self, rotated_disks, rotated_vel_z_cubes, pix_spatial_scale):
        """Assemble rotated component cubes into a final spectral cube.

        Projects multiple rotated galaxy components into a larger spatial grid,
        bins voxels by line-of-sight velocity into spectral channels, and
        returns a spectral cube together with metadata describing the
        configuration.

        Parameters
        ----------
        rotated_disks : list of ndarray
            List of 3D flux cubes (from :meth:`rotated_system`) for each
            component. Each array shape must match ``(init_grid_size,)*3``.
        rotated_vel_z_cubes : list of ndarray
            Corresponding list of 3D LOS velocity fields (km/s) for each
            component.
        pix_spatial_scale : float
            Physical scale (kpc/pixel) used for this cube; required for
            computing relative Hubble-flow offsets when placing multiple
            galaxies along the LOS.

        Returns
        -------
        spectral_cube_Jy_px : ndarray
            Spectral cube with shape ``(n_channels, grid_size, grid_size)``
            where the spectral axis corresponds to velocity-binned slices.
        params_gen : dict
            Metadata dictionary with keys: 'galaxy_centers', 'average_vels',
            'beam_info', 'n_gals', and 'pix_spatial_scale'.

        Notes
        -----
        - The method internally defines a symmetric velocity range (default
          -600 to +600 km/s) and creates ``self.n_spectral_slices`` fine
          bins. The code then averages groups of 5 fine bins to mimic
          spectral binning (i.e., 5x oversampling).
        - Components are placed at randomized centers near the cube centre and
          optionally offset along the LOS; small offsets are converted to a
          Hubble-flow velocity and added to that component's velocity cube.

        Example
        -------
        >>> cube, meta = g.make_spectral_cube([disk1, disk2], [vel1, vel2], 0.1)
        >>> cube.shape
        (40, 125, 125)
        """

        init_grid_size = self.init_grid_size
        grid_size = self.grid_size
        n_spectral_slices = self.n_spectral_slices
        n_galaxies = len(rotated_disks)
        assert n_galaxies == len(rotated_vel_z_cubes), "Mismatch between disks and velocity cubes"

        center_final_cube = np.array([(grid_size + 1) / 2] * 3)
        offset_range_1 = 0
        offset_range_2 = self.offset_gals #/pix_spatial_scale

        galaxy_centers = []

        half_size = init_grid_size // 2
        min_pos = half_size
        max_pos = grid_size - half_size

        # First galaxy near the center
        x = int(np.clip(center_final_cube[0] + np.random.randint(-offset_range_1, offset_range_1 + 1), min_pos, max_pos - 1))
        y = int(np.clip(center_final_cube[1] + np.random.randint(-offset_range_1, offset_range_1 + 1), min_pos, max_pos - 1))
        z = int(np.clip(center_final_cube[2] + np.random.randint(-offset_range_1, offset_range_1 + 1), min_pos, max_pos - 1))
        galaxy_centers.append(np.array([x, y, z]))

        # Additional galaxies nearby but offset

        for i in range(1, n_galaxies):
            x = int(np.clip(galaxy_centers[0][0] + np.random.randint(-offset_range_2, offset_range_2 + 1), min_pos, max_pos - 1))
            y = int(np.clip(galaxy_centers[0][1] + np.random.randint(-offset_range_2, offset_range_2 + 1), min_pos, max_pos - 1))
            z = int(np.clip(galaxy_centers[0][2] + np.random.randint(-offset_range_2, offset_range_2 + 1), min_pos, max_pos - 1))
            galaxy_centers.append(np.array([x, y, z]))

        if self.verbose:
            for idx, center in enumerate(galaxy_centers):
                print(f"Centre of galaxy {idx + 1}: {center}")

        # Apply Hubble flow relative to the first galaxy
        reference_z = galaxy_centers[0][2]
        H_z = cosmo.H(0).value  # km/s/Mpc

        for i in range(1, n_galaxies):
            delta_z_kpc = (galaxy_centers[i][2] - reference_z)*pix_spatial_scale
            delta_z_mpc = delta_z_kpc * 1e-3  # Convert kpc to Mpc
            relative_velocity = H_z * delta_z_mpc

            if self.verbose:
                direction = "farther" if delta_z_kpc > 0 else "closer"
                print(f"Galaxy {i+1} is {direction} than galaxy 1 by {delta_z_kpc:.2f} kpc")
                print(f"→ Adjusting velocity cube by {relative_velocity:.2f} km/s")

            # Add velocity offset to simulate redshift/blueshift
            rotated_vel_z_cubes[i] = (rotated_vel_z_cubes[i]+relative_velocity)
        



        # Creating lower and upper limits for the velocity observation bins
        # Create velocity bin edges across all galaxies

        min_vel = -600 #np.min([np.min(v) for v in all_velocities])
        max_vel = 600 #np.max([np.max(v) for v in all_velocities])

        limit = np.max([abs(min_vel), abs(max_vel)])  # Use the maximum absolute value for limits

        limits = np.linspace(-limit, limit, n_spectral_slices)

        if self.verbose:
            print('Overlaying all galaxy observations in a bigger spatial grid')
            print('Calculating the projected flux density of every voxel within the limits in each velocity slice')

        spectral_cube_S_px = []
        average_vels = np.zeros((n_spectral_slices - 1))


        for i in range(n_spectral_slices - 1):
            combined_cube = np.zeros((grid_size, grid_size, grid_size))
            for _, (disk, vel_cube, center) in enumerate(zip(rotated_disks, rotated_vel_z_cubes, galaxy_centers)):

                # Determine the voxels within current velocity bin
                if i < n_spectral_slices - 2:
                    condition = (vel_cube >= limits[i]) & (vel_cube < limits[i+1])
                else:
                    condition = (vel_cube >= limits[i]) & (vel_cube <= limits[i+1])  # include last edge
                selected_cube = np.zeros_like(disk)
                selected_cube[np.where(condition)] = disk[np.where(condition)]


                # Insert selected cube into the larger grid at the galaxy's center position
                xg, yg, zg = center
                half_size = init_grid_size // 2
                if init_grid_size % 2 == 0:
                    xs, xe = xg - half_size, xg + half_size
                    ys, ye = yg - half_size, yg + half_size
                    zs, ze = zg - half_size, zg + half_size
                else:
                    xs, xe = xg - half_size, xg + half_size + 1
                    ys, ye = yg - half_size, yg + half_size + 1
                    zs, ze = zg - half_size, zg + half_size + 1


                combined_cube[xs:xe, ys:ye, zs:ze] += selected_cube

           
            # Projecting along the LoS (Z-axis)
            spectral_slice = np.sum(combined_cube, axis=2)
            spectral_cube_S_px.append(spectral_slice)  # Transpose if needed

            # Store average velocity of this slice
            average_vel = np.mean([limits[i], limits[i+1]])
            average_vels[i] = average_vel

        spectral_cube_S_px = np.array(spectral_cube_S_px)


        spectral_cube_Jy_px = spectral_cube_S_px

        spectral_cube_Jy_px = spectral_cube_Jy_px.reshape(spectral_cube_Jy_px.shape[0]//5, 5, spectral_cube_Jy_px.shape[1], spectral_cube_Jy_px.shape[2]).mean(axis=1)  
        average_vels = average_vels.reshape(average_vels.shape[0]//5,5).mean(axis=1)

        # You can update the params_gals dictionary as needed
        params_gen = {
            'galaxy_centers': galaxy_centers,
            'average_vels': average_vels,
            'beam_info': self.beam_info,
            'n_gals': n_galaxies,
            'pix_spatial_scale': pix_spatial_scale,
        }

        return spectral_cube_Jy_px, params_gen



    def generate_cubes(self):
        """Run the full pipeline and generate the requested spectral cubes.

        This is the high-level convenience method that iterates over the
        pre-sampled per-cube parameters (``self.all_Re``, ``self.all_Se`` etc.),
        constructs each component via :meth:`rotated_system`, assembles the
        spectral cube with :meth:`make_spectral_cube`, applies beam
        convolution and light smoothing, saves the cube(s) to disk (unless
        ``self.fname`` is provided), and returns a list of ``(cube, params)``
        tuples stored in ``self.results``.

        Returns
        -------
        results : list
            A list with one entry per generated cube. Each entry is a tuple
            ``(spectral_cube_array, params_dict)`` where ``spectral_cube_array``
            has shape ``(n_channels, grid_size, grid_size)``.

        Example
        -------
        >>> g = GalCubeCraft(n_cubes=1, seed=42, verbose=False)
        >>> results = g.generate_cubes()
        >>> cube, meta = results[0]
        >>> cube.shape
        (40, 125, 125)

        Notes
        -----
        - The method performs several stochastic choices (positions, angles,
          flux scalings). Use ``seed`` in the constructor to reproduce
          results.
        - For large numbers of cubes or larger grids, consider refactoring the
          inner loops to use vectorized operations or offload heavy parts to
          compiled code for speed.
        """


        ASCII_BANNER = r"""
          _____       _    _____      _             _____            __ _   
         / ____|     | |  / ____|    | |           / ____|          / _| |  
        | |  __  __ _| | | |    _   _| |__   ___  | |     _ __ __ _| |_| |_ 
        | | |_ |/ _` | | | |   | | | | '_ \ / _ \ | |    | '__/ _` |  _| __|
        | |__| | (_| | | | |___| |_| | |_) |  __/ | |____| | | (_| | | | |_ 
         \_____|\__,_|_|  \_____\__,_|_.__/ \___|  \_____|_|  \__,_|_|  \__|
        """

        if self.verbose:
            print(ASCII_BANNER)


        for i in range(self.n_cubes):

            if self.verbose:
                    print(f'\n\n\u00a7------------ Creating cube # {i + 1} ------------\u00a7', end='\r')

            rotated_disks = []
            rotated_vel_z_cubes = []

            for j in range(self.n_gals[i]):

                params_gal_rot = {
                    'Re': self.all_Re[i][j],
                    'hz': self.all_hz[i][j],
                    'Se': self.all_Se[i][j],
                    'n': self.all_n[i][j],
                    'gal_x_angle': self.all_gal_x_angles[i][j],
                    'gal_y_angle': self.all_gal_y_angles[i][j],
                    'gal_vz_sigma': self.all_gal_vz_sigmas[i][j],
                    'gal_vz_sigma': self.all_gal_vz_sigmas[i][j],
                    'pix_spatial_scale': self.all_pix_spatial_scales[i][j],
                    'v_0': self.all_gal_v_0[i][j]
                }

                if self.verbose:
                    print(f'\nCreating disk #{j+1}...')


                rotated_disk, rotated_vel_z_cube = self.rotated_system(params_gal_rot)

                if self.verbose:
                    print(f'Disk #{j+1} generated!')

                rotated_disks.append(rotated_disk)
                rotated_vel_z_cubes.append(rotated_vel_z_cube)


            if self.verbose:
                print('\nCreating spectral cube...')
       
            spectral_cube_final, params = self.make_spectral_cube(rotated_disks, rotated_vel_z_cubes, self.all_pix_spatial_scales[i][0])


            #self.system_params.append(params)

            if self.verbose:
                print('\nSpectral cube created!')


            #Setting possible negative value artifacts to 0
            spectral_cube_final_resolved = np.maximum(spectral_cube_final, 0)

            spectral_cube_final_convolved = convolve_beam(spectral_cube_final, self.beam_info)
            spectral_cube_final_convolved = gaussian_filter1d(spectral_cube_final_convolved, sigma=0.6, axis=0)

            #self.spectral_cubes.append(spectral_cube_final)

            self.results.append((spectral_cube_final_convolved, params))

            if self.save:
                if self.fname is None:
                    # Use current working directory + /data/raw_data/ as default
                    base_dir = os.path.join(os.getcwd(), 'data', 'raw_data')
                    fname_save = os.path.join(base_dir, '{}x{}x{}'.format(self.n_spectral_slices-1, self.grid_size, self.grid_size))
                    if not os.path.exists(fname_save):
                        os.makedirs(fname_save)
                else:
                    fname_save = self.fname
                    if not os.path.exists(fname_save):
                        os.makedirs(fname_save)
                        
                np.save(fname_save+'/cube_{}.npy'.format(i+1),spectral_cube_final_convolved)

                if self.verbose:
                    print('saved as ' + fname_save + '/cube_{}.npy'.format(i+1))


        return self.results
    
    def __len__(self):
        return self.n_cubes



