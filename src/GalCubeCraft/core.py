"""Core routines to build synthetic IFU spectral cubes.

This module implements a compact pipeline to create toy ``n_vel x ny x nx``
spectral cubes that mimic IFU observations of disk galaxies. The implementation
is intentionally self-contained and focuses on the following responsibilities:

- Build a 3D light distribution from a Sérsic radial profile combined with an
    exponential vertical profile (see :meth:`GalCubeCraft.sersic_flux_density_3d`).
- Create a simple analytical rotation curve and assign tangential velocities to
    the 3D grid (see :meth:`GalCubeCraft.milky_way_rot_curve_analytical`).
- Rotate the full 3D flux and velocity fields to simulate arbitrary viewing
    angles and project galaxy emission into velocity bins to form a spectral cube
    (see :meth:`GalCubeCraft.rotated_system` and
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
    
    def __init__(self, n_gals=None, n_cubes=1, resolution='all', offset_gals=5, beam_info = [4,4,0], grid_size=125, n_spectral_slices=40, fname=None, verbose=True, seed=None):

        # Initialize random seeds for reproducible results
        #self.central_Re_kpc = 5 #kpc

        # Store configuration parameters
        self.resolution = resolution
        self.fname = fname
        self.seed = seed
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
        
        if self.resolution != 'visualise':
            if self.resolution == 'all':
                # Mixed resolution: wide range from unresolved to well-resolved
                r_min = 0.25   # Heavily beam-dominated
                r_max = 4      # Well-resolved structure
            elif self.resolution == 'unresolved':
                # Unresolved scenario: galaxy smaller than beam
                r_min = 0.25
                r_max = 1
            elif self.resolution == 'resolved':
                # Resolved scenario: galaxy much larger than beam
                r_min = 1
                r_max = 4
                
            # Log-uniform sampling to ensure good coverage across orders of magnitude
            log_r_min = np.log10(r_min)
            log_r_max = np.log10(r_max)
            log_r = np.random.uniform(log_r_min, log_r_max, size=n_cubes)
            r = 10 ** log_r

        else:
            # Special visualization mode with fixed resolution values
            r=np.asarray([0.3,1.2,2,3,5])

        # Convert resolution ratio to effective radius in pixels
        # Re = r * (beam minor axis / 2) gives effective radius
        Re_central = r * self.beam_info[0] /2


        # =======================================================================
        # GALAXY PARAMETER GENERATION  
        # =======================================================================
        # Generate physical parameters for each galaxy system
        
        # Fixed effective radius in physical units (could be varied)
        central_Re_kpc = np.random.uniform(5, 5, n_cubes)  # Central Re in kpc

        # Generate parameters for each cube
        for i in range(n_cubes):

            # Calculate pixel scale: kpc per pixel
            pix_spatial_scale = central_Re_kpc[i] / Re_central[i]  # Scale in pixels relative to Re

            # Primary galaxy parameters
            Re = [Re_central[i]]                               # Effective radius in pixels
            hz = [np.random.uniform(0.5, 1) / pix_spatial_scale]  # Scale height (thinner in high-res)
            
            # Adjust surface brightness based on resolution
            # Unresolved galaxies get higher flux to compensate for smaller size
            # =========================================================
            # ADAPTIVE FLUX SCALING
            # =========================================================
            # Physics: Total Flux ~ Se * Re^2
            # To keep Total Flux constant across different resolutions (r),
            # we scale Se by (1/r^2).
            
            # 1. Define base flux density (intrinsic)
            base_Se = np.random.uniform(0.08, 0.12)
            
            # 2. Define a reference r value (e.g., r=1.0 is the baseline "standard" size)
            # If r < r_ref, the galaxy is smaller, so Se increases to conserve flux.
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
            self.all_n.append(np.random.uniform(0.5, 1.5, n_gal))               # Sérsic index: 0.5-1.5
            self.all_gal_v_0.append(np.random.uniform(200, 200, n_gal))         # Rotation velocity: fixed at 200 km/s

        # Initialize cube generation process
        self.fname = fname


    # ==========================================================================
    # STATIC METHODS FOR GALAXY PHYSICS
    # ==========================================================================

    @staticmethod
    def milky_way_rot_curve_analytical(R,v_0, R_e,n):
        """
        Calculate rotation velocity using an analytical galaxy rotation curve model.
        
        Based on empirical fits to observed galaxy rotation curves, this function
        computes the circular velocity at a given galactocentric radius.
        
        Parameters
        ----------
        R : float or array_like
            Galactocentric radius in kpc where rotation velocity is calculated.
        v_0 : float
            Characteristic rotation velocity in km/s (typically 200-300 km/s).
        R_e : float
            Effective radius in kpc (scale length of the galaxy).
        n : float
            Sérsic index affecting the shape of the rotation curve.
            
        Returns
        -------
        vel : float or array_like
            Circular rotation velocity in km/s at radius R.
            
        Notes
        -----
        - Uses analytical approximation: v(R) = v_0 * 1.022 * (R/R_0)^0.0803
        - R_0 is computed from effective radius and Sérsic index
        - The form approximates realistic galaxy rotation curves
        - Valid for disk-dominated galaxies at moderate radii
        
        References
        ----------
        Based on empirical relations from galaxy kinematic studies.
        See https://www.aanda.org/articles/aa/pdf/2017/05/aa30540-17.pdf
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
        """
        Compute 3D Sérsic flux density profile for a galaxy disk.
        
        Combines a Sérsic profile in the disk plane (x-y) with exponential
        fall-off in the vertical direction (z-axis). This represents the
        3D light distribution of a typical disk galaxy.
        
        Parameters
        ----------
        x, y, z : array_like
            3D spatial coordinate grids in physical units (kpc).
        Se : float
            Flux density at the effective radius in arbitrary units.
        Re : float  
            Effective radius in the same units as x, y coordinates.
        n : float
            Sérsic index controlling profile shape:
            - n = 1: Exponential disk (typical for disk galaxies)
            - n = 4: de Vaucouleurs profile (elliptical galaxies)
            - 0.5 < n < 1.5: Range used for this simulation
        hz : float
            Exponential scale height in z-direction (kpc).
            
        Returns
        -------
        S : array_like
            3D flux density distribution matching input coordinate shape.
            
        Notes
        -----
        - Uses circular symmetry (axis ratio q = 1) in disk plane
        - Sérsic parameter bn calculated using series expansion approximation
        - Vertical profile: S_z(z) = exp(-|z|/hz)
        - Radial profile: S_r(r) = Se * exp(-bn * ((r/Re)^(1/n) - 1))
        - Total profile: S(x,y,z) = S_r(r) * S_z(z)
        
        References
        ----------
        Sérsic profile: Sérsic, J. L. 1963, Boletín de la Asociación Argentina 
        de Astronomía, 6, 41
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
        """
        Generate a single galaxy with 3D structure, kinematics, and coordinate rotations.
        
        This method creates the core 3D galaxy model by:
        1. Computing 3D Sérsic flux density distribution
        2. Calculating rotation curve and velocity field
        3. Applying coordinate transformations for realistic viewing angles
        4. Generating diagnostic plots if requested
        
        Parameters
        ----------
        params_gal_rot : dict
            Galaxy parameters containing:
            - 'pix_spatial_scale': Physical scale in kpc/pixel
            - 'Re': Effective radius in pixels
            - 'hz': Vertical scale height in pixels
            - 'Se': Effective flux density
            - 'n': Sérsic index
            - 'gal_x_angle': Rotation angle about X-axis (inclination)
            - 'gal_y_angle': Rotation angle about Y-axis (position angle)
            - 'gal_vz_sigma': Velocity dispersion in km/s
            - 'v_0': Characteristic rotation velocity in km/s
            
        Returns
        -------
        rotated_disk_xy : ndarray, shape (grid_size, grid_size, grid_size)
            3D flux density distribution after rotations.
        rotated_vel_z_cube_xy : ndarray, shape (grid_size, grid_size, grid_size)
            3D line-of-sight velocity field after rotations.
            
        Notes
        -----
        - Creates initial galaxy on grid with size self.init_grid_size
        - Applies cosmological flux density dimming: S ∝ (1+z)^-3
        - Velocity field includes rotation curve + random velocity dispersion
        - Coordinate rotations simulate realistic viewing angles
        - Line-of-sight velocities include projection effects
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

        print(f'\n[ § Creating {self.n_cubes} highly resolved cubes of dimensions {self.n_spectral_slices/5-1} (spectral) x {self.grid_size} x {self.grid_size} (spatial) § ]\n')

        for i in range(self.n_cubes):

            if self.verbose:
                    print(f'\n\n\u00a7--------------------- Creating cube # {i + 1} ---------------------\u00a7', end='\r')

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
    
    def visualise(self, data, idx, save=False, fname_save=None):
        """Wrapper method calling the function in visualise.py"""
        visualise(data, idx, save, fname_save)
    

    def __len__(self):
        return self.n_cubes



