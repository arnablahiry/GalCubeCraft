#!/usr/bin/env python
"""
This module provides essential functions for processing and analyzing Integral Field Unit (IFU) spectral cubes, particularly for astronomical observations and synthetic data generation. The functions support common operations needed in radio and optical astronomy workflows.
"""

from astropy.convolution import Gaussian2DKernel, convolve_fft
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib.patches import Ellipse

def add_beam(ax, bmin_pix, bmaj_pix, bpa_deg, xy_offset=(10, 10), color='white', crosshair=True):
    """
    Add a synthesized beam ellipse visualization to a matplotlib plot.
    
    This function draws the instrumental beam pattern for radio interferometric observations,
    showing both the beam ellipse and optional crosshairs indicating the major/minor axes.
    Essential for displaying the angular resolution and orientation of radio telescope data.
    
    The beam represents the point spread function (PSF) of the interferometer, which determines
    the minimum resolvable angular scale and affects how sources appear in the image. The
    elliptical shape results from the distribution of baselines in the interferometer array.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the beam will be drawn.
        Should be the same axes containing the astronomical image.
    bmin_pix : float
        Minor axis of the beam ellipse in pixels.
        Represents the highest angular resolution direction of the interferometer.
        Typical values: 1-20 pixels depending on image resolution and beam size.
    bmaj_pix : float  
        Major axis of the beam ellipse in pixels.
        Represents the lower angular resolution direction of the interferometer.
        Always >= bmin_pix for proper ellipse definition.
    bpa_deg : float
        Beam position angle in degrees, measured counter-clockwise from the x-axis (East).
        Follows astronomical convention where 0° points East, 90° points North.
        Range: typically -90° to +90° but can be any angle.
    xy_offset : tuple of float, optional
        Offset from bottom-left corner of plot in pixels, by default (10, 10).
        Controls beam ellipse placement for optimal visibility.
        Adjust if beam overlaps with important image features.
    color : str, optional
        Color of the beam ellipse and crosshair lines, by default 'white'.
        Choose color for good contrast against the background image.
        Common choices: 'white', 'black', 'red', 'cyan'.
    crosshair : bool, optional
        Whether to draw crosshairs along major/minor axes, by default True.
        Crosshairs help visualize beam orientation and principal axes.
        Set False for cleaner appearance when beam orientation is not critical.
        
    Notes
    -----
    Beam Coordinate System:
    - Position angle follows astronomical convention (counter-clockwise from East)
    - Ellipse center positioned relative to plot limits, not data coordinates
    - Crosshairs align with beam principal axes for orientation reference
    
    Typical Usage in Radio Astronomy:
    - ALMA observations: circular to mildly elliptical beams
    
    The beam size should be derived from the observation metadata:
    - ALMA: BMAJ, BMIN, BPA header keywords
    - Synthetic data: Specified during simulation setup
    
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(radio_image)
    >>> # Add beam for ALMA observation
    >>> add_beam(ax, bmin_pix=2.5, bmaj_pix=3.2, bpa_deg=45.0, 
    ...          xy_offset=(15, 15), color='white')
    
    >>> # Simple circular beam without crosshairs
    >>> add_beam(ax, bmin_pix=4.0, bmaj_pix=4.0, bpa_deg=0.0,
    ...          crosshair=False, color='red')
    """

    # Get current plot limits to determine beam placement
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate beam center coordinates from bottom-left corner plus offset
    x0 = xlim[0] + xy_offset[0]
    y0 = ylim[0] + xy_offset[1]

    # Create the main beam ellipse using matplotlib Ellipse patch
    beam = Ellipse((x0, y0),
                   width=bmaj_pix,      # Major axis defines ellipse width
                   height=bmin_pix,     # Minor axis defines ellipse height  
                   angle=bpa_deg,       # Rotation angle in degrees
                   edgecolor=color,     # Outline color
                   facecolor='none',    # Transparent interior
                   linewidth=1)         # Line thickness
    ax.add_patch(beam)

    # Draw optional crosshairs to show beam orientation and principal axes
    if crosshair:
        # Convert beam position angle from degrees to radians for trigonometry
        # Negative sign accounts for matplotlib coordinate system conventions
        theta = np.deg2rad(-bpa_deg)

        # Calculate crosshair endpoints for minor axis (higher resolution direction)
        # Minor axis is perpendicular to major axis, rotated by 90 degrees
        dx_minor = 0.5 * bmin_pix * np.sin(theta)
        dy_minor = 0.5 * bmin_pix * np.cos(theta)

        # Calculate crosshair endpoints for major axis (lower resolution direction)  
        # Major axis aligns with beam position angle
        dx_major = 0.5 * bmaj_pix * np.cos(theta)
        dy_major = -0.5 * bmaj_pix * np.sin(theta)

        # Draw minor axis crosshair (high resolution direction)
        ax.plot([x0 - dx_minor, x0 + dx_minor],
                [y0 - dy_minor, y0 + dy_minor],
                color=color, linewidth=1.2, alpha=0.7)

        # Draw major axis crosshair (low resolution direction) 
        ax.plot([x0 - dx_major, x0 + dx_major],
                [y0 - dy_major, y0 + dy_major],
                color=color, linewidth=1.2, alpha=0.7)

        


def create_circular_aperture_mask(cube, R_e, beam_width_px, x_center=None, y_center=None):
    """
    Create a circular aperture mask for photometric analysis of spectral cubes.
    
    This function generates a 3D boolean mask for extracting flux from a circular region,
    with intelligent sizing based on source effective radius and beam size. The aperture
    automatically adapts to ensure optimal flux recovery while minimizing noise contamination.
    
    The algorithm chooses the larger of twice the effective radius (2×R_e) or the beam
    width to define the aperture diameter. This ensures that both extended sources and 
    beam-limited sources are properly captured.

    Parameters
    ----------
    cube : np.ndarray, shape (n_channels, nx, ny)
        Input 3D spectral cube with spectral axis first.
        Each cube[i,:,:] represents a 2D image at a specific wavelength/velocity.
    R_e : float
        Effective radius of the source in pixels.
        Typical definition: radius containing ~63% of total flux (1-sigma for Gaussian).
        For extended sources, should be measured from integrated emission maps.
    beam_width_px : float
        Beam width (diameter) in pixels, representing instrumental resolution.
        For circular beams: beam_width_px = beam_fwhm_px
        For elliptical beams: use geometric mean or major axis depending on analysis needs.
    x_center : float, optional
        X-coordinate of aperture center in pixels, by default None.
        If None, uses geometric center of the image (nx // 2).
        Should be determined from source detection or moment analysis.
    y_center : float, optional  
        Y-coordinate of aperture center in pixels, by default None.
        If None, uses geometric center of the image (ny // 2).
        Should match x_center coordinate system and source location.

    Returns
    -------
    mask_3d : np.ndarray, shape (n_channels, nx, ny), dtype=bool
        3D boolean mask where True indicates pixels inside the aperture.
        Same dimensions as input cube for direct masking operations.
        Use as: flux = np.sum(cube[mask_3d]) for aperture photometry.
        
    Notes
    -----
    Aperture Sizing Logic:
    - If source diameter (2×R_e) >= beam width: aperture = 2×(2×R_e) = 4×R_e
    - If beam width > source diameter: aperture = 2×beam_width
    - This ensures capture of both source extent and beam-convolved emission
    
    The factor of 2 expansion accounts for:
    - Beam convolution effects that spread flux beyond source boundaries
    - Safety margin for flux recovery in low SNR conditions  
    - Compensation for imperfect source centering
    
    Best Practices:
    - Measure R_e from high SNR integrated emission maps when possible
    - Use beam parameters from observation metadata (BMAJ, BMIN headers)
    - Center coordinates should come from source detection algorithms
    - Consider background subtraction before applying aperture masks
    
    Common Applications:
    - Aperture photometry for flux measurements
    - Signal-to-noise ratio calculations
    - Emission line analysis in specific regions
    - Performance evaluation of denoising algorithms
    
    Examples
    --------
    >>> # Create mask for compact source
    >>> mask = create_circular_aperture_mask(cube, R_e=2.5, beam_width_px=4.0)
    >>> aperture_flux = np.sum(cube[mask])
    
    >>> # Create mask centered on detected source
    >>> mask = create_circular_aperture_mask(cube, R_e=3.2, beam_width_px=5.1,
    ...                                     x_center=45.3, y_center=67.8)
    
    See Also
    --------
    astropy.photutils.aperture_photometry : More sophisticated aperture analysis
    photutils.CircularAperture : Alternative aperture implementation
    """

    # Calculate source diameter from effective radius
    D_e = 2*R_e

    # Store beam width for comparison (already represents diameter/width)
    beam_px = beam_width_px

    # Determine optimal aperture diameter using adaptive sizing logic
    if D_e >= beam_px:
        # Extended source case: source is larger than beam resolution
        D_aper = 2*D_e  # Aperture = 4×R_e to capture extended emission
        #print(f'D_e {D_e} is larger than beam size {beam_px}')
    elif beam_px > D_e:
        # Beam-limited source case: beam is larger than intrinsic source size  
        D_aper = 2*beam_px  # Aperture = 2×beam to capture beam-convolved emission
        #print(f'Beam_px {beam_px} is larger than D_e {D_e}')

    # Extract cube dimensions
    n_channels, nx, ny = cube.shape
    
    # Set aperture center coordinates (default to image center if not specified)
    if not (x_center and y_center):
        # Use geometric center of the image as default
        x_center, y_center = nx // 2, ny // 2
   
    # Create coordinate grids for distance calculation
    Y, X = np.ogrid[:nx, :ny]
    # Calculate Euclidean distance from each pixel to aperture center
    dist_from_center = np.sqrt((X - x_center)**2 + (Y - y_center)**2)

    # Legacy code for more complex aperture sizing (kept as documentation)
    '''# Base aperture radius
    aperture_radius = offset + k * R_e'''

    # Create 2D circular mask: True for pixels within aperture radius
    mask = dist_from_center <= D_aper/2

    # Replicate 2D mask across all spectral channels to create 3D mask
    # Each spectral channel gets identical spatial mask
    mask_3d = np.asarray([mask for i in range(n_channels)])
    return mask_3d


def convolve_beam(spectral_cube, beam_list):
    """
    Convolve a 3D spectral cube with an elliptical 2D Gaussian beam to simulate instrumental effects.
    
    This function applies beam convolution to each spectral channel independently, simulating
    the spatial point spread function (PSF) of radio interferometers or single-dish telescopes.
    The convolution accounts for the finite angular resolution of the instrument and properly
    scales flux values to maintain surface brightness units.
    
    The beam convolution process:
    1. Creates a normalized 2D Gaussian kernel representing the instrument beam
    2. Convolves each spectral slice with this kernel using FFT for efficiency  
    3. Scales the result by beam area to maintain proper flux density units
    
    This is essential for creating realistic synthetic observations that match the resolution
    and flux scale of actual telescope data.

    Parameters
    ----------
    spectral_cube : np.ndarray, shape (n_channels, nx, ny)
        Input 3D spectral cube with spectral axis first.
        Flux units should be surface brightness (e.g., Jy/beam, K, MJy/sr).
        Each spectral_cube[i,:,:] is convolved independently.
    beam_list : list of float, length 3
        Beam parameters [bmin_pixels, bmaj_pixels, theta_degrees] where:
        - bmin_pixels: Minor axis FWHM in pixels (highest resolution direction)
        - bmaj_pixels: Major axis FWHM in pixels (lowest resolution direction)
        - theta_degrees: Position angle in degrees measured counterclockwise 
          from positive X-axis (East direction in astronomical coordinates)

    Returns
    -------
    convolved_spectral_cube_beam : np.ndarray, shape (n_channels, nx, ny)
        Beam-convolved spectral cube with proper flux scaling.
        Output flux units preserved from input (e.g., Jy/beam → Jy/beam).
        Spatial resolution reduced to match specified beam size.
        
    Notes
    -----
    Beam Kernel Properties:
    - Uses 2D Gaussian with specified major/minor axes and position angle
    - Position angle measured counterclockwise from positive X-axis (East)
    - Kernel normalized to preserve total flux during convolution
    - Standard deviation calculated from FWHM using Gaussian relation: σ = FWHM / (2√(2ln2))
    
    Flux Scaling:
    - Convolution spreads flux over larger area, reducing peak values
    - Multiplication by beam area restores proper surface brightness scale
    - Beam area = π × bmaj × bmin / (4 × ln(2)) for elliptical Gaussian beam
    - Accounts for flux conservation during spatial redistribution
    
    Performance Considerations:
    - Uses FFT convolution for computational efficiency O(N log N)
    - Memory usage scales linearly with cube size
    - Processing time dominated by FFT operations per channel
    
    Common Applications:
    - Synthetic observation generation for algorithm testing
    - Data simulation matching specific telescope characteristics
    - Resolution degradation for cross-instrument comparisons
    - Noise correlation modeling in realistic observations
    
    Examples
    --------
    >>> # Convolve with circular 5-pixel beam
    >>> beam_params = [5.0, 5.0, 0.0]  # [bmin, bmaj, theta]
    >>> convolved_cube = convolve_beam(cube, beam_params)
    
    >>> # Simulate ALMA elliptical beam
    >>> alma_beam = [2.3, 3.1, 45.0]  # Minor axis, major axis, 45° PA
    >>> alma_cube = convolve_beam(high_res_cube, alma_beam)
    
    >>> # VLA beam with specific orientation
    >>> vla_beam = [4.2, 6.8, 120.0]  # Elliptical beam at 120° PA
    >>> vla_cube = convolve_beam(sim_cube, vla_beam)
    
    See Also
    --------
    astropy.convolution.convolve_fft : Underlying convolution implementation
    astropy.convolution.Gaussian2DKernel : 2D Gaussian kernel creation
    radio_beam : Package for more sophisticated beam handling
    """
    # Initialize output array with same dimensions as input
    convolved_spectral_cube_px = np.zeros_like(spectral_cube)
    
    # Extract beam parameters from beam_list [bmin, bmaj, theta]

    if isinstance(beam_list, list) or isinstance(beam_list, tuple):
        # beam_list = [bmin_px, bmaj_px, bpa_deg]
        bmin_avg_px = beam_list[0]      # Minor axis FWHM in pixels
        bmaj_avg_px = beam_list[1]      # Major axis FWHM in pixels
        bpa_deg     = beam_list[2]      # Position angle (deg)

    elif isinstance(beam_list, (int, float)):
        # Same beam for both axes, PA = 0
        bmin_avg_px = beam_list
        bmaj_avg_px = beam_list
        bpa_deg     = 0



    # Convert FWHM to standard deviation for Gaussian kernel
    # Relationship: FWHM = 2 × sqrt(2 × ln(2)) × σ ≈ 2.355 × σ
    x_stddev = bmaj_avg_px / (2 * np.sqrt(2 * np.log(2)))  # Major axis std dev
    y_stddev = bmin_avg_px / (2 * np.sqrt(2 * np.log(2)))  # Minor axis std dev

    # Convert position angle from degrees to radians for kernel creation
    # Position angle measured counterclockwise from positive X-axis (East)
    theta = np.deg2rad(bpa_deg)

    # Create normalized 2D Gaussian beam kernel
    beam = Gaussian2DKernel(x_stddev, y_stddev, theta)
    beam.normalize()  # Ensure total kernel sum = 1 for flux conservation
    
    # Calculate beam area for proper flux scaling after convolution
    # For 2D Gaussian: area = π × σx × σy = π × (FWHM_x × FWHM_y) / (4 × ln(2))
    beam_area = (np.pi * bmaj_avg_px * bmin_avg_px)/(4 * np.log(2))

    # Convolve each spectral channel independently with the beam PSF
    for i in range(spectral_cube.shape[0]):  # Iterate over the spectral dimension (N)
        
        # Apply 2D convolution to each spatial slice using FFT for efficiency
        # convolve_fft handles boundary conditions and preserves flux normalization
        convolved_spectral_cube_px[i, :, :] = convolve_fft(spectral_cube[i, :, :], beam)

    # Scale convolved result by beam area to restore surface brightness units
    # This accounts for flux spreading during convolution process
    convolved_spectral_cube_beam = (convolved_spectral_cube_px * beam_area)

    return convolved_spectral_cube_beam


def apply_and_convolve_noise(spectral_cube, beam_list, peak_snr):
    """
    Apply realistic interferometric noise with beam convolution to a spectral cube.
    
    This function simulates realistic interferometer observations by adding correlated
    noise convolved with a synthesized beam PSF to each spectral channel
    
    The process follows standard radio astronomy calibration procedures where
    raw visibility data contains thermal noise that becomes correlated after
    imaging and beam convolution during the CLEAN deconvolution process.
    
    Parameters
    ----------
    spectral_cube : numpy.ndarray
        Input 3D spectral cube with shape (N, M, M) where:
        - N is the number of spectral channels
        - M x M is the spatial dimension
        Units: Surface brightness (e.g., Jy/beam, mJy/beam)
    beam_list : list of float, length 3
        Beam parameters [bmin_pixels, bmaj_pixels, theta_degrees] where:
        - bmin_pixels: Minor axis FWHM in pixels (highest resolution direction)
        - bmaj_pixels: Major axis FWHM in pixels (lowest resolution direction)
        - theta_degrees: Position angle in degrees measured counterclockwise 
          from positive X-axis (East direction in astronomical coordinates)
    peak_snr : float
        Target peak signal-to-noise ratio
        Defined as: peak_flux / rms_noise
        Higher values produce cleaner data with less noise
    
    Returns
    -------
    numpy.ndarray
        Noisy, beam-convolved spectral cube with same shape as input
        Units: Surface brightness per beam (e.g., Jy/beam, mJy/beam)
    
        
    Examples
    --------
    >>> # Simulate ALMA-like observation with circular beam
    >>> cube = np.random.randn(100, 64, 64) * 0.1  # Mock spectral cube
    >>> beam = [3.0, 3.0, 0.0]  # Circular beam: [bmin, bmaj, theta]
    >>> snr = 10.0  # Target peak SNR
    >>> observed_cube = apply_and_convolve_noise(cube, beam, snr)
    
    >>> # High-resolution VLA observation with elliptical beam
    >>> beam_params = [2.5, 4.2, 45.0]  # Elliptical beam at 45° PA
    >>> noisy_cube = apply_and_convolve_noise(clean_cube, beam_params, 15.0)
    
    See Also
    --------
    apply_noise : Add noise without beam convolution
    convolve_beam : Beam convolution without noise addition
    """
    # Step 1: Estimate peak flux for noise scaling
    # TODO: peak_snr is undefined - should be a function parameter
    peak_flux = np.max(spectral_cube)
    target_noise_rms = peak_flux / peak_snr  # Calculate target noise level

    # Step 2: Generate uncorrelated Gaussian noise with unit variance
    # This creates the base thermal noise before spatial correlation
    white_noise = np.random.normal(0,1.0, spectral_cube.shape)


    # Step 3: Convolve noise with beam to create realistic spatial correlation
    # TODO: beam_width_px is undefined - should be derived from beam_list
    convolved_noise = convolve_beam(white_noise, beam_list)
    
    # Step 4: Measure actual RMS in the convolved noise
    # This accounts for changes in noise statistics after convolution
    current_rms = np.std(convolved_noise)

    # Step 5: Scale convolved noise to match target RMS level
    # This ensures final noise level matches desired SNR
    scaled_noise = convolved_noise * (target_noise_rms / current_rms)

    #print(np.std(white_noise), np.std(scaled_noise), peak_flux/np.std(scaled_noise))

    # Step 6: Add scaled noise to original signal
    # Final result has realistic spatial noise correlation
    noisy_cube = spectral_cube + scaled_noise


    return noisy_cube




def apply_noise(spectral_cube, peak_snr):
    """
    Add uncorrelated Gaussian thermal noise to a spectral cube.
    
    This function simulates basic noise that would be present in
    an interferometric observation before any spatial processing or
    beam convolution. The noise is purely white (uncorrelated) and scales
    with the peak signal to achieve a specified signal-to-noise ratio.
    
    Parameters
    ----------
    spectral_cube : numpy.ndarray
        Input 3D spectral cube with shape (N, M, M) where:
        - N is the number of spectral channels
        - M x M is the spatial dimension
        Units: Surface brightness (e.g., Jy/beam, mJy/beam)
    peak_snr : float
        Target peak signal-to-noise ratio
        Defined as: peak_flux / rms_noise
        Higher values produce cleaner data with less noise
    
    Returns
    -------
    numpy.ndarray
        Noisy spectral cube with same shape and units as input
        Contains original signal plus additive Gaussian noise
    
    Notes
    -----
    - Noise is spatially uncorrelated (white noise)
    - Each pixel receives independent random noise
    - Noise level scales inversely with peak_snr parameter
    - Does not account for beam-induced spatial correlation
    - Commonly used for initial noise studies and algorithm testing
    
    See Also
    --------
    apply_and_convolve_noise : Add noise with beam convolution effects
    convolve_beam : Spatial convolution without noise addition
    """
    # Calculate peak signal amplitude for noise scaling
    signal_max = np.max(spectral_cube)
    
    # Determine noise RMS level from desired peak SNR
    # Higher peak_snr → lower noise level
    mock_noise = signal_max / peak_snr

    # Generate independent Gaussian noise for each spectral channel
    # Each channel gets spatially uncorrelated 2D noise with same RMS
    noise_cube = np.array([
        np.random.normal(0, mock_noise, (spectral_cube.shape[1], spectral_cube.shape[2]))
        for _ in range(spectral_cube.shape[0])  # Loop over spectral channels
    ])

    # Add noise to original signal preserving spectral structure
    noisy_spectral_cube = spectral_cube + noise_cube
    return noisy_spectral_cube

