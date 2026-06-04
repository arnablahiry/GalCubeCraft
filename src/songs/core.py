"""Synthetic IFU cube generator (toy model).

This module implements a compact, self-contained pipeline to create toy
``n_vel × ny × nx`` spectral cubes that mimic IFU observations of disk
galaxies. It focuses on clarity and inspectability over performance, and
encapsulates the following responsibilities:

- Build a 3D light distribution from a Sérsic radial profile combined with an
    exponential vertical profile (see :meth:`SONGS.sersic_flux_density_3d`).
- Create a simple analytical rotation curve and assign tangential velocities to
    the 3D grid (see :meth:`SONGS.milky_way_rot_curve_analytical`).
- Rotate the full 3D flux and velocity fields to simulate arbitrary viewing
    angles and project galaxy emission into velocity bins to form a spectral
    cube (see :meth:`SONGS.rotated_system` and
    :meth:`SONGS.make_spectral_cube`).
- Optionally convolve the final cube with a telescope beam and save cubes to
    disk (see :meth:`SONGS.generate_cubes`).

Design notes
------------
- Coordinates: internal grids are defined in pixels and converted to physical
    units (kpc) using a pixel scale stored per cube.
- Velocities: rotation is computed analytically and random Gaussian scatter is
    added per voxel to mimic dispersion.
- Output: spectral cubes are produced in units of flux per pixel and are
    optionally downsampled/averaged in the spectral axis to simulate channel
    binning/oversampling.

Usage and API
-------------
- The primary user-facing class is :class:`SONGS`. Call
    ``g.generate_cubes()`` to produce one or more cubes; the method appends the
    generated outputs to ``g.results`` and also returns that list. Each entry in
    ``g.results`` is typically a tuple ``(spectral_cube, params)`` where
    ``spectral_cube`` has shape ``(n_spectral, ny, nx)`` and ``params`` contains
    metadata (beam info, pixel scale, velocity axis, etc.). The GUI and utilities
    in this repository expect this layout.

- Visualisation helpers are provided as top-level functions in
    :mod:`songs.visualise` (``moment0``, ``moment1``, ``spectrum``,
    ``slice_view``). Plotting is intentionally kept separate from the generation
    core to avoid UI/plotting dependencies here.

This file provides the :class:`SONGS` helper class which encapsulates
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
import h5py
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


# Default diffuse-emission knobs. Units: lengths in pixels (of the full output
# grid, same pixel scale as the per-galaxy init grid); velocities in km/s.
# Length factors multiply the central galaxy's own Re or hz (also in pixels),
# so realistic sizes are obtained independently of the chosen pixel scale.
DEFAULT_DIFFUSE_PARAMS = {
    'enabled': True,
    # Halo around the central galaxy. Amplitudes are deliberately small so the
    # halo surface brightness stays well below the satellite disks'.
    'halo_Re_factor': 3.0,
    'halo_Se_factor': 0.065,
    'halo_hz_factor': 2.0,
    'halo_n': 0.5,
    'halo_sigma_vz': 70.0,
    # Bridges between central and each satellite. The bridge lives strictly
    # within [start_frac, 1 - stop_frac] along the central→satellite link, so
    # it never extends past the satellite or behind the central. Width tapers
    # from the halo end (thick) to the satellite end (narrow).
    'bridge_start_frac': 0.2,             # emerge out of the halo, not from the central core
    'bridge_stop_frac': 0.0,              # bridge reaches exactly the satellite
    'bridge_width_start_factor': 3.0,     # * central Re (σ at the halo end — halo-diffuse)
    'bridge_width_end_factor': 2.0,       # * satellite Re (σ at the satellite end)
    'bridge_edge_fade': 0.1,              # smooth fade-in / fade-out within the active segment
    'bridge_Se_factor': 0.03,             # * min(Se_central, Se_sat) (peak amplitude on-axis)
    # Streamers / tidal tails extending from each satellite. The streamer is
    # a Bezier curve in 3D. Designed so that
    # channel-by-channel viewers see the satellite's diffuse material visibly
    # traverse from one spatial position to another as LOS velocity changes.
    'tail_length_factor': 0.25,          # arc length as a fraction of sep (shorter = stays near source)
    'tail_curvature': 0.15,              # overall perpendicular offset of P_end (× sep)
    'tail_width_factor': 1.0,            # Gaussian σ_base (× satellite Re)
    'tail_Se_factor': 0.15,              # peak amplitude (× Se_satellite)
    'tail_vel_gradient': 0.5,            # dimensionless scale: v_grad = scale × (v_sat - v_central)
    'tail_sigma_vz': 100.0,              # per-voxel LOS dispersion (km/s)
    'tail_n_samples': 40,                # Gaussian samples along the curve
    'tail_n_control_points': 4,          # control points in the trunk Bezier
    'tail_jitter': 0.1,                  # perpendicular jitter of interior control points (× sep)
    'tail_decay_scale': 1.5,             # exponential decay length in u (brightness ∝ exp(-u/scale))
}


def _build_diffuse_cubes(grid_size, galaxy_centers, gal_params_list,
                         gal_systemic_vels, diffuse_params, rng=None):
    """Build a full-grid diffuse flux cube and a matching LOS-velocity cube.

    Three additive components (only the halo is used when n_gal == 1):
    - Halo: 2D Sérsic profile in the disk plane (n = 0.5, Gaussian-like) times
      an exponential vertical profile, centered on the central galaxy with
      scaled Re, hz, and reduced amplitude via diffuse_params factors.
    - Bridges: Gaussian tubes along the line connecting the central galaxy to
      each satellite, with flux strictly clamped to [bridge_start_frac,
      1 - bridge_stop_frac] along that line to prevent leakage. Width tapers
      from halo end to satellite end. LOS velocity interpolates linearly
      between endpoints.
    - Tidal tails: Curved Gaussian arcs extending from each satellite away
      from the central galaxy. Each tail is sampled at tail_n_samples points
      along a quadratic curve with perpendicular curvature. One-sided sigmoid
      gate suppresses emission on the side facing back toward the central.

    The LOS velocity cube is a flux-weighted blend over the three components,
    with per-voxel Gaussian noise added inside each component before blending.
    All three components are binned into the final spectral cube using the
    same velocity-channel masks as the per-galaxy disk components.
    """
    if rng is None:
        rng = np.random.default_rng()

    dp = diffuse_params
    shape = (grid_size,) * 3
    X, Y, Z = np.meshgrid(np.arange(grid_size),
                          np.arange(grid_size),
                          np.arange(grid_size), indexing='ij')

    diffuse_flux = np.zeros(shape)
    v_accum = np.zeros(shape)
    w_accum = np.zeros(shape)

    # Per-satellite streamer cubes — flux/velocity that should ALSO be routed
    # into the per-galaxy clean cubes (/galaxies/cubes) so the streamer counts
    # as part of each satellite's ground truth for source-separation ML.
    streamer_per_galaxy = [None] * len(galaxy_centers)

    def _accumulate(flux_field, vel_field):
        diffuse_flux[...] = diffuse_flux + flux_field
        v_accum[...] = v_accum + vel_field * flux_field
        w_accum[...] = w_accum + flux_field

    def _eval_bezier(ctrl, u_arr):
        """De Casteljau Bezier evaluation. ctrl: (K, 3); u_arr: (M,) → (M, 3)."""
        pts = np.broadcast_to(np.asarray(ctrl, dtype=float),
                              (u_arr.shape[0], ctrl.shape[0], 3)).copy()
        u = np.asarray(u_arr, dtype=float)[:, None, None]
        while pts.shape[1] > 1:
            pts = (1.0 - u) * pts[:, :-1, :] + u * pts[:, 1:, :]
        return pts[:, 0, :]

    # Halo around central
    c0 = np.asarray(galaxy_centers[0], dtype=float)
    Re_c = float(gal_params_list[0]['Re'])
    Se_c = float(gal_params_list[0]['Se'])
    hz_c = float(gal_params_list[0]['hz'])

    n_halo = dp['halo_n']
    Re_halo = dp['halo_Re_factor'] * Re_c
    Se_halo = dp['halo_Se_factor'] * Se_c
    hz_halo = dp['halo_hz_factor'] * hz_c
    bn = 2 * n_halo - 1/3 + 4/(405 * n_halo) + 46/(25515 * n_halo**2)

    # Shared per-voxel velocity noise used by both the halo and all bridges so
    # they are always in the same spectral channels and fade together.
    halo_vz_noise = rng.normal(0.0, dp['halo_sigma_vz'], shape)

    r3d = np.sqrt((X - c0[0])**2 + (Y - c0[1])**2 + (Z - c0[2])**2)
    halo_flux = Se_halo * np.exp(-bn * ((r3d / max(Re_halo, 1e-6))**(1.0 / n_halo) - 1.0))
    halo_vel = gal_systemic_vels[0] + halo_vz_noise
    _accumulate(halo_flux, halo_vel)

    # Bridges and tidal tails per satellite
    for i in range(1, len(galaxy_centers)):
        ci = np.asarray(galaxy_centers[i], dtype=float)
        vec = ci - c0
        sep = float(np.linalg.norm(vec))
        if sep < 1e-6:
            continue
        direction = vec / sep

        Re_s = float(gal_params_list[i]['Re'])
        Se_s = float(gal_params_list[i]['Se'])
        hz_s = float(gal_params_list[i]['hz'])

        # Bridge (Gaussian tube from halo edge toward satellite, tapering width).
        # The active segment is [s_start, s_end] with s_start > 0 (emerges from
        # the halo rather than the central core) and s_end < 1 (stops before
        # the satellite). Flux is strictly zero outside this segment, so the
        # bridge cannot extend past the satellite or behind the central.
        dx = X - c0[0]
        dy = Y - c0[1]
        dz = Z - c0[2]
        l = dx * direction[0] + dy * direction[1] + dz * direction[2]
        s = l / sep

        s_start = float(np.clip(dp['bridge_start_frac'], 0.0, 0.49))
        s_end = float(np.clip(1.0 - dp['bridge_stop_frac'], s_start + 1e-3, 1.0))
        fade = max(float(dp['bridge_edge_fade']), 1e-3)

        # Clamp s to [s_start, s_end] for the closest-point on the bridge axis;
        # this keeps perpendicular distance well defined at the capped ends.
        s_axis = np.clip(s, s_start, s_end)
        closest_x = c0[0] + s_axis * vec[0]
        closest_y = c0[1] + s_axis * vec[1]
        closest_z = c0[2] + s_axis * vec[2]
        perp_r = np.sqrt((X - closest_x)**2 + (Y - closest_y)**2 + (Z - closest_z)**2)

        # Position-dependent width scaled by disk Re so the bridge is as
        # spatially diffuse as the halo/satellite bodies themselves.
        sigma_start = max(dp['bridge_width_start_factor'] * Re_c, 1.0)
        sigma_end = max(dp['bridge_width_end_factor'] * Re_s, 1.0)
        denom = max(s_end - s_start, 1e-6)
        s_norm = np.clip((s - s_start) / denom, 0.0, 1.0)
        sigma_bridge = sigma_start * (1.0 - s_norm) + sigma_end * s_norm

        # Trapezoidal window over [s_start, s_end], exactly zero outside.
        ramp_in = np.clip((s - s_start) / fade, 0.0, 1.0)
        ramp_out = np.clip((s_end - s) / fade, 0.0, 1.0)
        window = np.minimum(ramp_in, ramp_out)

        Se_br = dp['bridge_Se_factor'] * min(Se_c, Se_s)
        bridge_flux = Se_br * np.exp(-0.5 * (perp_r / sigma_bridge)**2) * window

        s_clipped = np.clip(s, 0.0, 1.0)
        bridge_vel_mean = (1.0 - s_clipped) * gal_systemic_vels[0] + s_clipped * gal_systemic_vels[i]
        bridge_vel = bridge_vel_mean + halo_vz_noise
        _accumulate(bridge_flux, bridge_vel)

        # ------------------------------------------------------------------
        # Streamer (3D Bezier curve from the satellite outward, optionally
        # Designed so that
        # the LOS velocity varies smoothly along the arc: channel-by-channel
        # viewers see the diffuse material visibly march from one spatial
        # position to another, with possible mid-arc kinematic splits.
        # ------------------------------------------------------------------
        rv = rng.normal(size=3)
        perp = rv - np.dot(rv, direction) * direction
        if np.linalg.norm(perp) < 1e-6:
            alt = np.array([1.0, 0.0, 0.0])
            perp = alt - np.dot(alt, direction) * direction
        perp = perp / np.linalg.norm(perp)

        perp2 = np.cross(direction, perp)
        n_perp2 = np.linalg.norm(perp2)
        if n_perp2 > 1e-9:
            perp2 = perp2 / n_perp2
        else:
            perp2 = np.array([0.0, 0.0, 1.0])

        L_tail = float(dp['tail_length_factor']) * sep
        sigma_base = max(float(dp['tail_width_factor']) * Re_s, 1.0)
        Se_tail_peak = float(dp['tail_Se_factor']) * Se_s
        n_samples = max(int(dp['tail_n_samples']), 2)
        ds_norm = 5.0 / n_samples

        # Per-streamer accumulators (additionally fed into the global diffuse
        # field via _accumulate at the end).
        str_flux = np.zeros(shape)
        str_v_accum = np.zeros(shape)
        str_w_accum = np.zeros(shape)

        def _local_accum(f, v):
            str_flux[...] = str_flux + f
            str_v_accum[...] = str_v_accum + v * f
            str_w_accum[...] = str_w_accum + f

        # Trunk: Bezier in 3D from c_i to P_end with K-2 jittered interior pts.
        K = max(int(dp['tail_n_control_points']), 2)

        # End direction: random but constrained to the hemisphere away from the
        # central galaxy.  Tidal tails always extend outward from the progenitor,
        # never back toward the perturber.  We draw a random direction and flip it
        # if it points toward the central (dot product with the away axis < 0).
        rand_raw = rng.normal(size=3)
        if np.linalg.norm(rand_raw) < 1e-9:
            rand_raw = np.array([1.0, 0.0, 0.0])
        end_dir = rand_raw / np.linalg.norm(rand_raw)
        if np.dot(end_dir, direction) < 0:
            end_dir = -end_dir

        # Keep Gaussian centres far enough from every wall that the 10%-of-peak
        # isophote stays inside the FOV.  At distance d from the centre,
        # flux = exp(-0.5*(d/σ)²); for flux < 0.10 we need d > 2.15·σ.
        # Use 2.5·σ for a comfortable buffer.
        clamp_lo = float(2.5 * sigma_base)
        clamp_hi = float(grid_size) - 1.0 - float(2.5 * sigma_base)
        # If the satellite itself is outside the safe zone (e.g. very near the edge),
        # fall back to the grid boundary so we still produce something.
        if clamp_lo >= clamp_hi:
            clamp_lo, clamp_hi = 0.0, float(grid_size) - 1.0

        L_to_wall = L_tail
        for ax in range(3):
            if end_dir[ax] > 1e-9:
                d = (clamp_hi - ci[ax]) / end_dir[ax]
            elif end_dir[ax] < -1e-9:
                d = (clamp_lo - ci[ax]) / end_dir[ax]
            else:
                continue
            if d > 0:
                L_to_wall = min(L_to_wall, d)
        L_tail = max(min(L_tail, L_to_wall * 0.9), sep * 0.03)

        # P_end along the random direction; curvature adds perpendicular offset.
        P_end = ci + L_tail * end_dir + float(dp['tail_curvature']) * sep * perp
        P_end = np.clip(P_end, clamp_lo, clamp_hi)

        jitter_scale = float(dp['tail_jitter']) * sep
        ctrl_pts = [ci.copy()]
        for k in range(1, K - 1):
            t = k / (K - 1)
            base = (1.0 - t) * ci + t * P_end
            j1 = rng.normal(scale=jitter_scale)
            j2 = rng.normal(scale=jitter_scale)
            pt = base + j1 * perp + j2 * perp2
            ctrl_pts.append(np.clip(pt, clamp_lo, clamp_hi))
        ctrl_pts.append(P_end)
        ctrl_pts = np.asarray(ctrl_pts, dtype=float)

        u_vals = np.linspace(0.0, 1.0, n_samples)
        trunk_pts = _eval_bezier(ctrl_pts, u_vals)

        # Amplitude: exponential decay from root (u=0) to tip (u=1) — physically,
        # tidal debris density decreases with distance from the progenitor.
        decay = max(float(dp['tail_decay_scale']), 0.05)
        amp_profile = np.exp(-u_vals / decay)

        # Width: tapers toward the tip (broader near the satellite where material
        # is freshly stripped, narrowing as the stream thins out).
        sigma_profile = np.maximum(sigma_base * (1.0 - 0.5 * u_vals), 1.0)

        # Velocity gradient derived from the satellite–central systemic velocity
        # difference.  Sign is physical: receding satellite → trailing debris is
        # more redshifted; approaching satellite → more blueshifted.
        v_start = gal_systemic_vels[i]
        delta_v = gal_systemic_vels[i] - gal_systemic_vels[0]
        v_grad = delta_v * float(dp['tail_vel_gradient'])
        vel_along_u = v_start + v_grad * u_vals

        # Precompute an in-grid mask: voxels outside [clamp_lo, clamp_hi] on any
        # axis are zeroed so Gaussian tails never reach the FOV boundary.
        ingrid_mask = ((X >= clamp_lo) & (X <= clamp_hi) &
                       (Y >= clamp_lo) & (Y <= clamp_hi) &
                       (Z >= clamp_lo) & (Z <= clamp_hi)).astype(np.float32)

        trunk_pts = np.clip(trunk_pts, clamp_lo, clamp_hi)
        for k in range(n_samples):
            p = trunk_pts[k]
            d2 = (X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2
            seg_flux = (amp_profile[k] * Se_tail_peak
                        * np.exp(-0.5 * d2 / sigma_profile[k] ** 2) * ds_norm * ingrid_mask)
            seg_vel = vel_along_u[k] + rng.normal(0.0, dp['tail_sigma_vz'], shape)
            _local_accum(seg_flux, seg_vel)

        # Finalise per-streamer velocity, route into both per-galaxy and diffuse paths.
        with np.errstate(invalid='ignore', divide='ignore'):
            str_vel = np.where(str_w_accum > 1e-12, str_v_accum / str_w_accum, 0.0)
        _accumulate(str_flux, str_vel)
        streamer_per_galaxy[i] = (str_flux, str_vel)

    with np.errstate(invalid='ignore', divide='ignore'):
        diffuse_vel = np.where(w_accum > 1e-12, v_accum / w_accum, 0.0)
    return diffuse_flux, diffuse_vel, streamer_per_galaxy


def _save_cube_hdf5(path, cube, params, gen, cube_idx):
    """Write one spectral cube plus full ground-truth metadata to HDF5.

    Layout:
      /cube                           float32 (n_vel, n_y, n_x)   — beam-convolved spectral cube
      /channel_velocities_km_s        float64 (n_vel,)            — channel centres
      /galaxies/positions_xyz_px      int     (n_gals, 3)         — (x, y, z) in cube pixels
      /galaxies/types                 str     (n_gals,)           — 'central' or 'satellite'
      /galaxies/Re_px, Se, hz_px,     float   (n_gals,)           — disk parameters
        sersic_n, inclination_x_deg,
        inclination_y_deg,
        sigma_vz_km_s, v0_km_s,
        systemic_vel_km_s,
        channel_index,
        distance_to_central_px,
        distance_to_central_kpc
      /beam (group, attrs: bmin_px, bmaj_px, bpa_deg)
      /diffuse_params (group with one attr per knob)

    File-level attrs expose the common observational metadata:
      grid_size, n_channels, n_gals, n_satellites,
      spatial_resolution_kpc_per_px, fov_kpc,
      spectral_resolution_km_s, diffuse_emission.
    """
    gcenters = np.asarray(params.get('galaxy_centers'))                       # (n_gals, 3)
    avg_v = np.asarray(params.get('average_vels'), dtype=np.float64)
    sys_v = np.asarray(params.get('systemic_vels'), dtype=np.float64)
    pix_scale = float(gen.all_pix_spatial_scales[cube_idx][0])
    n_gals_i = int(gen.n_gals[cube_idx])
    grid_size = int(gen.grid_size)
    beam = np.asarray(gen.beam_info, dtype=np.float64)

    # Per-galaxy helpers
    dist_px = np.linalg.norm(gcenters - gcenters[0][None, :], axis=1)
    dist_kpc = dist_px * pix_scale
    ch_idx = np.argmin(np.abs(sys_v[:, None] - avg_v[None, :]), axis=1)

    # Spectral resolution (km/s/ch). Prefer the generator's explicit value
    # when available (Phy); otherwise derive from channel spacing.
    spec_res = getattr(gen, 'spectral_resolution', None)
    if spec_res is None:
        spec_res = float(np.mean(np.diff(avg_v))) if len(avg_v) > 1 else 0.0

    types = np.array(['central'] + ['satellite'] * (n_gals_i - 1), dtype=object)

    with h5py.File(path, 'w') as f:
        f.create_dataset('cube', data=cube.astype(np.float32), compression='gzip', compression_opts=4)
        f.create_dataset('channel_velocities_km_s', data=avg_v)

        f.attrs['grid_size'] = grid_size
        f.attrs['n_channels'] = int(cube.shape[0])
        f.attrs['n_gals'] = n_gals_i
        f.attrs['n_satellites'] = n_gals_i - 1
        f.attrs['spatial_resolution_kpc_per_px'] = pix_scale
        f.attrs['fov_kpc'] = grid_size * pix_scale
        f.attrs['spectral_resolution_km_s'] = float(spec_res)
        f.attrs['diffuse_emission'] = bool(params.get('diffuse_emission', False))

        g = f.create_group('galaxies')
        g.create_dataset('positions_xyz_px', data=np.asarray(gcenters, dtype=np.int64))
        g.create_dataset('types', data=types.astype(h5py.string_dtype()))
        g.create_dataset('Re_px', data=np.asarray(gen.all_Re[cube_idx]))
        g.create_dataset('Se', data=np.asarray(gen.all_Se[cube_idx]))
        g.create_dataset('hz_px', data=np.asarray(gen.all_hz[cube_idx]))
        g.create_dataset('sersic_n', data=np.asarray(gen.all_n[cube_idx]))
        g.create_dataset('inclination_x_deg', data=np.asarray(gen.all_gal_x_angles[cube_idx]))
        g.create_dataset('inclination_y_deg', data=np.asarray(gen.all_gal_y_angles[cube_idx]))
        g.create_dataset('sigma_vz_km_s', data=np.asarray(gen.all_gal_vz_sigmas[cube_idx]))
        g.create_dataset('v0_km_s', data=np.asarray(gen.all_gal_v_0[cube_idx]))
        g.create_dataset('systemic_vel_km_s', data=sys_v)
        g.create_dataset('channel_index', data=np.asarray(ch_idx, dtype=np.int64))
        g.create_dataset('distance_to_central_px', data=dist_px)
        g.create_dataset('distance_to_central_kpc', data=dist_kpc)

        # Per-galaxy diffuse-free spectral cubes in the final FOV, beam- and
        # spectrally-smoothed to match `/cube`. Shape: (n_gals, n_ch, n_y, n_x).
        per_gal_cubes = params.get('per_galaxy_cubes')
        if per_gal_cubes is not None:
            g.create_dataset(
                'cubes',
                data=np.asarray(per_gal_cubes, dtype=np.float32),
                compression='gzip', compression_opts=4,
            )

        bg = f.create_group('beam')
        bg.attrs['bmin_px'] = float(beam[0])
        bg.attrs['bmaj_px'] = float(beam[1])
        bg.attrs['bpa_deg'] = float(beam[2])

        dp_grp = f.create_group('diffuse_params')
        for k, v in getattr(gen, 'diffuse_params', {}).items():
            try:
                dp_grp.attrs[k] = v
            except TypeError:
                dp_grp.attrs[k] = str(v)


class SONGS:
    """Generator for ensembles of synthetic IFU spectral cubes (pixel units).

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

    The class operates primarily in pixel units and samples parameter values
    (e.g., sizes, brightness, orientations) from uniform ranges over physically
    viable values to produce diverse, unique systems. It intentionally focuses
    on clarity and inspectability rather than performance; nested Python loops
    are used to build fields which is adequate for moderate grid sizes used in
    examples.
    """
    
    def __init__(self, n_gals=None, n_cubes=1, resolution='all', offset_gals=30, beam_info = [4,4,0], grid_size=125, n_spectral_slices=40, n_sersic=None, save=False, fname=None, verbose=True, seed=None, diffuse_params=None, sat_brightness_frac=None, diffuse_flux_frac=1.0, sat_vel_dispersion=150.0):
        """
        Initialize the SONGS generator.

        Parameters
        ----------
        n_gals : int or None
            If an integer, the fixed number of galaxies per cube. If ``None``,
            a random number of galaxies (1--3) is sampled for each cube.
        n_cubes : int
            Number of cubes to generate when ``generate_cubes`` is called.
        resolution : {'all', 'resolved', 'unresolved', 'visualise'} or float
            Controls sampling of effective radii relative to the beam. 'all'
            samples a broad log-uniform range; 'resolved' and 'unresolved'
            constrain the ratio; 'visualise' uses a fixed set useful for
            producing illustrative figures. When a float is provided (e.g.
            via the GUI for single-cube runs), it is interpreted as the
            ratio ``r = Re / (beam_minor/2)`` and used directly.
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
            Number of output spectral channels. Internally a 5× oversampled
            grid is used and then averaged back into ``n_spectral_slices``
            channels to simulate spectral binning.
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
        diffuse_params : dict or None
            Optional overrides for diffuse emission knobs (halo, bridges,
            tidal tails). If provided, updates are merged with
            ``DEFAULT_DIFFUSE_PARAMS``. If ``None``, all defaults are used.
            Set ``diffuse_params['enabled'] = False`` to disable diffuse
            components. See ``DEFAULT_DIFFUSE_PARAMS`` for available keys.

        Notes
        -----
        After instantiation the object contains arrays (e.g. ``all_Re``,
        ``all_Se``, ``all_pix_spatial_scales``) describing the sampled
        parameters for each cube. Call :meth:`generate_cubes` to run the
        full pipeline and fill ``self.results``.

        Example
        -------
        >>> g = SONGS(n_cubes=2, grid_size=125, n_spectral_slices=40, seed=42)
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

        # Galaxy separation parameter: scalar px or (min_px, max_px) tuple.
        self.offset_gals = offset_gals

        # Satellite brightness fraction relative to central Se.
        # None → legacy random range [Se/2, Se/1.6].
        self.sat_brightness_frac = sat_brightness_frac
        self.sat_vel_dispersion = float(sat_vel_dispersion)

        # Diffuse-emission configuration (halo, bridges, tidal tails)
        merged_dp = dict(DEFAULT_DIFFUSE_PARAMS)
        if diffuse_params:
            merged_dp.update(diffuse_params)
        # Scale all diffuse Se factors by diffuse_flux_frac.
        if diffuse_flux_frac != 1.0:
            for _key in ('halo_Se_factor', 'bridge_Se_factor', 'tail_Se_factor'):
                if _key in merged_dp:
                    merged_dp[_key] = merged_dp[_key] * diffuse_flux_frac
        self.diffuse_params = merged_dp

        # Determine number of galaxies per cube
        if n_gals is None:
            # Default: 1–2 galaxies per cube (legacy behaviour).
            self.n_gals = np.random.randint(1, 3, n_cubes)
        elif isinstance(n_gals, (tuple, list)) and len(n_gals) == 2:
            # Random inclusive range per cube: n_gals=(lo, hi) → uniform in [lo, hi].
            lo, hi = int(n_gals[0]), int(n_gals[1])
            self.n_gals = np.random.randint(lo, hi + 1, n_cubes)
        else:
            # Fixed number of galaxies across all cubes
            self.n_gals = [int(n_gals) for _ in range(n_cubes)]

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

                # Sample satellite Sérsic indices first so we can compute bn
                # and normalise to peak surface brightness (Se*exp(bn)) rather
                # than Se alone.  This ensures a high-n satellite (steep cusp)
                # never appears brighter than the central regardless of index.
                sat_ns = list(np.random.uniform(0.5, 1.5, n_gal - 1))
                n_sersics += sat_ns

                n_c = float(n_sersics[0])
                bn_c = 2*n_c - 1/3 + 4/(405*n_c) + 46/(25515*n_c**2)
                peak_c = Se[0] * np.exp(bn_c)

                _b = float(np.clip(self.sat_brightness_frac, 1e-6, 1.0 - 1e-6)) \
                    if self.sat_brightness_frac is not None \
                    else None
                for _k in range(n_gal - 1):
                    n_s = float(sat_ns[_k])
                    bn_s = 2*n_s - 1/3 + 4/(405*n_s) + 46/(25515*n_s**2)
                    frac = _b if _b is not None else np.random.uniform(0.15, 0.35)
                    # Se_sat chosen so peak_sat = frac * peak_c, guaranteed < peak_c
                    Se_sat = frac * peak_c / np.exp(bn_s) * np.random.uniform(0.85, 1.0)
                    Se.append(float(Se_sat))

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
        >>> SONGS.milky_way_rot_curve_analytical(np.array([0.1,1,10]), 200, 5.0, 1.0)
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
        - The vertical structure is a symmetric exponential: :math:`\exp(-|z|/h_z)`.

        Example
        -------
        >>> nx = ny = nz = 21
        >>> x = np.arange(nx) - (nx-1)/2
        >>> X,Y,Z = np.meshgrid(x,x,x,indexing='ij')
        >>> S = SONGS.sersic_flux_density_3d(X, Y, Z, Se=0.1, Re=5.0, n=1.0, hz=0.5)
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
        """Create a rotated 3D galaxy flux cube and LOS velocity cube.

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
            3D line-of-sight velocity cube (same shape) giving LOS velocity
            (km/s) at each voxel after rotation and projection.

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


    def make_spectral_cube(self, rotated_disks, rotated_vel_z_cubes, pix_spatial_scale, gal_params_list=None):
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
        gal_params_list : list of dict, optional
            Per-galaxy parameter dicts with keys 'Re', 'Se', 'hz' (pixel
            units). When provided and ``self.diffuse_params['enabled']`` is
            True, three diffuse components are built on the full grid and
            added to the spectral cube: (1) a halo (2D Sérsic × vertical
            exponential) around the central galaxy, (2) one bridge (tapered
            Gaussian tube with clamped spatial extent) per satellite, and
            (3) one tidal tail (curved arc with one-sided sigmoid gate) per
            satellite.

        Returns
        -------
        spectral_cube_Jy_px : ndarray
            Spectral cube with shape ``(n_channels, grid_size, grid_size)``
            where the spectral axis corresponds to velocity-binned slices.
        params_gen : dict
            Metadata with keys: 'galaxy_centers', 'average_vels', 'beam_info',
            'n_gals', and 'pix_spatial_scale'.

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

        galaxy_centers = []

        half_size = init_grid_size // 2
        min_pos = half_size
        max_pos = grid_size - half_size

        # First galaxy near the center
        x = int(np.clip(center_final_cube[0], min_pos, max_pos - 1))
        y = int(np.clip(center_final_cube[1], min_pos, max_pos - 1))
        z = int(np.clip(center_final_cube[2], min_pos, max_pos - 1))
        galaxy_centers.append(np.array([x, y, z]))

        # Minimum 3-D separation: just large enough that galaxy disks don't overlap.
        _min_sep_3d = init_grid_size // 2 + 4

        # Resolve offset_gals: scalar → (min_sep, max); tuple → (min, max).
        # _off_min is floored to _min_sep_3d so satellites are always placed far
        # enough from the central to be visually distinct.
        if isinstance(self.offset_gals, (tuple, list)) and len(self.offset_gals) == 2:
            _off_min = int(round(self.offset_gals[0]))
            _off_max = int(round(self.offset_gals[1]))
        else:
            _off_min = _min_sep_3d
            _off_max = int(round(self.offset_gals))
        _off_min = max(_min_sep_3d, _off_min)
        _off_max = max(_off_min, _off_max)

        # Additional galaxies at a random distance in [_off_min, _off_max] on each axis.
        _grid_max = max_pos - min_pos - 1
        for i in range(1, n_galaxies):
            placed = False
            best_cand = None
            best_min_dist = -1.0
            for _expand in range(8):
                lo = min(_off_min, _grid_max)
                hi = min(_off_max + _expand * max(4, _off_max // 4), _grid_max)
                hi = max(hi, lo + 1)
                for _attempt in range(400):
                    dx = int(np.random.randint(lo, hi + 1)) * np.random.choice([-1, 1])
                    dy = int(np.random.randint(lo, hi + 1)) * np.random.choice([-1, 1])
                    dz = int(np.random.randint(lo, hi + 1)) * np.random.choice([-1, 1])
                    cx = galaxy_centers[0][0] + dx
                    cy = galaxy_centers[0][1] + dy
                    cz = galaxy_centers[0][2] + dz
                    # Skip out-of-bounds rather than clipping — clipping causes
                    # all candidates to snap to the same boundary corners.
                    if not (min_pos <= cx < max_pos and
                            min_pos <= cy < max_pos and
                            min_pos <= cz < max_pos):
                        continue
                    cand = np.array([cx, cy, cz], dtype=float)
                    min_dist = min(np.linalg.norm(cand - np.asarray(c, dtype=float))
                                   for c in galaxy_centers)
                    # Also enforce 2-D (x,y) separation — the cube is projected
                    # along Z so galaxies at the same sky position always overlap.
                    min_dist_2d = min(
                        np.linalg.norm(cand[:2] - np.asarray(c, dtype=float)[:2])
                        for c in galaxy_centers)
                    # Track best candidate by combined 3D+2D score so the
                    # fallback also avoids sky projection overlaps.
                    score = min(min_dist, min_dist_2d)
                    if score > best_min_dist:
                        best_min_dist = score
                        best_cand = cand
                    if min_dist >= _off_min and min_dist_2d >= _off_min:
                        placed = True
                        break
                if placed:
                    break
            if best_cand is None:
                # All offset-constrained attempts failed — scan in-bounds positions
                # and pick the one maximising min(3D dist, 2D sky dist).
                candidates = []
                for _ in range(1000):
                    cx = int(np.random.randint(min_pos, max_pos))
                    cy = int(np.random.randint(min_pos, max_pos))
                    cz = int(np.random.randint(min_pos, max_pos))
                    cand = np.array([cx, cy, cz], dtype=float)
                    d3 = min(np.linalg.norm(cand - np.asarray(c, dtype=float))
                             for c in galaxy_centers)
                    d2 = min(np.linalg.norm(cand[:2] - np.asarray(c, dtype=float)[:2])
                             for c in galaxy_centers)
                    candidates.append((min(d3, d2), cand))
                best_cand = max(candidates, key=lambda t: t[0])[1]
            galaxy_centers.append(best_cand.astype(int))

        if self.verbose:
            for idx, center in enumerate(galaxy_centers):
                print(f"Centre of galaxy {idx + 1}: {center}")

        # Apply Hubble flow + peculiar velocity relative to the first galaxy
        reference_z = galaxy_centers[0][2]
        H_z = cosmo.H(0).value  # km/s/Mpc

        for i in range(1, n_galaxies):
            delta_z_kpc = (galaxy_centers[i][2] - reference_z)*pix_spatial_scale
            delta_z_mpc = delta_z_kpc * 1e-3  # Convert kpc to Mpc
            relative_velocity = H_z * delta_z_mpc

            # Peculiar velocity: random blueshift/redshift for each satellite
            peculiar_vel = np.random.normal(0.0, self.sat_vel_dispersion)
            relative_velocity += peculiar_vel

            if self.verbose:
                direction = "farther" if delta_z_kpc > 0 else "closer"
                print(f"Galaxy {i+1} is {direction} than galaxy 1 by {delta_z_kpc:.2f} kpc")
                print(f"→ Hubble + peculiar velocity offset: {relative_velocity:.2f} km/s  (peculiar: {peculiar_vel:+.1f} km/s)")

            # Add velocity offset to simulate redshift/blueshift
            rotated_vel_z_cubes[i] = (rotated_vel_z_cubes[i]+relative_velocity)


        # Flux-weighted systemic LOS velocity per galaxy (includes Hubble offset
        # already applied above). Useful as ground-truth for ML detection.
        gal_systemic_vels = []
        for disk, vel_cube in zip(rotated_disks, rotated_vel_z_cubes):
            total = float(np.sum(disk))
            if total > 0:
                gal_systemic_vels.append(float(np.sum(disk * vel_cube) / total))
            else:
                gal_systemic_vels.append(0.0)

        # Build diffuse emission (halo + bridges + tidal tails) on the full grid
        use_diffuse = (gal_params_list is not None
                       and getattr(self, 'diffuse_params', {}).get('enabled', False))
        if use_diffuse:
            if self.verbose:
                print('Building diffuse emission (halo, bridges, streamers)...')
            diffuse_flux, diffuse_vel, streamer_per_galaxy = _build_diffuse_cubes(
                grid_size, galaxy_centers, gal_params_list,
                gal_systemic_vels, self.diffuse_params,
            )
        else:
            diffuse_flux = None
            diffuse_vel = None
            streamer_per_galaxy = [None] * n_galaxies


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
        # Per-galaxy (diffuse-free) spectral slices in the final FOV.
        per_gal_slices = [[] for _ in range(n_galaxies)]
        average_vels = np.zeros((n_spectral_slices - 1))


        for i in range(n_spectral_slices - 1):
            combined_cube = np.zeros((grid_size, grid_size, grid_size))
            per_gal_slab = [np.zeros((grid_size, grid_size)) for _ in range(n_galaxies)]
            for g, (disk, vel_cube, center) in enumerate(zip(rotated_disks, rotated_vel_z_cubes, galaxy_centers)):

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
                # Clean per-galaxy contribution (LOS-projected, diffuse-free
                # other than the streamer that belongs to this satellite).
                per_gal_slab[g][xs:xe, ys:ye] += selected_cube.sum(axis=2)

            # Route each satellite's streamer flux for this channel into the
            # per-galaxy clean cube (full FOV, not the init-grid slab).
            for g in range(n_galaxies):
                sg = streamer_per_galaxy[g]
                if sg is None:
                    continue
                str_flux_g, str_vel_g = sg
                if i < n_spectral_slices - 2:
                    sc = (str_vel_g >= limits[i]) & (str_vel_g < limits[i+1])
                else:
                    sc = (str_vel_g >= limits[i]) & (str_vel_g <= limits[i+1])
                per_gal_slab[g] += np.where(sc, str_flux_g, 0.0).sum(axis=2)

            # Overlay diffuse flux within the current velocity bin
            if diffuse_flux is not None:
                if i < n_spectral_slices - 2:
                    diff_cond = (diffuse_vel >= limits[i]) & (diffuse_vel < limits[i+1])
                else:
                    diff_cond = (diffuse_vel >= limits[i]) & (diffuse_vel <= limits[i+1])
                combined_cube[diff_cond] += diffuse_flux[diff_cond]

            # Projecting along the LoS (Z-axis)
            spectral_slice = np.sum(combined_cube, axis=2)
            spectral_cube_S_px.append(spectral_slice)  # Transpose if needed
            for g in range(n_galaxies):
                per_gal_slices[g].append(per_gal_slab[g])

            # Store average velocity of this slice
            average_vel = np.mean([limits[i], limits[i+1]])
            average_vels[i] = average_vel

        spectral_cube_S_px = np.array(spectral_cube_S_px)


        spectral_cube_Jy_px = spectral_cube_S_px

        n_ch_raw = spectral_cube_Jy_px.shape[0]
        n_ch_trim = (n_ch_raw // 5) * 5
        spectral_cube_Jy_px = spectral_cube_Jy_px[:n_ch_trim]
        average_vels = average_vels[:n_ch_trim]
        spectral_cube_Jy_px = spectral_cube_Jy_px.reshape(n_ch_trim // 5, 5, spectral_cube_Jy_px.shape[1], spectral_cube_Jy_px.shape[2]).mean(axis=1)
        average_vels = average_vels.reshape(n_ch_trim // 5, 5).mean(axis=1)

        # Same 5× spectral averaging for each per-galaxy (diffuse-free) cube.
        per_galaxy_cubes = []
        for g in range(n_galaxies):
            arr = np.array(per_gal_slices[g])[:n_ch_trim]
            arr = arr.reshape(n_ch_trim // 5, 5, arr.shape[1], arr.shape[2]).mean(axis=1)
            per_galaxy_cubes.append(arr)
        per_galaxy_cubes = np.stack(per_galaxy_cubes, axis=0)  # (n_gals, n_ch, n_y, n_x)

        # You can update the params_gals dictionary as needed
        params_gen = {
            'galaxy_centers': galaxy_centers,
            'average_vels': average_vels,
            'beam_info': self.beam_info,
            'n_gals': n_galaxies,
            'pix_spatial_scale': pix_spatial_scale,
            'diffuse_emission': bool(use_diffuse),
            'systemic_vels': np.asarray(gal_systemic_vels),
            'per_galaxy_cubes': per_galaxy_cubes,
        }

        return spectral_cube_Jy_px, params_gen



    def generate_cubes(self):
        """Run the full pipeline and generate the requested spectral cubes.

        This is the high-level convenience method that iterates over the
        pre-sampled per-cube parameters (``self.all_Re``, ``self.all_Se`` etc.),
        constructs each component via :meth:`rotated_system`, assembles the
        spectral cube with :meth:`make_spectral_cube`, applies beam
        convolution and light smoothing, optionally saves the cube(s) to disk
        when ``save=True`` (the ``fname`` argument controls the destination
        directory), and returns a list of ``(cube, params)``
        tuples stored in ``self.results``.

        Returns
        -------
        results : list
            A list with one entry per generated cube. Each entry is a tuple
            ``(spectral_cube_array, params_dict)`` where ``spectral_cube_array``
            has shape ``(n_channels, grid_size, grid_size)``.

        Example
        -------
        >>> g = SONGS(n_cubes=1, seed=42, verbose=False)
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

            gal_params_list = [
                {
                    'Re': float(self.all_Re[i][j]),
                    'Se': float(self.all_Se[i][j]),
                    'hz': float(self.all_hz[i][j]),
                    'n': float(self.all_n[i][j]),
                }
                for j in range(self.n_gals[i])
            ]
            spectral_cube_final, params = self.make_spectral_cube(
                rotated_disks, rotated_vel_z_cubes,
                self.all_pix_spatial_scales[i][0],
                gal_params_list=gal_params_list,
            )


            #self.system_params.append(params)

            if self.verbose:
                print('\nSpectral cube created!')


            #Setting possible negative value artifacts to 0
            spectral_cube_final_resolved = np.maximum(spectral_cube_final, 0)

            spectral_cube_final_convolved = convolve_beam(spectral_cube_final, self.beam_info)
            spectral_cube_final_convolved = gaussian_filter1d(spectral_cube_final_convolved, sigma=1.0, axis=0)

            # Apply the same beam + spectral smoothing to each clean per-galaxy cube
            # so they share the effective resolution of the observed cube.
            per_gal_raw = params.get('per_galaxy_cubes')
            if per_gal_raw is not None and len(per_gal_raw) > 0:
                per_gal_convolved = np.stack([
                    gaussian_filter1d(convolve_beam(pg, self.beam_info), sigma=1.0, axis=0)
                    for pg in per_gal_raw
                ], axis=0)
                params['per_galaxy_cubes'] = per_gal_convolved

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
                        
                h5_path = os.path.join(fname_save, 'cube_{}.h5'.format(i + 1))
                _save_cube_hdf5(h5_path, spectral_cube_final_convolved, params, self, i)

                if self.verbose:
                    print('saved as ' + h5_path)


        return self.results
    
    def __len__(self):
        return self.n_cubes

        
class SONGSPhy:
    """Generator for single-system synthetic IFU spectral cubes (physical units).

    High-level behaviour
    --------------------
        - Uses explicit physical-unit parametrisation (e.g. kpc, km s⁻¹) to
            construct highly specific galaxies, typically a single system intended
            for interactive use (GUI). Each component has a 3D Sérsic + exponential
            vertical light distribution and a simple analytical rotation field.
        - Components are rotated to a viewing geometry and placed into a larger
            spatial grid. Voxels are binned by line-of-sight velocity to produce a
            3D spectral cube.

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

    Parameters are provided explicitly (not randomly sampled) to enable
    controlled, reproducible experiments. The implementation prioritises
    clarity and inspectability over performance.
    """
    
    def __init__(self, n_gals=None, n_cubes=1, spatial_resolution=4.5, spectral_resolution=10, offset_gals=20, beam_info = [18,18,0], fov=125, n_sersic=None, save=False, fname=None, verbose=True, seed=None, diffuse_params=None, sat_brightness_frac=None, diffuse_flux_frac=1.0, sat_vel_dispersion=150.0):
        """
        Initialize the physically parameterised generator (explicit units).

        Parameters
        ----------
        n_gals : int or None
            Fixed number of galaxies per cube. If ``None``, a small number
            (1–3) is used for variety.
        n_cubes : int
            Number of cubes to generate.
        spatial_resolution : float
            Spatial pixel scale in kpc per pixel (kpc/px). Controls the
            conversion between physical distances and image pixels.
        spectral_resolution : float
            Spectral channel width in km s⁻¹. Internally, a 5× oversampled
            grid is used and averaged to emulate binning.
        offset_gals : float
            Typical spatial offset between secondary galaxies and the primary
            in kpc. Converted to pixels via ``spatial_resolution``.
        beam_info : sequence of float
            Telescope beam description in physical units [bmin_kpc, bmaj_kpc,
            bpa_deg]. Internally converted to pixel units using
            ``spatial_resolution``.
        fov : int
            Field of view in kpc. Converted to image size via
            ``grid_size = int(fov / spatial_resolution)``.
        n_sersic : float or None
            Optional fixed Sérsic index for all galaxies; when ``None``,
            reasonable defaults are used.
        save : bool
            When ``True``, save generated cubes to disk; otherwise keep
            results in memory.
        fname : str or None
            Destination directory if saving; created if missing.
        verbose : bool
            Print progress messages during generation.
        seed : int or None
            RNG seed to make results reproducible across NumPy, PyTorch
            and Python ``random``.

        Notes
        -----
        This mode is designed for interactive, single-system construction via
        the GUI. Parameters are specified in explicit physical units to enable
        controlled, reproducible systems.
        """

        # Initialize random seeds for reproducible results
        #self.central_Re_kpc = 5 #kpc

        # Store configuration parameters
        self.fname = fname
        self.seed = seed
        self.save = save
        if self.seed is not None:
            # Set all random number generators for consistency
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        # Galaxy separation parameter (affects interaction dynamics)
        # Convert kpc → pixels using spatial_resolution
        if isinstance(offset_gals, (tuple, list)):
            self.offset_gals = (offset_gals[0] / spatial_resolution,
                                offset_gals[1] / spatial_resolution)
        else:
            self.offset_gals = offset_gals / spatial_resolution

        # Satellite brightness fraction (total-flux fraction vs central).
        self.sat_brightness_frac = sat_brightness_frac
        self.sat_vel_dispersion = float(sat_vel_dispersion)

        # Diffuse-emission configuration (halo, bridges, tidal tails)
        merged_dp = dict(DEFAULT_DIFFUSE_PARAMS)
        if diffuse_params:
            merged_dp.update(diffuse_params)
        if diffuse_flux_frac != 1.0:
            for _key in ('halo_Se_factor', 'bridge_Se_factor', 'tail_Se_factor'):
                if _key in merged_dp:
                    merged_dp[_key] = merged_dp[_key] * diffuse_flux_frac
        self.diffuse_params = merged_dp

        # Determine number of galaxies per cube
        if n_gals is None:
            # Default: 1–2 galaxies per cube (legacy behaviour).
            self.n_gals = np.random.randint(1, 3, n_cubes)
        elif isinstance(n_gals, (tuple, list)) and len(n_gals) == 2:
            # Random inclusive range per cube: n_gals=(lo, hi) → uniform in [lo, hi].
            lo, hi = int(n_gals[0]), int(n_gals[1])
            self.n_gals = np.random.randint(lo, hi + 1, n_cubes)
        else:
            # Fixed number of galaxies across all cubes
            self.n_gals = [int(n_gals) for _ in range(n_cubes)]

        # Grid and observational parameters
        self.n_cubes = n_cubes

        # Image size from physical field-of-view: kpc → pixels
        grid_size = int(fov/spatial_resolution)
        init_grid_size = (31/64)*grid_size
        if int(init_grid_size)%2!=0:
            self.init_grid_size = int(init_grid_size)-1
        else:
            self.init_grid_size = int(init_grid_size)

        self.grid_size = grid_size      # Size for combined output cube
        #self.n_spectral_slices = 5*n_spectral_slices + 1  # 5x oversampling + 1 for binning
        self.spectral_resolution = spectral_resolution/5
        self.spatial_resolution = spatial_resolution
        # Convert beam from physical units (kpc, deg) → pixel units (px, deg)
        self.beam_info = [beam_info[0]/spatial_resolution, beam_info[1]/spatial_resolution, beam_info[2]]            # [bmin_px, bmaj_px, bpa]
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
        
        '''if isinstance(self.resolution, float) and n_cubes==1:
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
        Re_central = r * (self.beam_info[0]/self.spatial_resolution) /2'''

        # =======================================================================
        # GALAXY PARAMETER GENERATION  
        # =======================================================================
        # Generate physical parameters for each galaxy system
        
        # Fixed effective radius in physical units (could be varied)
        central_Re_kpc = np.random.uniform(5, 5, n_cubes)  # Central Re in kpc
        Re_central = central_Re_kpc/spatial_resolution

        r = [5/((self.beam_info[0]*spatial_resolution)/2)]

        # Generate parameters for each cube
        for i in range(n_cubes):

            # Calculate pixel scale: kpc per pixel
            pix_spatial_scale = central_Re_kpc[i] / Re_central[i]  # Scale in pixels relative to Re

            # Primary galaxy parameters
            Re = [Re_central[i]]                                      # Effective radius in pixels
            hz = [np.random.uniform(0.4, 9) / pix_spatial_scale]      # Scale height (thinner in high-res)
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
                # Satellites are smaller than the primary
                Re += list(np.random.uniform(Re[0]/3, Re[0]/2, n_gal - 1))
                hz += list(np.random.uniform(hz[0]/3, hz[0]/2, n_gal - 1))

                # Sample satellite Sérsic indices first so we can compute bn
                # and normalise to peak surface brightness (Se*exp(bn)) rather
                # than Se alone.  This ensures a high-n satellite (steep cusp)
                # never appears brighter than the central regardless of index.
                sat_ns = list(np.random.uniform(0.5, 1.5, n_gal - 1))
                n_sersics += sat_ns

                n_c = float(n_sersics[0])
                bn_c = 2*n_c - 1/3 + 4/(405*n_c) + 46/(25515*n_c**2)
                peak_c = Se[0] * np.exp(bn_c)

                _b = float(np.clip(self.sat_brightness_frac, 1e-6, 1.0 - 1e-6)) \
                    if self.sat_brightness_frac is not None \
                    else None
                for _k in range(n_gal - 1):
                    n_s = float(sat_ns[_k])
                    bn_s = 2*n_s - 1/3 + 4/(405*n_s) + 46/(25515*n_s**2)
                    frac = _b if _b is not None else np.random.uniform(0.15, 0.35)
                    # Se_sat chosen so peak_sat = frac * peak_c, guaranteed < peak_c
                    Se_sat = frac * peak_c / np.exp(bn_s) * np.random.uniform(0.85, 1.0)
                    Se.append(float(Se_sat))

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
        >>> SONGS.milky_way_rot_curve_analytical(np.array([0.1,1,10]), 200, 5.0, 1.0)
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
        - The vertical structure is a symmetric exponential: :math:`\exp(-|z|/h_z)`.

        Example
        -------
        >>> nx = ny = nz = 21
        >>> x = np.arange(nx) - (nx-1)/2
        >>> X,Y,Z = np.meshgrid(x,x,x,indexing='ij')
        >>> S = SONGS.sersic_flux_density_3d(X, Y, Z, Se=0.1, Re=5.0, n=1.0, hz=0.5)
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
        """Create a rotated 3D galaxy flux cube and LOS velocity cube.

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
            3D line-of-sight velocity cube (same shape) giving LOS velocity
            (km/s) at each voxel after rotation and projection.

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


    def make_spectral_cube(self, rotated_disks, rotated_vel_z_cubes, pix_spatial_scale, gal_params_list=None):
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
            Metadata with keys: 'galaxy_centers', 'average_vels', 'beam_info',
            'n_gals', and 'pix_spatial_scale'.

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
        spectral_resolution = self.spectral_resolution
        n_galaxies = len(rotated_disks)
        assert n_galaxies == len(rotated_vel_z_cubes), "Mismatch between disks and velocity cubes"

        # Re-seed at placement time so the galaxy-property sampling in __init__
        # does not consume the seed before placement, giving truly random positions.
        if self.seed is not None:
            np.random.seed(self.seed + 1)

        center_final_cube = np.array([(grid_size + 1) / 2] * 3)

        galaxy_centers = []

        half_size = init_grid_size // 2
        min_pos = half_size
        max_pos = grid_size - half_size

        # First galaxy near the center
        x = int(np.clip(center_final_cube[0], min_pos, max_pos - 1))
        y = int(np.clip(center_final_cube[1], min_pos, max_pos - 1))
        z = int(np.clip(center_final_cube[2], min_pos, max_pos - 1))
        galaxy_centers.append(np.array([x, y, z]))

        # Minimum 3-D separation: just large enough that galaxy disks don't overlap.
        _min_sep_3d = init_grid_size // 2 + 4

        # _off_min floored to _min_sep_3d so satellites are always placed far
        # enough from the central to be visually distinct.
        if isinstance(self.offset_gals, (tuple, list)):
            _off_min = max(_min_sep_3d, int(round(self.offset_gals[0])))
            _off_max = max(_off_min, int(round(self.offset_gals[1])))
        else:
            _off_min = _min_sep_3d
            _off_max = max(_off_min, int(round(self.offset_gals)))

        # Additional galaxies nearby but offset, with overlap rejection.
        _grid_max = max_pos - min_pos - 1
        for i in range(1, n_galaxies):
            placed = False
            best_cand = None
            best_min_dist = -1.0
            for _expand in range(8):
                lo = min(_off_min, _grid_max)
                hi = min(_off_max + _expand * max(2, _off_max // 4), _grid_max)
                hi = max(hi, lo + 1)
                for _attempt in range(400):
                    dx = int(np.random.randint(lo, hi + 1)) * np.random.choice([-1, 1])
                    dy = int(np.random.randint(lo, hi + 1)) * np.random.choice([-1, 1])
                    dz = int(np.random.randint(lo, hi + 1)) * np.random.choice([-1, 1])
                    cx = galaxy_centers[0][0] + dx
                    cy = galaxy_centers[0][1] + dy
                    cz = galaxy_centers[0][2] + dz
                    # Skip out-of-bounds rather than clipping — clipping causes
                    # all candidates to snap to the same boundary corners.
                    if not (min_pos <= cx < max_pos and
                            min_pos <= cy < max_pos and
                            min_pos <= cz < max_pos):
                        continue
                    cand = np.array([cx, cy, cz], dtype=float)
                    min_dist = min(np.linalg.norm(cand - np.asarray(c, dtype=float))
                                   for c in galaxy_centers)
                    # Also enforce 2-D (x,y) separation — the cube is projected
                    # along Z so galaxies at the same sky position always overlap.
                    min_dist_2d = min(
                        np.linalg.norm(cand[:2] - np.asarray(c, dtype=float)[:2])
                        for c in galaxy_centers)
                    # Track best candidate by combined 3D+2D score so the
                    # fallback also avoids sky projection overlaps.
                    score = min(min_dist, min_dist_2d)
                    if score > best_min_dist:
                        best_min_dist = score
                        best_cand = cand
                    if min_dist >= _off_min and min_dist_2d >= _off_min:
                        placed = True
                        break
                if placed:
                    break
            if best_cand is None:
                # All offset-constrained attempts failed — scan in-bounds positions
                # and pick the one maximising min(3D dist, 2D sky dist).
                candidates = []
                for _ in range(1000):
                    cx = int(np.random.randint(min_pos, max_pos))
                    cy = int(np.random.randint(min_pos, max_pos))
                    cz = int(np.random.randint(min_pos, max_pos))
                    cand = np.array([cx, cy, cz], dtype=float)
                    d3 = min(np.linalg.norm(cand - np.asarray(c, dtype=float))
                             for c in galaxy_centers)
                    d2 = min(np.linalg.norm(cand[:2] - np.asarray(c, dtype=float)[:2])
                             for c in galaxy_centers)
                    candidates.append((min(d3, d2), cand))
                best_cand = max(candidates, key=lambda t: t[0])[1]
            galaxy_centers.append(best_cand.astype(int))

        if self.verbose:
            for idx, center in enumerate(galaxy_centers):
                print(f"Centre of galaxy {idx + 1}: {center}")

        # Apply Hubble flow + peculiar velocity relative to the first galaxy
        reference_z = galaxy_centers[0][2]
        H_z = cosmo.H(0).value  # km/s/Mpc

        for i in range(1, n_galaxies):
            delta_z_kpc = (galaxy_centers[i][2] - reference_z)*pix_spatial_scale
            delta_z_mpc = delta_z_kpc * 1e-3  # Convert kpc to Mpc
            relative_velocity = H_z * delta_z_mpc

            # Peculiar velocity: random blueshift/redshift for each satellite
            peculiar_vel = np.random.normal(0.0, self.sat_vel_dispersion)
            relative_velocity += peculiar_vel

            if self.verbose:
                direction = "farther" if delta_z_kpc > 0 else "closer"
                print(f"Galaxy {i+1} is {direction} than galaxy 1 by {delta_z_kpc:.2f} kpc")
                print(f"→ Hubble + peculiar velocity offset: {relative_velocity:.2f} km/s  (peculiar: {peculiar_vel:+.1f} km/s)")

            # Add velocity offset to simulate redshift/blueshift
            rotated_vel_z_cubes[i] = (rotated_vel_z_cubes[i]+relative_velocity)


        # Flux-weighted systemic LOS velocity per galaxy (includes Hubble offset
        # already applied above). Useful as ground-truth for ML detection.
        gal_systemic_vels = []
        for disk, vel_cube in zip(rotated_disks, rotated_vel_z_cubes):
            total = float(np.sum(disk))
            if total > 0:
                gal_systemic_vels.append(float(np.sum(disk * vel_cube) / total))
            else:
                gal_systemic_vels.append(0.0)

        # Build diffuse emission (halo + bridges + tidal tails) on the full grid
        use_diffuse = (gal_params_list is not None
                       and getattr(self, 'diffuse_params', {}).get('enabled', False))
        if use_diffuse:
            if self.verbose:
                print('Building diffuse emission (halo, bridges, streamers)...')
            diffuse_flux, diffuse_vel, streamer_per_galaxy = _build_diffuse_cubes(
                grid_size, galaxy_centers, gal_params_list,
                gal_systemic_vels, self.diffuse_params,
            )
        else:
            diffuse_flux = None
            diffuse_vel = None
            streamer_per_galaxy = [None] * n_galaxies


        # Creating lower and upper limits for the velocity observation bins
        # Create velocity bin edges across all galaxies

        min_vel = -600 #np.min([np.min(v) for v in all_velocities])
        max_vel = 600 #np.max([np.max(v) for v in all_velocities])

        limit = np.max([abs(min_vel), abs(max_vel)])  # Use the maximum absolute value for limits

        limits = np.arange(-limit, limit+spectral_resolution, spectral_resolution)
    
        if self.verbose:
            print('Overlaying all galaxy observations in a bigger spatial grid')
            print('Calculating the projected flux density of every voxel within the limits in each velocity slice')

        spectral_cube_S_px = []
        # Per-galaxy (diffuse-free) spectral slices in the final FOV.
        per_gal_slices = [[] for _ in range(n_galaxies)]
        average_vels = np.zeros((len(limits) - 1))


        for i in range(len(limits) - 1):
            combined_cube = np.zeros((grid_size, grid_size, grid_size))
            per_gal_slab = [np.zeros((grid_size, grid_size)) for _ in range(n_galaxies)]
            for g, (disk, vel_cube, center) in enumerate(zip(rotated_disks, rotated_vel_z_cubes, galaxy_centers)):

                # Determine the voxels within current velocity bin
                if i < len(limits) - 2:
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
                # Clean per-galaxy contribution (LOS-projected, diffuse-free
                # other than the streamer that belongs to this satellite).
                per_gal_slab[g][xs:xe, ys:ye] += selected_cube.sum(axis=2)

            # Route each satellite's streamer flux for this channel into the
            # per-galaxy clean cube (full FOV, not the init-grid slab).
            for g in range(n_galaxies):
                sg = streamer_per_galaxy[g]
                if sg is None:
                    continue
                str_flux_g, str_vel_g = sg
                if i < len(limits) - 2:
                    sc = (str_vel_g >= limits[i]) & (str_vel_g < limits[i+1])
                else:
                    sc = (str_vel_g >= limits[i]) & (str_vel_g <= limits[i+1])
                per_gal_slab[g] += np.where(sc, str_flux_g, 0.0).sum(axis=2)

            # Overlay diffuse flux within the current velocity bin
            if diffuse_flux is not None:
                if i < len(limits) - 2:
                    diff_cond = (diffuse_vel >= limits[i]) & (diffuse_vel < limits[i+1])
                else:
                    diff_cond = (diffuse_vel >= limits[i]) & (diffuse_vel <= limits[i+1])
                combined_cube[diff_cond] += diffuse_flux[diff_cond]

            # Projecting along the LoS (Z-axis)
            spectral_slice = np.sum(combined_cube, axis=2)
            spectral_cube_S_px.append(spectral_slice)  # Transpose if needed
            for g in range(n_galaxies):
                per_gal_slices[g].append(per_gal_slab[g])

            # Store average velocity of this slice
            average_vel = np.mean([limits[i], limits[i+1]])
            average_vels[i] = average_vel

        spectral_cube_S_px = np.array(spectral_cube_S_px)


        spectral_cube_Jy_px = spectral_cube_S_px

        n_ch_raw = spectral_cube_Jy_px.shape[0]
        n_ch_trim = (n_ch_raw // 5) * 5
        spectral_cube_Jy_px = spectral_cube_Jy_px[:n_ch_trim]
        average_vels = average_vels[:n_ch_trim]
        spectral_cube_Jy_px = spectral_cube_Jy_px.reshape(n_ch_trim // 5, 5, spectral_cube_Jy_px.shape[1], spectral_cube_Jy_px.shape[2]).mean(axis=1)
        average_vels = average_vels.reshape(n_ch_trim // 5, 5).mean(axis=1)

        # Same 5× spectral averaging for each per-galaxy (diffuse-free) cube.
        per_galaxy_cubes = []
        for g in range(n_galaxies):
            arr = np.array(per_gal_slices[g])[:n_ch_trim]
            arr = arr.reshape(n_ch_trim // 5, 5, arr.shape[1], arr.shape[2]).mean(axis=1)
            per_galaxy_cubes.append(arr)
        per_galaxy_cubes = np.stack(per_galaxy_cubes, axis=0)  # (n_gals, n_ch, n_y, n_x)

        # You can update the params_gals dictionary as needed
        params_gen = {
            'galaxy_centers': galaxy_centers,
            'average_vels': average_vels,
            'beam_info': self.beam_info,
            'n_gals': n_galaxies,
            'pix_spatial_scale': pix_spatial_scale,
            'diffuse_emission': bool(use_diffuse),
            'systemic_vels': np.asarray(gal_systemic_vels),
            'per_galaxy_cubes': per_galaxy_cubes,
        }

        return spectral_cube_Jy_px, params_gen



    def generate_cubes(self):
        """Run the full pipeline and generate the requested spectral cubes.

        This is the high-level convenience method that iterates over the
        pre-sampled per-cube parameters (``self.all_Re``, ``self.all_Se`` etc.),
        constructs each component via :meth:`rotated_system`, assembles the
        spectral cube with :meth:`make_spectral_cube`, applies beam
        convolution and light smoothing, optionally saves the cube(s) to disk
        when ``save=True`` (the ``fname`` argument controls the destination
        directory), and returns a list of ``(cube, params)``
        tuples stored in ``self.results``.

        Returns
        -------
        results : list
                A list with one entry per generated cube. Each entry is a tuple
                ``(spectral_cube_array, params_dict)`` where ``spectral_cube_array``
                has shape ``(n_channels, grid_size, grid_size)``.

        Example
        -------
        >>> g = SONGS(n_cubes=1, seed=42, verbose=False)
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

            gal_params_list = [
                {
                    'Re': float(self.all_Re[i][j]),
                    'Se': float(self.all_Se[i][j]),
                    'hz': float(self.all_hz[i][j]),
                    'n': float(self.all_n[i][j]),
                }
                for j in range(self.n_gals[i])
            ]
            spectral_cube_final, params = self.make_spectral_cube(
                rotated_disks, rotated_vel_z_cubes,
                self.all_pix_spatial_scales[i][0],
                gal_params_list=gal_params_list,
            )


            #self.system_params.append(params)

            if self.verbose:
                print('\nSpectral cube created!')


            #Setting possible negative value artifacts to 0
            spectral_cube_final_resolved = np.maximum(spectral_cube_final, 0)

            spectral_cube_final_convolved = convolve_beam(spectral_cube_final, self.beam_info)
            spectral_cube_final_convolved = gaussian_filter1d(spectral_cube_final_convolved, sigma=1.0, axis=0)

            # Apply the same beam + spectral smoothing to each clean per-galaxy cube
            # so they share the effective resolution of the observed cube.
            per_gal_raw = params.get('per_galaxy_cubes')
            if per_gal_raw is not None and len(per_gal_raw) > 0:
                per_gal_convolved = np.stack([
                    gaussian_filter1d(convolve_beam(pg, self.beam_info), sigma=1.0, axis=0)
                    for pg in per_gal_raw
                ], axis=0)
                params['per_galaxy_cubes'] = per_gal_convolved

            #self.spectral_cubes.append(spectral_cube_final)

            self.results.append((spectral_cube_final_convolved, params))

            if self.save:
                if self.fname is None:
                    # Use current working directory + /data/raw_data/ as default
                    base_dir = os.path.join(os.getcwd(), 'data', 'raw_data')
                    fname_save = os.path.join(base_dir, '{}x{}x{}'.format(spectral_cube_final_convolved.shape[0], self.grid_size, self.grid_size))
                    if not os.path.exists(fname_save):
                        os.makedirs(fname_save)
                else:
                    fname_save = self.fname
                    if not os.path.exists(fname_save):
                        os.makedirs(fname_save)
                        
                h5_path = os.path.join(fname_save, 'cube_{}.h5'.format(i + 1))
                _save_cube_hdf5(h5_path, spectral_cube_final_convolved, params, self, i)

                if self.verbose:
                    print('saved as ' + h5_path)


        return self.results
    
    def __len__(self):
        return self.n_cubes



