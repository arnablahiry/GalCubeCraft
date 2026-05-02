# Diffuse emission model

`GalCubeCraft` adds three additive diffuse components on top of the rotated
per-galaxy disks produced by `rotated_system`:

1. A **stellar/gaseous halo** around the central galaxy.
2. One **bridge** per satellite, connecting the halo to the satellite.
3. One **tidal tail** per satellite, extending beyond the satellite away from
   the central.

All three are built in a single pass by `_build_diffuse_cubes` (in
[`core.py`](../src/GalCubeCraft/core.py)) on the full output grid. The
diffuse flux cube and its LOS-velocity cube are then binned into the spectral
channels alongside the per-galaxy disks.

Notation:

- $\mathbf{c}_0 = (x_0, y_0, z_0)$ — central galaxy position (pixels).
- $\mathbf{c}_i$ — $i$-th satellite position (pixels).
- $R_{e,c}, S_{e,c}, h_{z,c}$ — central galaxy disk parameters (pixels / flux).
- $R_{e,s}, S_{e,s}, h_{z,s}$ — satellite disk parameters.
- $v_c, v_s$ — flux-weighted mean LOS velocity of central / satellite
  (km s⁻¹; includes any Hubble offset that was applied).
- $\mathcal{N}(0,\sigma)$ — Gaussian random noise per voxel.
- All factors (`halo_Re_factor`, …) are keys of `DEFAULT_DIFFUSE_PARAMS`.

---

## 1. Halo (central galaxy)

3D Sérsic in the disk plane times an exponential vertical profile, centred on
$\mathbf{c}_0$:

$$
S_\text{halo}(\mathbf{r}) \;=\; S_{e,\text{halo}} \,
  \exp\!\left[-b_n\!\left(\left(\frac{R}{R_{e,\text{halo}}}\right)^{1/n_\text{halo}} - 1\right)\right]
  \exp\!\left(-\frac{|z - z_0|}{h_{z,\text{halo}}}\right),
$$

with

- $R = \sqrt{(x-x_0)^2 + (y-y_0)^2}$,
- $b_n \approx 2n - \tfrac{1}{3} + \tfrac{4}{405 n} + \tfrac{46}{25515 n^2}$
  (series for the Sérsic constant; evaluated at $n_\text{halo}$),
- $R_{e,\text{halo}} = \texttt{halo\_Re\_factor}\cdot R_{e,c}$,
- $h_{z,\text{halo}} = \texttt{halo\_hz\_factor}\cdot h_{z,c}$,
- $S_{e,\text{halo}} = \texttt{halo\_Se\_factor}\cdot S_{e,c}$,
- $n_\text{halo} = \texttt{halo\_n}$ (default 0.5 — Gaussian-like).

LOS velocity (pressure-supported halo, so no coherent rotation):

$$
v_\text{halo}(\mathbf{r}) \;=\; v_c + \mathcal{N}(0,\texttt{halo\_sigma\_vz}).
$$

---

## 2. Bridges (central → each satellite)

For satellite $i$, let
$\mathbf{d} = \mathbf{c}_i - \mathbf{c}_0$, $\text{sep}=\|\mathbf{d}\|$,
$\hat{\mathbf{d}} = \mathbf{d} / \text{sep}$. Parametrise every voxel
$\mathbf{r}$ by its signed fractional position along the link:

$$
s(\mathbf{r}) \;=\; \frac{(\mathbf{r} - \mathbf{c}_0)\cdot \hat{\mathbf{d}}}{\text{sep}}.
$$

The bridge lives on the interval $s \in [s_\text{start}, s_\text{end}]$ with

$$
s_\text{start} = \texttt{bridge\_start\_frac},\qquad
s_\text{end} = 1 - \texttt{bridge\_stop\_frac}.
$$

Perpendicular distance from the bridge axis (with $s$ clamped to the active
interval so the endpoints stay well-defined):

$$
s^\ast = \operatorname{clip}(s, s_\text{start}, s_\text{end}),\qquad
p(\mathbf{r}) = \| \mathbf{r} - (\mathbf{c}_0 + s^\ast\,\mathbf{d}) \|.
$$

Width tapers linearly from the halo end to the satellite end:

$$
u = \operatorname{clip}\!\left(\frac{s - s_\text{start}}{s_\text{end} - s_\text{start}},\,0,\,1\right),
\quad
\sigma(s) = (1-u)\,\sigma_\text{start} + u\,\sigma_\text{end},
$$

with

$$
\sigma_\text{start} = \texttt{bridge\_width\_start\_factor}\cdot R_{e,c},\qquad
\sigma_\text{end}   = \texttt{bridge\_width\_end\_factor}\cdot R_{e,s}.
$$

Trapezoidal window $w(s)$ with smooth fade of width $f=\texttt{bridge\_edge\_fade}$,
exactly zero outside $[s_\text{start}, s_\text{end}]$:

$$
w(s) \;=\; \min\!\Bigl(
  \operatorname{clip}\!\left(\tfrac{s - s_\text{start}}{f},0,1\right),\;
  \operatorname{clip}\!\left(\tfrac{s_\text{end} - s}{f},0,1\right)
\Bigr).
$$

Flux:

$$
S_{\text{bridge},i}(\mathbf{r})
= S_{e,\text{br}}\, \exp\!\left(-\tfrac{1}{2}\,\tfrac{p(\mathbf{r})^2}{\sigma(s)^2}\right) \, w(s),
$$

with $S_{e,\text{br}} = \texttt{bridge\_Se\_factor}\cdot \min(S_{e,c}, S_{e,s})$.

LOS velocity (linear interpolation between endpoints):

$$
v_{\text{bridge},i}(\mathbf{r}) = (1 - s^\ast)\,v_c + s^\ast\,v_s + \mathcal{N}(0,\texttt{bridge\_sigma\_vz}).
$$

---

## 3. Tidal tails

For each satellite $i$, pick a unit vector $\hat{\mathbf{p}} \perp \hat{\mathbf{d}}$
by Gram–Schmidt against a random draw. The tail is a quadratic curve:

$$
\mathbf{P}(u) = \mathbf{c}_i + u\,L_\text{tail}\,\hat{\mathbf{d}} + \kappa\,u^2\,\hat{\mathbf{p}},
\quad u \in [0, 1],
$$

with

$$
L_\text{tail} = \texttt{tail\_length\_factor}\cdot\text{sep},\qquad
\kappa = \texttt{tail\_curvature}\cdot\text{sep}.
$$

The tail flux is built by superposing Gaussian blobs at $N$ sampled points
$u_k$ along the curve:

$$
S_\text{tail}(\mathbf{r}) =
\sum_{k=1}^{N} S_{e,\text{tail}}\,(1 - u_k)\,
  \exp\!\left(-\tfrac{1}{2}\,\tfrac{\|\mathbf{r} - \mathbf{P}(u_k)\|^2}{\sigma_\text{tail}^2}\right)\,
  \Delta_N\,\mathcal{G}(\mathbf{r}),
$$

with

- $\sigma_\text{tail} = \texttt{tail\_width\_factor}\cdot R_{e,s}$,
- $S_{e,\text{tail}} = \texttt{tail\_Se\_factor}\cdot S_{e,s}$ (tail amplitude fades linearly with $u$),
- $\Delta_N = 5/N$ — normalisation keeping the integrated flux roughly independent of $N$,
- $\mathcal{G}(\mathbf{r})$ — a sigmoid gate that suppresses emission on the
  wrong side of the satellite, keeping the tail one-sided:

$$
\mathcal{G}(\mathbf{r}) =
  \sigma_\text{sig}\!\left(\frac{(\mathbf{r} - \mathbf{c}_i)\cdot\hat{\mathbf{d}}}{0.5\,\sigma_\text{tail}}\right),
\qquad
\sigma_\text{sig}(t) = \tfrac{1}{1 + e^{-t}}.
$$

LOS velocity (drift along the tail):

$$
v_\text{tail}(u_k) = v_s + \texttt{tail\_vel\_gradient}\cdot u_k + \mathcal{N}(0,\texttt{tail\_sigma\_vz}).
$$

---

## 4. Combining components

Total diffuse flux and flux-weighted LOS velocity:

$$
S_\text{diff}(\mathbf{r}) = S_\text{halo}(\mathbf{r}) + \sum_i\bigl(S_{\text{bridge},i}(\mathbf{r}) + S_{\text{tail},i}(\mathbf{r})\bigr),
$$

$$
v_\text{diff}(\mathbf{r}) = \frac{
  v_\text{halo}\,S_\text{halo} + \sum_i \bigl(v_{\text{bridge},i}\,S_{\text{bridge},i} + v_{\text{tail},i}\,S_{\text{tail},i}\bigr)
}{S_\text{diff}(\mathbf{r})}.
$$

The diffuse cube is then binned into spectral channels using exactly the same
velocity-bin masks as the per-galaxy disks, so the halo / bridges / tails
appear in the appropriate channels of the final $(n_\text{vel},n_y,n_x)$ cube
after LOS integration and beam convolution.
