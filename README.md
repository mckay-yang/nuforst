# NUFROST

NUFROST is a Rust-first research implementation for reconstructing optical remote-sensing time series from irregular satellite observations. The active method is a vector-valued, non-uniform Fourier reconstruction model: it first estimates a shared spectral support from non-uniform observation times, then fits a stable multi-band trajectory with a date-level robust multi-output ridge model.

This README documents the current NUFROST mathematical model only.

## Problem

For one pixel, assume there are $B$ optical bands observed at irregular timestamps

$$
t_1,t_2,\dots,t_T.
$$

At time $t_i$, the multi-band observation is treated as a vector

$$
\mathbf y_i =
\begin{bmatrix}
y_{i,1} & y_{i,2} & \cdots & y_{i,B}
\end{bmatrix}
\in \mathbb R^B.
$$

The pixel trajectory is therefore a vector-valued curve

$$
\mathbf y(t): \mathbb R \rightarrow \mathbb R^B.
$$

The reconstruction task is to predict the missing or held-out vector value

$$
\hat{\mathbf y}(t_\star)
$$

at a target time $t_\star$, while preserving temporal stability and cross-band consistency.

In the Sentinel-2 full-scene workflow, the usual band order is:

$$
B2,\ B3,\ B4,\ B8,\ B11,\ B12.
$$

So $B=6$ for the current multi-band NUFROST path.

## Overview

NUFROST has two main stages:

1. **Vector NUFFT frequency discovery**

   Estimate a shared frequency set from the irregular multi-band time series.

2. **Vector Huber-IRLS multi-output Ridge fitting**

   Fit all bands together with one design matrix and one date-level robust weight sequence.

The current active multi-band model can be summarized as:

$$
\boxed{
\mathbf y(t)
\approx
\boldsymbol\mu
+
\mathbf s \odot
\left(
\mathbf x(t)^\top \Theta
\right)
}
$$

where:

- $\boldsymbol\mu\in\mathbb R^B$ is a per-band robust center.
- $\mathbf s\in\mathbb R^B$ is a per-band robust scale.
- $\mathbf x(t)\in\mathbb R^P$ is the harmonic/trend basis vector.
- $\Theta\in\mathbb R^{P\times B}$ is the multi-output coefficient matrix.
- $\odot$ denotes element-wise multiplication.

## Robust Standardization

Before spectral estimation and fitting, each band is robustly standardized. For band $b$:

$$
\mu_b = \operatorname{median}_i(y_{i,b})
$$

and

$$
s_b =
1.4826\,
\operatorname{median}_i
\left(
\left|y_{i,b}-\mu_b\right|
\right).
$$

The standardized observation matrix is:

$$
Z_{i,b}
=
\frac{y_{i,b}-\mu_b}{s_b+\epsilon}.
$$

Equivalently:

$$
Z\in\mathbb R^{T\times B}.
$$

This normalization is important because optical bands have different reflectance ranges. The robust Huber threshold is then applied in a common standardized space rather than raw DN space.

If MAD is degenerate for a band, the implementation falls back to standard deviation. A small positive floor prevents division by zero.

## Vector NUFFT

The observation times are generally non-uniform because clouds, snow, shadows, orbit geometry, and quality filters remove many dates. NUFROST therefore avoids assuming a uniform temporal grid.

For each standardized band trajectory $z_b(t_i)$, the type-1 non-uniform Fourier transform is:

$$
F_b(f_k)
=
\sum_{i=1}^{T}
z_b(t_i)
\exp(-\mathrm{i}\,2\pi f_k t_i).
$$

The current implementation computes this as a vector-valued NUFFT on the shared timestamp grid:

$$
\mathbf F(f_k)
=
\begin{bmatrix}
F_1(f_k)\\
F_2(f_k)\\
\vdots\\
F_B(f_k)
\end{bmatrix}
\in\mathbb C^B.
$$

All bands share:

- the same non-uniform time coordinates,
- the same Kaiser-Bessel spreading positions,
- the same oversampled FFT grid,
- the same kernel deconvolution,
- the same output frequency bins.

For frequency selection, NUFROST compresses the vector spectrum into a joint power spectrum:

$$
P(f_k)
=
\left\|\mathbf F(f_k)\right\|_2^2
=
\sum_{b=1}^{B}
\left|F_b(f_k)\right|^2.
$$

Thus the frequency discovery step is vector-valued in computation, while the selected support is shared across all bands.

## Frequency Selection

NUFROST selects a compact set of harmonic frequencies

$$
\mathcal F
=
\{f_1,\dots,f_m\}
$$

from the joint power spectrum $P(f)$.

The selection process uses the configured mode:

- `spectral`: use dominant peaks from the estimated spectrum.
- `preferred`: use configured physically meaningful periods.
- `shared_spectral` / `hybrid`: combine preferred periods with spectral peaks.

Preferred periods are expressed in days. For example:

$$
365.25,\quad 182.625,\quad 91.3125,\quad 30.4375
$$

correspond to annual, semiannual, quarterly, and monthly-like components:

$$
f = \frac{1}{p\cdot 86400}.
$$

When peak refinement is enabled, selected peaks are locally refined by fitting a parabola around neighboring spectral bins. Nearby selected frequencies are merged using a relative tolerance.

The result is one shared frequency set:

$$
\mathcal F
$$

used by all bands.

## Design Matrix

Given selected frequencies $\mathcal F$, NUFROST builds a harmonic design matrix:

$$
X\in\mathbb R^{T\times P}.
$$

With DC and trend enabled, each row is:

$$
\mathbf x(t_i)
=
\begin{bmatrix}
1,\,
t_i-\bar t,\,
\cos(2\pi f_1 t_i),\,
\sin(2\pi f_1 t_i),\,
\dots,\,
\cos(2\pi f_m t_i),\,
\sin(2\pi f_m t_i)
\end{bmatrix}.
$$

The number of columns is:

$$
P = 1 + \mathbf 1_{\text{trend}} + 2m.
$$

This matrix is shared by all bands.

## Vector Huber-IRLS Multi-Output Ridge

After frequency discovery, NUFROST fits the standardized multi-band trajectory:

$$
Z \approx X\Theta,
$$

where:

$$
Z\in\mathbb R^{T\times B},
\quad
X\in\mathbb R^{T\times P},
\quad
\Theta\in\mathbb R^{P\times B}.
$$

The current objective is:

$$
\boxed{
\min_{\Theta}
\frac12
\left\|
W^{1/2}
(Z-X\Theta)
\right\|_F^2
+
\frac12
\left\|
\Lambda^{1/2}\Theta
\right\|_F^2
}
$$

where:

- $W=\operatorname{diag}(w_1,\dots,w_T)$ is a date-level robust weight matrix.
- $\Lambda\in\mathbb R^{P\times P}$ is a diagonal frequency-weighted ridge penalty.
- $\|\cdot\|_F$ is the Frobenius norm.

This is a multi-output regression problem: all bands are solved together.

For fixed weights $W$, the normal equation is:

$$
\left(
X^\top W X + \Lambda
\right)
\Theta
=
X^\top W Z.
$$

Let

$$
A = X^\top W X + \Lambda,
\quad
C = X^\top W Z.
$$

Then:

$$
A\Theta=C.
$$

The implementation factorizes $A$ once and solves all $B$ right-hand sides together. This is the main difference from band-wise fitting: one pixel uses one shared linear system per IRLS round, rather than one system per band.

## Date-Level Vector Huber Weights

NUFROST does not assign independent Huber weights to each band. Instead, it treats one date as one multi-band observation and computes a vector residual:

$$
\mathbf r_i
=
\mathbf z_i
-
\mathbf x(t_i)^\top\Theta
\in\mathbb R^B.
$$

The residual magnitude is the band-averaged RMS:

$$
e_i
=
\sqrt{
\frac{1}{B}
\sum_{b=1}^{B}
r_{i,b}^2
}.
$$

The date-level Huber weight is:

$$
w_i
=
\begin{cases}
1, & e_i \le \delta,\\
\frac{\delta}{e_i+\epsilon}, & e_i > \delta.
\end{cases}
$$

This means a polluted acquisition date is down-weighted as a whole. That matches the remote-sensing failure mode: clouds, haze, shadows, snow contamination, and atmospheric artifacts usually affect the observation date, not only one isolated band.

The implementation also applies damping:

$$
w_i^{(k+1)}
\leftarrow
\rho w_i^{(k)}
+
(1-\rho)\tilde w_i^{(k+1)}
$$

and a small minimum weight:

$$
w_i \ge w_{\min}.
$$

This avoids unstable weight oscillation and keeps the ridge system numerically well-conditioned under sparse observations.

## Frequency-Weighted Ridge

The ridge penalty is diagonal:

$$
\Lambda
=
\operatorname{diag}(\lambda_1,\dots,\lambda_P).
$$

DC and trend terms receive ordinary ridge weights. Harmonic terms receive frequency-dependent penalties. For frequency $f_k$, the base multiplier is:

$$
d_k
=
\left(
\frac{f_k}{f_{\min}}
\right)^\alpha,
$$

where $\alpha$ is controlled by `freq_weight`.

Cosine and sine columns for the same frequency share the same penalty:

$$
\lambda_{\cos,k}
=
\lambda_{\sin,k}
=
\lambda_\beta d_k^2.
$$

This suppresses high-frequency noise and improves conditioning when the selected harmonic basis is nearly collinear, especially across long temporal gaps.

The configuration also supports a high-frequency tier through:

- `lambda_high`
- `low_freq_period_days`

Frequencies whose periods are shorter than the configured low-frequency threshold can receive additional ridge penalty.

## Prediction

At target time $t_\star$, NUFROST builds the same basis vector:

$$
\mathbf x_\star = \mathbf x(t_\star).
$$

The standardized prediction is:

$$
\hat{\mathbf z}_\star
=
\mathbf x_\star^\top\Theta.
$$

The final prediction is transformed back to the original band scale:

$$
\boxed{
\hat{\mathbf y}_\star
=
\boldsymbol\mu
+
\mathbf s \odot \hat{\mathbf z}_\star
}
$$

This gives one reconstructed value per band.

## Current Active Multi-Band Model

The active NUFROST full-scene path is:

$$
\mathbf y_i
\rightarrow
Z
\rightarrow
\mathbf F(f)
\rightarrow
P(f)
\rightarrow
\mathcal F
\rightarrow
X
\rightarrow
\Theta
\rightarrow
\hat{\mathbf y}(t_\star).
$$

In words:

1. Build a valid joint timestamp mask for the pixel.
2. Robustly standardize each band.
3. Run vector-valued NUFFT on the shared non-uniform timestamp grid.
4. Convert the vector spectrum to a joint power spectrum.
5. Select shared harmonic frequencies.
6. Build one harmonic/trend design matrix.
7. Fit all bands with vector Huber-IRLS and multi-output frequency-weighted Ridge.
8. Predict the target multi-band vector and de-standardize.

This is the current high-dimensional vector trajectory model.

## Configuration

NUFROST defaults live in:

```text
config/nufrost.json
```

Important fields:

- `modes`: number of NUFFT modes before positive-frequency selection.
- `frequency_selection`: `spectral`, `preferred`, or `shared_spectral`.
- `preferred_periods_days`: comma-separated periods used by the hybrid/preferred modes.
- `preferred_top_k`: number of preferred components to keep.
- `spectral_top_k`: number of spectral peaks to keep.
- `spectral_merge_tol`: relative tolerance for merging nearby frequencies.
- `refine_peaks`: enable parabolic local peak refinement.
- `include_trend`: include the centered linear trend column.
- `ridge`: base ridge coefficient $\lambda_\beta$.
- `freq_weight`: exponent $\alpha$ for frequency-weighted ridge.
- `lambda_high`: extra high-frequency ridge level.
- `low_freq_period_days`: period threshold for high-frequency classification.
- `huber_iters`: maximum vector Huber-IRLS iterations.
- `huber_delta`: configured Huber threshold; the vector path applies a standardized residual lower bound internally.
- `min_obs`: minimum valid observations needed for a fit.

Some older configuration keys for step/fused-lasso experiments remain in the JSON for compatibility, but the current vector NUFROST path described above uses the multi-output Huber-Ridge trajectory model.

## Rust Workspace

The active Rust implementation is under `crates/`:

```text
crates/
  gdal/            # GeoTIFF/VRT I/O, timestamp parsing, full-scene helpers
  nufrost-core/    # NUFROST mathematical model and NUFFT/fitting logic
  hants-core/      # HANTS comparison baseline
  zhu2015-core/    # Zhu2015 comparison baseline
  nufrost-cli/     # command-line entrypoint
```

The baseline crates are separate comparison algorithms and are not part of the NUFROST mathematical model described here.

## Build And Test

The workspace requires Rust 1.85+. Raster and CLI builds also need GDAL. `./scripts/setup.sh` installs `libgdal-core` 3.10.3 into `.conda/gdal` with conda, mamba, or micromamba.

```bash
./scripts/setup.sh
./scripts/test.sh
```

## CLI Examples

Single NUFROST GeoTIFF reconstruction:

```bash
cargo run -p nufrost-cli -- nufrost \
  --input-geotiff input.tif \
  --output pred.tif
```

Full-scene Sentinel-2 reconstruction:

```bash
cargo run --release -p nufrost-cli -- full-scene \
  --source-name sentinel-2 \
  --lon 94.2605 \
  --lat 29.7733 \
  --data-root data \
  --output-root data/products/reconstruction
```

One-pixel timing and RMSE smoke test:

```bash
cargo run --release -p nufrost-cli -- pixel-bench \
  --source-name sentinel-2 \
  --lon 94.2605 \
  --lat 29.7733 \
  --data-root data \
  --row 513 \
  --col 587 \
  --repeats 50
```

## Implementation Notes

The key implementation entry points are:

```text
crates/nufrost-core/src/nufft.rs
  type1_vector_power_kb

crates/nufrost-core/src/nufrost.rs
  nufrost_pixel_vector
  multi_output_tiered_ridge_solve
```

`type1_vector_power_kb` performs the vector-valued gridded NUFFT and returns the shared frequency grid with joint power.

`nufrost_pixel_vector` is the active multi-band pixel model. It performs robust standardization, shared frequency discovery, date-level vector Huber weighting, multi-output Ridge fitting, and final de-standardized prediction.

`multi_output_tiered_ridge_solve` builds the shared normal equation and solves all band outputs together.
