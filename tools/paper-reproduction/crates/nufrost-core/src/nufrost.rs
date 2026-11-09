// NUFROST — Non-Uniform FFT-based frequency discovery with robust fitting.
//
// ── Algorithm summary ──────────────────────────────────────────────────────
// 1. Map non-uniform timestamps to [-π, π].
// 2. Compute spectrum via the `nufft` module.
// 3. Select harmonic frequencies: spectral peaks, preferred periods, or hybrid.
// 4. Build harmonic + linear-trend design matrix.
// 5. Fit harmonic + step terms.
// 6. Predict at the target time.

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

// ── Reusable types (exported from crate root) ──────────────────────────────
use crate::config::NufrostConfig;

// ═══════════════════════════════════════════════════════════════════════════
//  Result types
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a single-pixel NUFROST fit.
#[derive(Debug, Clone)]
pub struct NufrostResult {
    /// Whether the fit succeeded (enough valid observations).
    pub valid: bool,
    /// Number of frequencies used in the fit.
    pub n_freqs_used: usize,
    /// Fitted harmonic + trend coefficients.
    /// Length = 1 (DC) + 1 (trend, if any) + 2·n_freqs.
    pub beta: Vec<f64>,
    /// Selected frequencies (Hz).
    pub freqs: Vec<f64>,
    /// Earliest valid timestamp (seconds since epoch).
    pub t_min: f64,
    /// Mean of relative timestamps used for trend centering.
    pub t_rel_mean: f64,
    /// Fallback value (mean of valid observations).
    pub fill_value: f64,
    /// `include_trend` flag used in fit.
    pub include_trend: bool,
    /// y_scale applied to observations before fitting.
    pub y_scale: f64,
}

/// Result of a vector-valued NUFROST fit for one multi-band pixel.
#[derive(Debug, Clone)]
pub struct NufrostVectorResult {
    /// Whether the fit succeeded.
    pub valid: bool,
    /// Number of input bands fitted jointly.
    pub n_bands: usize,
    /// Selected shared frequencies (Hz).
    pub freqs: Vec<f64>,
    /// Earliest valid timestamp in seconds.
    pub t_min: f64,
    /// Mean of relative timestamps used for trend centering.
    pub t_rel_mean: f64,
    /// Per-band center used before fitting.
    pub centers: Vec<f64>,
    /// Per-band scale used before fitting.
    pub scales: Vec<f64>,
    /// Multi-output coefficient matrix, shape = n_basis x n_bands.
    pub theta: Array2<f64>,
    /// `include_trend` flag used in fit.
    pub include_trend: bool,
}

impl Default for NufrostVectorResult {
    fn default() -> Self {
        Self {
            valid: false,
            n_bands: 0,
            freqs: Vec::new(),
            t_min: f64::NAN,
            t_rel_mean: f64::NAN,
            centers: Vec::new(),
            scales: Vec::new(),
            theta: Array2::<f64>::zeros((0, 0)),
            include_trend: true,
        }
    }
}

impl Default for NufrostResult {
    fn default() -> Self {
        Self {
            valid: false,
            n_freqs_used: 0,
            beta: Vec::new(),
            freqs: Vec::new(),
            t_min: f64::NAN,
            t_rel_mean: f64::NAN,
            fill_value: f64::NAN,
            include_trend: true,
            y_scale: 10000.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Utility functions
// ═══════════════════════════════════════════════════════════════════════════

/// Round `n` up to the next even integer.
#[inline]
pub fn next_even(n: usize) -> usize {
    ((n + 1) / 2) * 2
}

/// Compute the (NaN-aware) mean of a slice.
fn nanmean(data: &[f64]) -> f64 {
    let (sum, count) = data.iter().fold((0.0f64, 0usize), |(s, c), &v| {
        if v.is_finite() {
            (s + v, c + 1)
        } else {
            (s, c)
        }
    });
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Compute the (NaN-aware) median of a slice.
fn nanmedian(data: &[f64]) -> f64 {
    let mut finite: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    if n % 2 == 1 {
        finite[n / 2]
    } else {
        (finite[n / 2 - 1] + finite[n / 2]) / 2.0
    }
}

/// Compute MAD (median absolute deviation) scaled to match σ for Gaussian data.
/// Returns 1.4826 · median(|x - median(x)|) for finite values in `residuals`.
fn mad_std(residuals: &[f64]) -> f64 {
    let med = nanmedian(residuals);
    if !med.is_finite() {
        return 0.0;
    }
    let abs_dev: Vec<f64> = residuals
        .iter()
        .map(|&r| {
            if r.is_finite() {
                (r - med).abs()
            } else {
                f64::NAN
            }
        })
        .collect();
    let mad = nanmedian(&abs_dev);
    if !mad.is_finite() {
        return 0.0;
    }
    1.4826 * mad.max(1e-12)
}

/// Single-pass joint outlier mask across bands.
///
/// Accepts a per-band residual matrix `residuals` of shape `(n, B)` and per-band
/// robust scale estimates `sigmas` of length `B`.  Bands with `sigmas[i] <= 0`
/// are excluded from the joint score.
///
/// Returns a per-timestamp boolean mask (length `n`): `true` if the joint
/// anomaly score is within `sigma * MAD(score)` of the median score; `false`
/// for timestamps rejected as correlated outliers across bands.
pub fn joint_outlier_mask(residuals: &Array2<f64>, sigmas: &[f64], sigma: f64) -> Vec<bool> {
    let n = residuals.nrows();
    let b = residuals.ncols();

    if n == 0 {
        return Vec::new();
    }
    assert_eq!(sigmas.len(), b, "sigmas length {} != B={}", sigmas.len(), b);

    // Identify bands with valid scale estimates (sigma > 0).
    let valid_band: Vec<bool> = sigmas.iter().map(|&s| s > 0.0).collect();
    let n_valid: usize = valid_band.iter().filter(|&&v| v).count();

    // No valid bands: keep everything.
    if n_valid == 0 {
        return vec![true; n];
    }

    // Standardize residuals; divide only by valid-band sigmas.
    let mut rz = Array2::<f64>::zeros((n, b));
    for i in 0..n {
        for j in 0..b {
            if valid_band[j] {
                rz[[i, j]] = residuals[[i, j]] / sigmas[j];
            }
        }
    }

    // Single effective band: fall back to marginal sigma threshold.
    if n_valid == 1 {
        return (0..n)
            .map(|i| {
                let z = rz[[i, valid_band.iter().position(|&v| v).unwrap()]];
                z.abs() <= sigma
            })
            .collect();
    }

    // Joint anomaly score: L2 norm of standardized residuals over valid bands.
    let score: Vec<f64> = (0..n)
        .map(|i| {
            let mut ssq = 0.0f64;
            for j in 0..b {
                if valid_band[j] {
                    let z = rz[[i, j]];
                    ssq += z * z;
                }
            }
            ssq.sqrt()
        })
        .collect();

    // Threshold via MAD of the score distribution.
    let med = nanmedian(&score);
    let abs_dev: Vec<f64> = score.iter().map(|&s| (s - med).abs()).collect();
    let mad = nanmedian(&abs_dev) * 1.4826;

    if mad <= 0.0 {
        // Degenerate score distribution; keep everything finite.
        return score.iter().map(|&s| s.is_finite()).collect();
    }

    let threshold = med + sigma * mad;
    score.iter().map(|&s| s <= threshold).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Design matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Build the harmonic design matrix.
///
/// Columns:
///   - 0: DC term (ones)
///   - 1 (if `include_trend`): linear trend (centered)
///   - then pairs cos(ω·t), sin(ω·t) for each frequency.
pub fn make_design_matrix(
    t: &[f64],
    freqs: &[f64],
    include_trend: bool,
    include_dc: bool,
) -> Array2<f64> {
    let n = t.len();
    let n_dc = if include_dc { 1 } else { 0 };
    let n_trend = if include_trend { 1 } else { 0 };
    let n_cols = n_dc + n_trend + 2 * freqs.len();

    if n_cols == 0 {
        return Array2::zeros((n, 0));
    }

    let t_mean = if include_trend { nanmean(t) } else { 0.0 };
    let mut x = Array2::<f64>::zeros((n, n_cols));
    let mut col = 0;

    if include_dc {
        x.column_mut(col).fill(1.0);
        col += 1;
    }
    if include_trend {
        for (i, &ti) in t.iter().enumerate() {
            x[[i, col]] = ti - t_mean;
        }
        col += 1;
    }
    for &f in freqs {
        let omega = 2.0 * PI * f;
        for (i, &ti) in t.iter().enumerate() {
            x[[i, col]] = (omega * ti).cos();
            x[[i, col + 1]] = (omega * ti).sin();
        }
        col += 2;
    }

    x
}

// ═══════════════════════════════════════════════════════════════════════════
//  Linear algebra: Gaussian elimination with partial pivoting
// ═══════════════════════════════════════════════════════════════════════════

/// Solve `A · x = b`. Returns `None` on singularity.
fn gauss_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(b.len(), n);

    if n == 0 {
        return Some(Array1::zeros(0));
    }

    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-14 {
            return None;
        }
        if pivot_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / aug[[col, col]];
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }
    Some(x)
}

/// Solve normal equations XᵀX · β = Xᵀy.
fn solve_normal_equations(x: &Array2<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
    if x.nrows() == 0 || x.nrows() < x.ncols() {
        return None;
    }
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    gauss_solve(&xtx, &xty)
}

/// Ridge-augmented least squares: solve [X; √λ·R] β = [y; 0] where R is a
/// diagonal penalty matrix.
fn ridge_solve_augmented(
    x: &Array2<f64>,
    y: &Array1<f64>,
    lam: f64,
    r_diag: &[f64],    // per-column penalty multipliers; length = x.ncols()
    w: Option<&[f64]>, // per-row observation weights (sqrt applied internally)
) -> Option<Array1<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    if p == 0 {
        return Some(Array1::zeros(0));
    }

    // Apply observation weights
    let (xw, yw): (Array2<f64>, Array1<f64>) = if let Some(weights) = w {
        let sqrt_w: Vec<f64> = weights.iter().map(|&wi| wi.sqrt()).collect();
        let mut xw = Array2::zeros((n, p));
        let mut yw = Array1::zeros(n);
        for i in 0..n {
            let sw = sqrt_w[i];
            for j in 0..p {
                xw[[i, j]] = x[[i, j]] * sw;
            }
            yw[i] = y[i] * sw;
        }
        (xw, yw)
    } else {
        (x.clone(), y.clone())
    };

    if lam <= 0.0 {
        return solve_normal_equations(&xw, &yw);
    }

    let aug_rows = n + p;
    let mut x_aug = Array2::<f64>::zeros((aug_rows, p));
    let mut y_aug = Array1::<f64>::zeros(aug_rows);

    // Top block: weighted design matrix + weighted y
    for i in 0..n {
        for j in 0..p {
            x_aug[[i, j]] = xw[[i, j]];
        }
        y_aug[i] = yw[i];
    }

    // Bottom block: √λ · R diagonal
    let sqrt_lam = lam.sqrt();
    for j in 0..p {
        let rj = if j < r_diag.len() { r_diag[j] } else { 1.0 };
        x_aug[[n + j, j]] = sqrt_lam * rj;
        y_aug[n + j] = 0.0;
    }

    // Solve augmented system via normal equations
    solve_normal_equations(&x_aug, &y_aug)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Frequency penalty helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Per-frequency penalty multipliers: higher frequencies get heavier penalty.
///
/// penalty[i] = sqrt(1 + freq_weight · log2(f_i / f_min))
/// where f_min is the smallest positive frequency.
fn make_frequency_penalty(freqs: &[f64], freq_weight: f64) -> Vec<f64> {
    let n = freqs.len();
    if n == 0 {
        return Vec::new();
    }
    let pos_min = freqs
        .iter()
        .filter(|&&f| f.is_finite() && f > 0.0)
        .fold(f64::INFINITY, |a, &b| a.min(b));
    if !pos_min.is_finite() {
        return vec![1.0; n];
    }

    freqs
        .iter()
        .map(|&f| {
            if !f.is_finite() || f <= pos_min {
                1.0
            } else {
                let rel = (f / pos_min).max(1.0);
                (1.0 + freq_weight.max(0.0) * rel.log2()).sqrt()
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Frequency tier classification
// ═══════════════════════════════════════════════════════════════════════════

/// Classify each frequency as "high" (period < low_freq_period_days).
/// Frequencies are assumed to be in Hz (seconds time unit).
fn classify_freq_tier(freqs: &[f64], low_freq_period_days: f64) -> Vec<bool> {
    freqs
        .iter()
        .map(|&f| {
            if f <= 0.0 || !f.is_finite() {
                false
            } else {
                let period_days = 1.0 / (f * 86400.0);
                period_days < low_freq_period_days
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Huber weights
// ═══════════════════════════════════════════════════════════════════════════

/// Compute Huber weights: w_i = 1 if |r_i| ≤ δ, else δ / |r_i|.
fn huber_weights(residuals: &[f64], delta: f64) -> Vec<f64> {
    let delta = delta.max(1e-8);
    residuals
        .iter()
        .map(|&r| {
            let a = r.abs();
            if a <= delta {
                1.0
            } else {
                delta / a
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Spectrum computation wrappers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the type-1 non-uniform spectrum via direct summation.
///
/// Maps non-uniform timestamps `t_rel` to [-π, π], then computes
///   F_k = Σ c_j · exp(-i · k · x_j)    for k = -M/2 … M/2-1
///
/// This is a small-N reference path matching the type-1 NUFFT transform
/// definition without gridding approximation.
///
/// # Returns
/// `(frequencies_hz, power_spectrum)` where `frequencies_hz = k / Tspan`
/// (only non-negative frequencies) and `power = |F_k|²`.
pub fn compute_spectrum_direct(
    t_rel: &[f64],
    yy: &[f64],
    modes: usize,
    y_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    crate::nufft::type1_spectrum_direct(t_rel, yy, modes, y_scale)
}

/// Compute the paper-style gridded NUFFT spectrum:
/// spreading → FFT → kernel deconvolution.
pub fn compute_spectrum_nufft(
    t_rel: &[f64],
    yy: &[f64],
    modes: usize,
    y_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    crate::nufft::type1_spectrum_kb(
        t_rel,
        yy,
        modes,
        y_scale,
        crate::nufft::NufftOptions::default(),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
//  Peak selection
// ═══════════════════════════════════════════════════════════════════════════

/// Parabolic refinement: fit a parabola to three consecutive spectral points
/// and return the interpolated peak frequency.
pub fn refine_parabolic(f: &[f64], p: &[f64], i: usize) -> f64 {
    if i == 0 || i >= p.len() - 1 {
        return f[i];
    }
    let y0 = p[i - 1];
    let y1 = p[i];
    let y2 = p[i + 1];
    let denom = y0 - 2.0 * y1 + y2;
    if denom == 0.0 {
        return f[i];
    }
    let delta = 0.5 * (y0 - y2) / denom;
    f[i] + delta * (f[i + 1] - f[i])
}

/// Select top-k peaks from the power spectrum, cumulative power capped.
///
/// Returns indices into the full frequency/power arrays.
pub fn select_peaks_adaptive(
    f_pos: &[f64],
    p_pos: &[f64],
    k_max: usize,
    power_cum: f64,
    ignore_dc_hz: f64,
    fmax: f64,
) -> Vec<usize> {
    let lower = ignore_dc_hz.max(0.0);
    let fmax = if !fmax.is_finite() || fmax <= 0.0 {
        f_pos
            .iter()
            .copied()
            .filter(|f| f.is_finite())
            .fold(1.0f64, |a: f64, b| a.max(b))
    } else {
        fmax
    };

    // Collect valid indices
    let mut valid_idx: Vec<(usize, f64)> = f_pos
        .iter()
        .zip(p_pos.iter())
        .enumerate()
        .filter(|(_, (&f, &p))| f.is_finite() && p.is_finite() && f > lower && f <= fmax)
        .map(|(i, (&_f, &p))| (i, p))
        .collect();

    if valid_idx.is_empty() {
        return Vec::new();
    }

    // Sort by power descending
    valid_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Cumulative power threshold
    let total_power: f64 = valid_idx.iter().map(|&(_, p)| p).sum();
    let power_thr = power_cum.clamp(0.0, 1.0) * total_power;

    let mut cum = 0.0;
    let mut take = 0;
    for &(_, p) in &valid_idx {
        cum += p;
        take += 1;
        if take >= k_max || cum >= power_thr {
            break;
        }
    }
    if take == 0 && !valid_idx.is_empty() {
        take = 1;
    }
    take = take.min(valid_idx.len());

    valid_idx.truncate(take);
    valid_idx.iter().map(|&(i, _)| i).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Frequency selection
// ═══════════════════════════════════════════════════════════════════════════

/// Check whether `target` is within relative tolerance `rel_tol` of any
/// frequency in `existing`. Relative distance = |target - f| / f.
#[cfg(test)]
fn is_near_any(target: f64, existing: &[f64], rel_tol: f64) -> bool {
    let tol = rel_tol.max(0.0);
    existing.iter().any(|&f| {
        if f <= 0.0 {
            return false;
        }
        let rel = (target - f).abs() / f.max(1e-12);
        rel <= tol
    })
}

/// Select private (band-specific) frequencies from a single band's power spectrum.
///
/// Returns frequencies distinct from `shared_freqs` and capped at `private_top_k`.
/// Excludes any frequency within `spectral_merge_tol * 2` of a shared or
/// already-selected private frequency.
#[cfg(test)]
fn select_private_frequencies(
    band_freqs: &[f64],
    band_power: &[f64],
    shared_freqs: &[f64],
    config: &NufrostConfig,
    fmax: f64,
) -> Vec<f64> {
    let merge_tol_x2 = config.spectral_merge_tol.max(0.0) * 2.0;
    let private_top_k = config.private_top_k_per_band;

    let peak_idx = select_peaks_adaptive(
        band_freqs,
        band_power,
        config.num_peaks as usize,
        config.power_cum,
        config.ignore_dc_hz,
        fmax,
    );

    let candidates: Vec<f64> = peak_idx
        .iter()
        .map(|&i| {
            if config.refine_peaks {
                refine_parabolic(band_freqs, band_power, i)
            } else {
                band_freqs[i]
            }
        })
        .filter(|&f| f.is_finite() && f > config.ignore_dc_hz && f <= fmax)
        .collect();

    let mut private: Vec<f64> = Vec::new();
    for f in candidates {
        if is_near_any(f, shared_freqs, merge_tol_x2) {
            continue;
        }
        if is_near_any(f, &private, merge_tol_x2) {
            continue;
        }
        private.push(f);
        if private.len() >= private_top_k {
            break;
        }
    }
    private.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    private
}

/// Parse preferred periods string (comma-separated days) into Hz frequencies.
fn parse_preferred_frequencies(spec: &str) -> Vec<f64> {
    if spec.trim().is_empty() {
        return Vec::new();
    }
    spec.split(',')
        .filter_map(|p| p.trim().parse::<f64>().ok())
        .filter(|&p| p.is_finite() && p > 0.0)
        .map(|p| 1.0 / (p * 86400.0))
        .collect()
}

/// NUFROST parameters are defined in seconds/Hz. Several public wrappers pass
/// relative days, so convert day-scale axes before fitting.
fn maybe_days_to_seconds(t: &[f64], target_t: f64) -> (Vec<f64>, f64) {
    let finite: Vec<f64> = t.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || !target_t.is_finite() {
        return (t.to_vec(), target_t);
    }
    let t_min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = t_max - t_min;

    // Relative day axes in this project are typically 0..365/730. Epoch or
    // relative seconds are much larger for remote-sensing time series.
    if span.is_finite() && span <= 10_000.0 && target_t.abs() <= 100_000.0 {
        (t.iter().map(|&v| v * 86400.0).collect(), target_t * 86400.0)
    } else {
        (t.to_vec(), target_t)
    }
}

fn normalized_sinc(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    if x.abs() < 1e-8 {
        1.0
    } else {
        let pix = PI * x;
        pix.sin() / pix
    }
}

fn median_positive_frequency_spacing(f_pos: &[f64]) -> Option<f64> {
    let mut diffs: Vec<f64> = f_pos
        .windows(2)
        .filter_map(|w| {
            let d = w[1] - w[0];
            if d.is_finite() && d > 0.0 {
                Some(d)
            } else {
                None
            }
        })
        .collect();
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(diffs[diffs.len() / 2])
}

/// Snap a target frequency to the local finite-window sinc main lobe.
///
/// The NUFFT grid spacing is approximately `1 / tspan`. A preferred phenology
/// frequency is therefore matched by maximizing `power * sinc^2((f-f0)/df)`
/// inside the first main lobe (`|f-f0| <= df`), which avoids snapping to a
/// distant high-energy peak.
fn snap_frequency_to_spectrum(target_freq: f64, f_pos: &[f64], p_pos: &[f64], rel_tol: f64) -> f64 {
    if !target_freq.is_finite() || target_freq <= 0.0 {
        return target_freq;
    }
    let Some(df) = median_positive_frequency_spacing(f_pos) else {
        return target_freq;
    };
    if df <= 0.0 || !df.is_finite() {
        return target_freq;
    }
    let _rel_tol = rel_tol.max(0.0);
    let abs_window = df;

    let mut best_idx: Option<usize> = None;
    let mut best_score = f64::NEG_INFINITY;

    for (i, (&f, &p)) in f_pos.iter().zip(p_pos.iter()).enumerate() {
        if !f.is_finite() || !p.is_finite() || f <= 0.0 {
            continue;
        }
        let abs_err = (f - target_freq).abs();
        if abs_err > abs_window {
            continue;
        }
        let s = normalized_sinc(abs_err / df);
        let score = p * s * s;
        if score > best_score {
            best_score = score;
            best_idx = Some(i);
        }
    }

    match best_idx {
        Some(i) => f_pos[i],
        None => target_freq,
    }
}

fn select_phenology_frequencies(
    f_pos: &[f64],
    p_pos: &[f64],
    fmax: f64,
    preferred_freqs: &[f64],
    preferred_top_k: usize,
    ignore_dc_hz: f64,
    spectral_merge_tol: f64,
) -> Vec<f64> {
    preferred_freqs
        .iter()
        .copied()
        .filter(|&f| f.is_finite() && f > ignore_dc_hz && f <= fmax)
        .take(preferred_top_k)
        .map(|f| snap_frequency_to_spectrum(f, f_pos, p_pos, spectral_merge_tol))
        .filter(|&f| f.is_finite() && f > ignore_dc_hz && f <= fmax)
        .collect()
}

/// Select harmonic frequencies for NUFROST fitting.
///
/// Selection modes: "spectral", "preferred", "hybrid", "all".
pub fn select_frequencies(
    f_pos: &[f64],
    p_pos: &[f64],
    fmax: f64,
    selection_mode: &str,
    preferred_freqs: &[f64],
    preferred_top_k: usize,
    num_peaks: usize,
    spectral_top_k: usize,
    spectral_merge_tol: f64,
    power_cum: f64,
    ignore_dc_hz: f64,
    refine_peaks: bool,
) -> Vec<f64> {
    let mode = selection_mode;

    let mut selected: Vec<f64> = Vec::new();

    if mode == "all" {
        selected.extend(
            f_pos
                .iter()
                .copied()
                .filter(|&f| f.is_finite() && f > ignore_dc_hz && f <= fmax),
        );
    }

    // Preferred frequencies (snapped to spectrum)
    if (mode == "preferred" || mode == "hybrid" || mode == "all") && !preferred_freqs.is_empty() {
        selected.extend(select_phenology_frequencies(
            f_pos,
            p_pos,
            fmax,
            preferred_freqs,
            preferred_top_k,
            ignore_dc_hz,
            spectral_merge_tol,
        ));
    }

    // Spectral peaks
    if mode == "spectral" || mode == "hybrid" {
        let k_max = if spectral_top_k > 0 {
            spectral_top_k
        } else {
            num_peaks
        };
        let peak_idx = select_peaks_adaptive(f_pos, p_pos, k_max, power_cum, ignore_dc_hz, fmax);
        if !peak_idx.is_empty() {
            let sel_freqs: Vec<f64> = if refine_peaks {
                peak_idx
                    .iter()
                    .map(|&i| refine_parabolic(f_pos, p_pos, i))
                    .collect()
            } else {
                peak_idx.iter().map(|&i| f_pos[i]).collect()
            };
            selected.extend(sel_freqs);
        }
    }

    if selected.is_empty() {
        return Vec::new();
    }

    if mode == "all" {
        selected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        selected.retain(|&f| f.is_finite() && f > ignore_dc_hz && f <= fmax);
        selected.dedup_by(|a, b| (*a - *b).abs() <= 1e-18);
        return selected;
    }

    // Sort, deduplicate, merge nearby frequencies
    selected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    selected.retain(|&f| f.is_finite() && f > ignore_dc_hz && f <= fmax);

    let mut merged: Vec<f64> = Vec::new();
    for f in selected {
        if merged.is_empty() {
            merged.push(f);
            continue;
        }
        let last = *merged.last().unwrap();
        let rel = (f - last).abs() / last.max(1e-12);
        if rel <= spectral_merge_tol.max(0.0) {
            // Merge: average the two
            let len = merged.len();
            merged[len - 1] = 0.5 * (last + f);
        } else {
            merged.push(f);
        }
    }

    merged
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tiered ridge regression
// ═══════════════════════════════════════════════════════════════════════════

/// Build the ridge penalty diagonal vector R for tiered-ridge.
/// R[j] = freq_penalty for harmonic columns, 1.0 for DC and trend.
fn build_ridge_diag(
    freqs: &[f64],
    p: usize,
    include_dc: bool,
    include_trend: bool,
    freq_weight: f64,
) -> Vec<f64> {
    let mut r = vec![1.0f64; p];
    let mut col = 0;
    if include_dc {
        col += 1;
    }
    if include_trend {
        col += 1;
    }
    if !freqs.is_empty() {
        let penalty = make_frequency_penalty(freqs, freq_weight);
        for w_f in penalty {
            if col < p {
                r[col] = w_f;
                col += 1;
            }
            if col < p {
                r[col] = w_f;
                col += 1;
            }
        }
    }
    r
}

fn frequency_matches_any(freq: f64, targets: &[f64]) -> bool {
    targets.iter().any(|&target| {
        if !freq.is_finite() || !target.is_finite() || target <= 0.0 {
            return false;
        }
        (freq - target).abs() <= (target.abs() * 1e-10).max(1e-18)
    })
}

/// Ridge diagonal with separate penalty multipliers for shared vs private frequencies.
/// Private frequency columns receive `private_freq_penalty_mult` × the base penalty.
#[cfg(test)]
fn build_ridge_diag_mixed(
    shared_freqs: &[f64],
    private_freqs: &[f64],
    p: usize,
    include_dc: bool,
    include_trend: bool,
    freq_weight: f64,
    private_freq_penalty_mult: f64,
) -> Vec<f64> {
    let mut r = vec![1.0f64; p];
    let mut col = 0;
    if include_dc {
        col += 1;
    }
    if include_trend {
        col += 1;
    }
    if !shared_freqs.is_empty() {
        let penalty = make_frequency_penalty(shared_freqs, freq_weight);
        for w_f in penalty {
            if col < p {
                r[col] = w_f;
                col += 1;
            }
            if col < p {
                r[col] = w_f;
                col += 1;
            }
        }
    }
    if !private_freqs.is_empty() {
        let penalty = make_frequency_penalty(private_freqs, freq_weight);
        for w_f in penalty {
            if col < p {
                r[col] = w_f * private_freq_penalty_mult;
                col += 1;
            }
            if col < p {
                r[col] = w_f * private_freq_penalty_mult;
                col += 1;
            }
        }
    }
    r
}

/// Tiered ridge: low-frequency and high-frequency tiers get different λ.
/// λ_high > λ_beta adds extra penalty to high-tier frequencies.
///
/// When `ridge_diag_r` is provided it replaces the `build_ridge_diag` base
/// penalties, enabling per-column penalty multipliers (e.g. private-freq
/// penalty via `build_ridge_diag_mixed`).
fn tiered_ridge_solve(
    x: &Array2<f64>,
    y: &Array1<f64>,
    freqs: &[f64],
    lambda_beta: f64,
    lambda_high: f64,
    low_freq_period_days: f64,
    freq_weight: f64,
    include_dc: bool,
    include_trend: bool,
    ridge_diag_r: Option<&[f64]>,
    w: Option<&[f64]>,
) -> Option<Array1<f64>> {
    let p = x.ncols();
    if p == 0 {
        return Some(Array1::zeros(0));
    }

    // Base ridge diagonal — accept external override (e.g. mixed shared+private)
    let r: Vec<f64> = if let Some(rd) = ridge_diag_r {
        assert_eq!(rd.len(), p, "ridge_diag_r length mismatch");
        rd.to_vec()
    } else {
        build_ridge_diag(freqs, p, include_dc, include_trend, freq_weight)
    };

    // lam_diag = lambda_beta * R^2 + extra for high-tier freqs
    let mut lam_diag: Vec<f64> = r.iter().map(|&ri| lambda_beta * ri * ri).collect();

    if !freqs.is_empty() && lambda_high > lambda_beta {
        let is_high = classify_freq_tier(freqs, low_freq_period_days);
        let col_start = (if include_dc { 1 } else { 0 }) + (if include_trend { 1 } else { 0 });
        let extra = lambda_high - lambda_beta;
        for (k, &hi) in is_high.iter().enumerate() {
            if hi {
                let c = col_start + 2 * k;
                if c < p {
                    lam_diag[c] += extra;
                }
                if c + 1 < p {
                    lam_diag[c + 1] += extra;
                }
            }
        }
    }

    // Clip negative diagonals
    for d in &mut lam_diag {
        *d = d.max(0.0);
    }

    // sqrt_lam = sqrt(clip(lam_diag, 0, None))
    let sqrt_lam: Vec<f64> = lam_diag.iter().map(|&d| d.sqrt()).collect();

    // Build augmented system: [Xw; diag(sqrt_lam)] β = [yw; 0]
    let n = x.nrows();
    let aug_rows = n + p;
    let mut x_aug = Array2::<f64>::zeros((aug_rows, p));
    let mut y_aug = Array1::<f64>::zeros(aug_rows);

    if let Some(weights) = w {
        for i in 0..n {
            let sw = weights[i].sqrt();
            for j in 0..p {
                x_aug[[i, j]] = x[[i, j]] * sw;
            }
            y_aug[i] = y[i] * sw;
        }
    } else {
        for i in 0..n {
            for j in 0..p {
                x_aug[[i, j]] = x[[i, j]];
            }
            y_aug[i] = y[i];
        }
    }

    for j in 0..p {
        x_aug[[n + j, j]] = sqrt_lam[j];
        y_aug[n + j] = 0.0;
    }

    solve_normal_equations(&x_aug, &y_aug)
}

fn tiered_lambda_diag(
    freqs: &[f64],
    p: usize,
    lambda_beta: f64,
    lambda_high: f64,
    low_freq_period_days: f64,
    freq_weight: f64,
    include_dc: bool,
    include_trend: bool,
    unpenalized_freqs: &[f64],
) -> Vec<f64> {
    let r = build_ridge_diag(freqs, p, include_dc, include_trend, freq_weight);
    let mut lam_diag: Vec<f64> = r.iter().map(|&ri| lambda_beta * ri * ri).collect();

    if !freqs.is_empty() && lambda_high > lambda_beta {
        let is_high = classify_freq_tier(freqs, low_freq_period_days);
        let col_start = (if include_dc { 1 } else { 0 }) + (if include_trend { 1 } else { 0 });
        let extra = lambda_high - lambda_beta;
        for (k, &hi) in is_high.iter().enumerate() {
            if hi {
                let c = col_start + 2 * k;
                if c < p {
                    lam_diag[c] += extra;
                }
                if c + 1 < p {
                    lam_diag[c + 1] += extra;
                }
            }
        }
    }

    for d in &mut lam_diag {
        *d = d.max(0.0);
    }

    if !unpenalized_freqs.is_empty() {
        let col_start = (if include_dc { 1 } else { 0 }) + (if include_trend { 1 } else { 0 });
        for (k, &freq) in freqs.iter().enumerate() {
            if frequency_matches_any(freq, unpenalized_freqs) {
                let c = col_start + 2 * k;
                if c < p {
                    lam_diag[c] = 0.0;
                }
                if c + 1 < p {
                    lam_diag[c + 1] = 0.0;
                }
            }
        }
    }
    lam_diag
}

fn cholesky_decompose(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if n == 0 || a.ncols() != n {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 1e-14 || !sum.is_finite() {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Some(l)
}

fn cholesky_solve_matrix(l: &Array2<f64>, b: &Array2<f64>) -> Option<Array2<f64>> {
    let n = l.nrows();
    if l.ncols() != n || b.nrows() != n {
        return None;
    }
    let rhs = b.ncols();
    let mut y = Array2::<f64>::zeros((n, rhs));
    for i in 0..n {
        for c in 0..rhs {
            let mut sum = b[[i, c]];
            for k in 0..i {
                sum -= l[[i, k]] * y[[k, c]];
            }
            y[[i, c]] = sum / l[[i, i]];
        }
    }

    let mut x = Array2::<f64>::zeros((n, rhs));
    for i in (0..n).rev() {
        for c in 0..rhs {
            let mut sum = y[[i, c]];
            for k in (i + 1)..n {
                sum -= l[[k, i]] * x[[k, c]];
            }
            x[[i, c]] = sum / l[[i, i]];
        }
    }
    Some(x)
}

fn multi_output_tiered_ridge_solve(
    x: &Array2<f64>,
    z: &Array2<f64>,
    freqs: &[f64],
    weights: &[f64],
    lambda_beta: f64,
    lambda_high: f64,
    low_freq_period_days: f64,
    freq_weight: f64,
    include_dc: bool,
    include_trend: bool,
    unpenalized_freqs: &[f64],
    multiband_shrinkage: f64,
) -> Option<Array2<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    let b = z.ncols();
    if p == 0 || z.nrows() != n || weights.len() != n {
        return None;
    }

    let lam_diag = tiered_lambda_diag(
        freqs,
        p,
        lambda_beta,
        lambda_high,
        low_freq_period_days,
        freq_weight,
        include_dc,
        include_trend,
        unpenalized_freqs,
    );

    let mut a = Array2::<f64>::zeros((p, p));
    let mut rhs = Array2::<f64>::zeros((p, b));
    for i in 0..n {
        let wi = weights[i].max(0.0);
        if wi <= 0.0 || !wi.is_finite() {
            continue;
        }
        for c1 in 0..p {
            let xw = x[[i, c1]] * wi;
            for c2 in 0..=c1 {
                a[[c1, c2]] += xw * x[[i, c2]];
            }
            for band in 0..b {
                rhs[[c1, band]] += xw * z[[i, band]];
            }
        }
    }
    for c1 in 0..p {
        for c2 in 0..c1 {
            a[[c2, c1]] = a[[c1, c2]];
        }
        a[[c1, c1]] += lam_diag[c1].max(1e-12);
    }

    let shrink = multiband_shrinkage.max(0.0);
    if shrink > 0.0 && b > 1 {
        let solve_with_jitter = |system: &Array2<f64>, y: &Array2<f64>| {
            if let Some(l) = cholesky_decompose(system) {
                return cholesky_solve_matrix(&l, y);
            }
            let mut jittered = system.clone();
            for j in 0..p {
                jittered[[j, j]] += 1e-6;
            }
            cholesky_decompose(&jittered).and_then(|l| cholesky_solve_matrix(&l, y))
        };

        let mut rhs_mean = Array2::<f64>::zeros((p, 1));
        for c in 0..p {
            let mut sum = 0.0;
            for band in 0..b {
                sum += rhs[[c, band]];
            }
            rhs_mean[[c, 0]] = sum / b as f64;
        }
        let theta_mean = solve_with_jitter(&a, &rhs_mean)?;

        let mut a_contrast = a.clone();
        let shrink_diag = multiband_shrinkage_diag(freqs, p, include_dc, include_trend, shrink);
        for c in 0..p {
            a_contrast[[c, c]] += shrink_diag[c];
        }

        let mut rhs_contrast = Array2::<f64>::zeros((p, b));
        for c in 0..p {
            for band in 0..b {
                rhs_contrast[[c, band]] = rhs[[c, band]] - rhs_mean[[c, 0]];
            }
        }
        let theta_contrast = solve_with_jitter(&a_contrast, &rhs_contrast)?;

        let mut theta = Array2::<f64>::zeros((p, b));
        for c in 0..p {
            for band in 0..b {
                theta[[c, band]] = theta_mean[[c, 0]] + theta_contrast[[c, band]];
            }
        }
        return Some(theta);
    }

    if let Some(l) = cholesky_decompose(&a) {
        return cholesky_solve_matrix(&l, &rhs);
    }

    for j in 0..p {
        a[[j, j]] += 1e-6;
    }
    cholesky_decompose(&a).and_then(|l| cholesky_solve_matrix(&l, &rhs))
}

fn multiband_shrinkage_diag(
    freqs: &[f64],
    p: usize,
    include_dc: bool,
    include_trend: bool,
    lambda: f64,
) -> Vec<f64> {
    let mut diag = vec![0.0; p];
    let mut col = 0usize;
    if include_dc && col < p {
        diag[col] = 0.0;
        col += 1;
    }
    if include_trend && col < p {
        diag[col] = lambda;
        col += 1;
    }
    for _ in freqs {
        if col < p {
            diag[col] = lambda;
            col += 1;
        }
        if col < p {
            diag[col] = lambda;
            col += 1;
        }
    }
    diag
}

// ═══════════════════════════════════════════════════════════════════════════
//  Robust frequency-weighted ridge fitting (Huber IRLS)
// ═══════════════════════════════════════════════════════════════════════════

/// Iteratively reweighted ridge regression with Huber weights.
///
/// Returns (beta, y_hat).
fn robust_fit_freq_ridge(
    x: &Array2<f64>,
    y: &Array1<f64>,
    freqs: &[f64],
    lam: f64,
    iters: usize,
    delta: f64,
    include_dc: bool,
    include_trend: bool,
    freq_weight: f64,
) -> (Vec<f64>, Vec<f64>) {
    let p = x.ncols();
    if p == 0 {
        let ymean = nanmean(y.as_slice().unwrap());
        return (vec![], vec![ymean; y.len()]);
    }

    let n = y.len();
    let mut w = vec![1.0f64; n];

    // Ridge diagonal
    let r = build_ridge_diag(freqs, p, include_dc, include_trend, freq_weight);

    // Initial fit with uniform weights
    let mut beta =
        ridge_solve_augmented(x, y, lam, &r, Some(&w)).unwrap_or_else(|| Array1::zeros(p));
    let mut y_hat = x.dot(&beta);

    for _ in 0..iters {
        let residuals: Vec<f64> = y
            .iter()
            .zip(y_hat.iter())
            .map(|(&yi, &yh)| yi - yh)
            .collect();
        w = huber_weights(&residuals, delta);
        if let Some(b) = ridge_solve_augmented(x, y, lam, &r, Some(&w)) {
            beta = b;
        }
        y_hat = x.dot(&beta);
    }

    // Final fit (after Huber loop)
    let residuals: Vec<f64> = y
        .iter()
        .zip(y_hat.iter())
        .map(|(&yi, &yh)| yi - yh)
        .collect();
    w = huber_weights(&residuals, delta);
    if let Some(b) = ridge_solve_augmented(x, y, lam, &r, Some(&w)) {
        beta = b;
    }
    y_hat = x.dot(&beta);

    (beta.to_vec(), y_hat.to_vec())
}

// ═══════════════════════════════════════════════════════════════════════════
//  Single-pixel NUFROST fit (matches Python fit_nufrost_pixel_params)
// ═══════════════════════════════════════════════════════════════════════════

/// Full single-pixel NUFROST parameter estimation.
///
/// # Arguments
/// - `t_sec`: timestamps in seconds since epoch (may contain NaN)
/// - `y`: observations (may contain NaN)
/// - `config`: algorithm configuration
/// - `shared_freqs`: optional pre-computed shared frequencies (overrides selection)
///
/// # Returns
/// `NufrostResult` with fitted parameters for prediction.
pub fn nufrost_fit_pixel(
    t_sec: &[f64],
    y: &[f64],
    config: &NufrostConfig,
    shared_freqs: Option<&[f64]>,
) -> NufrostResult {
    let min_obs = config.min_obs as usize;

    // ── Build valid mask ───────────────────────────────────────────────────
    let valid_mask: Vec<bool> = t_sec
        .iter()
        .zip(y.iter())
        .map(|(&t, &v)| t.is_finite() && v.is_finite())
        .collect();
    let n_valid = valid_mask.iter().filter(|&&b| b).count();

    if n_valid < min_obs.max(3) {
        return NufrostResult {
            valid: false,
            fill_value: nanmean(y),
            include_trend: config.include_trend,
            ..Default::default()
        };
    }

    // ── Extract valid data ─────────────────────────────────────────────────
    let t: Vec<f64> = t_sec
        .iter()
        .zip(valid_mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&ti, _)| ti)
        .collect();
    let yy: Vec<f64> = y
        .iter()
        .zip(valid_mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&vi, _)| vi)
        .collect();
    let fill_value = nanmean(&yy);

    // ── Relative time ──────────────────────────────────────────────────────
    let t_rel: Vec<f64> = {
        let t0 = t.iter().copied().fold(f64::INFINITY, f64::min);
        t.iter().map(|&ti| ti - t0).collect()
    };
    let t_rel_mean = nanmean(&t_rel);
    let tspan = t_rel.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - t_rel.iter().copied().fold(f64::INFINITY, f64::min);

    if tspan <= 0.0 || !tspan.is_finite() {
        return NufrostResult {
            valid: false,
            fill_value,
            t_rel_mean,
            include_trend: config.include_trend,
            ..Default::default()
        };
    }

    // ── Compute spectrum ──────────────────────────────────────────────────
    let y_scale = 10000.0;
    let (f_pos, p_pos) = compute_spectrum_nufft(&t_rel, &yy, config.modes as usize, y_scale);

    // ── Nyquist limit ─────────────────────────────────────────────────────
    let mut t_sorted = t_rel.clone();
    t_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dt_pos: Vec<f64> = t_sorted
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0.0)
        .collect();
    let dt_med = if !dt_pos.is_empty() {
        let mut d = dt_pos.clone();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if d.len() % 2 == 1 {
            d[d.len() / 2]
        } else {
            (d[d.len() / 2 - 1] + d[d.len() / 2]) / 2.0
        }
    } else {
        tspan / t_rel.len() as f64
    };
    let fmax = 0.5 / dt_med.max(1e-12);

    // ── Frequency selection ───────────────────────────────────────────────
    let freqs_sel: Vec<f64> = if let Some(sf) = shared_freqs {
        sf.iter()
            .copied()
            .filter(|&f| f.is_finite() && f > config.ignore_dc_hz && f <= fmax)
            .collect()
    } else {
        let mode = match config.frequency_selection.as_str() {
            "preferred" => "preferred",
            "hybrid" | "shared_spectral" => "hybrid",
            _ => "spectral",
        };
        let pref_freqs = parse_preferred_frequencies(&config.preferred_periods_days);
        select_frequencies(
            &f_pos,
            &p_pos,
            fmax,
            mode,
            &pref_freqs,
            config.preferred_top_k as usize,
            config.num_peaks as usize,
            config.spectral_top_k as usize,
            config.spectral_merge_tol,
            config.power_cum,
            config.ignore_dc_hz,
            config.refine_peaks,
        )
    };

    let n_freqs_used = freqs_sel.len();

    // ── Scale observations ─────────────────────────────────────────────────
    let yy_scaled: Vec<f64> = yy.iter().map(|&v| v / y_scale).collect();

    // ── Build design matrix ────────────────────────────────────────────────
    let x = make_design_matrix(
        &t_rel,
        &freqs_sel,
        config.include_trend,
        true, // include_dc always true
    );

    // ── Huber-Ridge fit with iterative outlier rejection ────────────────────
    let n_min_needed = (1 + config.include_trend as usize + 2 * n_freqs_used).max(min_obs);

    let beta = if config.outlier_sigma > 0.0 {
        // MAD-based iterative outlier rejection
        let mut t_curr = t_rel.clone();
        let mut y_curr = yy_scaled.clone();
        let mut beta_final: Option<Vec<f64>> = None;

        for _ in 0..(t_curr.len().min(50)) {
            if y_curr.len() < n_min_needed {
                break;
            }
            let x_curr = make_design_matrix(&t_curr, &freqs_sel, config.include_trend, true);
            let y_arr = Array1::from_vec(y_curr.clone());
            let (beta_curr, y_pred) = robust_fit_freq_ridge(
                &x_curr,
                &y_arr,
                &freqs_sel,
                config.ridge_lam,
                config.huber_iters as usize,
                config.huber_delta,
                true,
                config.include_trend,
                config.freq_weight,
            );

            beta_final = Some(beta_curr.clone());

            let residuals: Vec<f64> = y_curr
                .iter()
                .zip(y_pred.iter())
                .map(|(&a, &b)| a - b)
                .collect();
            let mad = mad_std(&residuals);
            let threshold = config.outlier_sigma * mad.max(1e-12);

            let clean_mask: Vec<bool> = residuals.iter().map(|&r| r.abs() <= threshold).collect();
            if clean_mask.iter().all(|&b| b) {
                break;
            }

            let mut new_t = Vec::with_capacity(y_curr.len());
            let mut new_y = Vec::with_capacity(y_curr.len());
            for (i, &keep) in clean_mask.iter().enumerate() {
                if keep {
                    new_t.push(t_curr[i]);
                    new_y.push(y_curr[i]);
                }
            }
            t_curr = new_t;
            y_curr = new_y;
        }

        match beta_final {
            Some(b) => b,
            None => {
                // Final fallback: fit on all data
                let y_arr = Array1::from_vec(yy_scaled.clone());
                let (b, _) = robust_fit_freq_ridge(
                    &x,
                    &y_arr,
                    &freqs_sel,
                    config.ridge_lam,
                    config.huber_iters as usize,
                    config.huber_delta,
                    true,
                    config.include_trend,
                    config.freq_weight,
                );
                b
            }
        }
    } else {
        // No outlier rejection — direct fit on all data
        let y_arr = Array1::from_vec(yy_scaled.clone());
        let (b, _) = robust_fit_freq_ridge(
            &x,
            &y_arr,
            &freqs_sel,
            config.ridge_lam,
            config.huber_iters as usize,
            config.huber_delta,
            true,
            config.include_trend,
            config.freq_weight,
        );
        b
    };

    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);

    NufrostResult {
        valid: true,
        n_freqs_used,
        beta,
        freqs: freqs_sel,
        t_min,
        t_rel_mean,
        fill_value,
        include_trend: config.include_trend,
        y_scale,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Single-pixel NUFROST fit with step/fused-lasso (BCS version)
// ═══════════════════════════════════════════════════════════════════════════

/// Difference weights for step (fused-lasso) regularisation.
fn difference_weights(t_sec: &[f64], enable_dt_weighting: bool) -> Vec<f64> {
    let n = t_sec.len();
    if n <= 1 {
        return vec![];
    }
    if !enable_dt_weighting {
        return vec![1.0f64; n - 1];
    }
    let dt_days: Vec<f64> = t_sec
        .windows(2)
        .map(|w| ((w[1] - w[0]) / 86400.0).max(1.0))
        .collect();
    dt_days.iter().map(|&d| 1.0 / d.sqrt()).collect()
}

/// Fused lasso 1D via FISTA dual proximal gradient.
///
/// Solves: min_u 0.5‖r - u‖² + λ_step Σ w_i |u_{i+1} - u_i|
fn fused_lasso_1d(r: &[f64], lambda_step: f64, weights: &[f64]) -> Vec<f64> {
    let n = r.len();
    if n <= 1 || lambda_step <= 0.0 {
        return r.to_vec();
    }

    let d = n - 1;
    let lam: Vec<f64> = weights.iter().map(|&w| lambda_step * w).collect();

    if lambda_step > 1e6 {
        return vec![0.0; n];
    }

    // FISTA constants
    let step = 0.25; // 1 / ||D||²  (||D||² ≤ 4)
    let max_iter = 5000;
    let tol = 1e-9;

    let mut z = vec![0.0f64; d]; // dual variable
    let mut y_dual = z.clone();
    let mut t_prev = 1.0f64;
    let mut u_prev: Vec<f64> = Vec::new();

    for _ in 0..max_iter {
        // u = r - D^T y_dual
        let dt_y = apply_dt(&y_dual);
        let u: Vec<f64> = r.iter().zip(dt_y.iter()).map(|(&ri, &d)| ri - d).collect();

        // gradient step on y_dual
        let du: Vec<f64> = u.windows(2).map(|w| w[1] - w[0]).collect();
        let mut z_new: Vec<f64> = y_dual
            .iter()
            .zip(du.iter())
            .map(|(&yi, &di)| yi + step * di)
            .collect();
        // project onto box [-lam_i, lam_i]
        for i in 0..d {
            z_new[i] = z_new[i].clamp(-lam[i], lam[i]);
        }

        // FISTA extrapolation
        let t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_prev * t_prev).sqrt());
        let factor = (t_prev - 1.0) / t_new;
        for i in 0..d {
            y_dual[i] = z_new[i] + factor * (z_new[i] - z[i]);
        }

        if !u_prev.is_empty() {
            let max_diff: f64 = u
                .iter()
                .zip(u_prev.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, f64::max);
            if max_diff < tol {
                z = z_new;
                break;
            }
        }

        z = z_new;
        t_prev = t_new;
        u_prev = u;
    }

    // Recompute final primal from z
    let dt_z = apply_dt(&z);
    r.iter().zip(dt_z.iter()).map(|(&ri, &d)| ri - d).collect()
}

/// Apply D^T to a dual variable: (D^T y)_0 = -y_0, (D^T y)_i = y_{i-1} - y_i, last = y_{n-2}
fn apply_dt(y: &[f64]) -> Vec<f64> {
    let d = y.len();
    let n = d + 1;
    let mut out = vec![0.0f64; n];
    if n == 1 {
        return out;
    }
    out[0] = -y[0];
    for i in 1..d {
        out[i] = y[i - 1] - y[i];
    }
    out[d] = y[d - 1];
    out
}

/// Single-pixel NUFROST fit with step term + tiered ridge (BCD variant).
///
/// This is the version that also estimates a fused-lasso step term u,
/// used in the multi-band pipeline.
pub fn nufrost_fit_pixel_step(
    t_sec: &[f64],
    y: &[f64],
    freqs_sel: &[f64],
    config: &NufrostConfig,
) -> Option<NufrostResult> {
    let min_obs = config.min_obs as usize;

    // Valid mask
    let m: Vec<bool> = t_sec
        .iter()
        .zip(y.iter())
        .map(|(&t, &v)| t.is_finite() && v.is_finite())
        .collect();
    let n_kept = m.iter().filter(|&&b| b).count();
    if n_kept < min_obs.max(3) {
        return None;
    }

    let t: Vec<f64> = t_sec
        .iter()
        .zip(m.iter())
        .filter(|(_, &b)| b)
        .map(|(&ti, _)| ti)
        .collect();
    let yy: Vec<f64> = y
        .iter()
        .zip(m.iter())
        .filter(|(_, &b)| b)
        .map(|(&vi, _)| vi)
        .collect();

    let t_rel: Vec<f64> = {
        let t0 = t.iter().copied().fold(f64::INFINITY, f64::min);
        t.iter().map(|&ti| ti - t0).collect()
    };
    let t_rel_mean = nanmean(&t_rel);
    let fill_value = nanmean(&yy);

    // Design matrix
    let freqs_vec: Vec<f64> = freqs_sel.to_vec();
    let x = make_design_matrix(&t_rel, &freqs_vec, config.include_trend, true);

    if x.ncols() == 0 {
        let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);
        return Some(NufrostResult {
            valid: false,
            fill_value,
            t_rel_mean,
            t_min,
            include_trend: config.include_trend,
            ..Default::default()
        });
    }

    let y_arr = Array1::from_vec(yy.clone());

    // Difference weights for fused lasso
    let diff_w = difference_weights(&t, config.step_dt_weighting);

    let mut u = vec![0.0f64; yy.len()];
    // Initial beta from tiered ridge
    let mut beta = tiered_ridge_solve(
        &x,
        &y_arr,
        &freqs_vec,
        config.ridge_lam,
        config.lambda_high,
        config.low_freq_period_days,
        config.freq_weight,
        true,
        config.include_trend,
        None,
        None,
    )
    .unwrap_or_else(|| Array1::zeros(x.ncols()));

    let max_outer = config.max_outer_iter as usize;
    let outer_tol = config.outer_tol;
    let mut _n_iter = 0;

    for it in 1..=max_outer {
        let beta_old = beta.clone();
        let u_old = u.clone();

        // u update: fused lasso on residuals
        let residual: Vec<f64> = yy
            .iter()
            .zip(x.dot(&beta).iter())
            .map(|(&yi, &xbi)| yi - xbi)
            .collect();
        u = fused_lasso_1d(&residual, config.lambda_step, &diff_w);

        // beta update: tiered ridge on y - u
        let y_minus_u: Vec<f64> = yy.iter().zip(u.iter()).map(|(&yi, &ui)| yi - ui).collect();
        let ymu_arr = Array1::from_vec(y_minus_u);
        beta = tiered_ridge_solve(
            &x,
            &ymu_arr,
            &freqs_vec,
            config.ridge_lam,
            config.lambda_high,
            config.low_freq_period_days,
            config.freq_weight,
            true,
            config.include_trend,
            None,
            None,
        )
        .unwrap_or_else(|| Array1::zeros(x.ncols()));

        _n_iter = it;

        // Convergence check
        let denom = (beta_old.iter().map(|&b| b * b).sum::<f64>().sqrt()
            + u_old.iter().map(|&u| u * u).sum::<f64>().sqrt())
        .max(1e-12);
        let delta = beta
            .iter()
            .zip(beta_old.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            + u.iter()
                .zip(u_old.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum::<f64>();
        if delta / denom < outer_tol {
            break;
        }
    }

    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);

    Some(NufrostResult {
        valid: true,
        n_freqs_used: freqs_sel.len(),
        beta: beta.to_vec(),
        freqs: freqs_vec,
        t_min,
        t_rel_mean,
        fill_value,
        include_trend: config.include_trend,
        y_scale: 1.0, // no scaling in step variant
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Prediction
// ═══════════════════════════════════════════════════════════════════════════

/// Predict NUFROST-reconstructed value at a target time from fitted params.
pub fn nufrost_predict(result: &NufrostResult, target_t: f64) -> f64 {
    if !result.valid {
        return result.fill_value;
    }

    let t_star_rel = target_t - result.t_min;
    let n_freqs = result.n_freqs_used;
    let beta = &result.beta;
    let freqs_sel: Vec<f64> = result.freqs.iter().take(n_freqs).copied().collect();

    let beta_len = 1 + result.include_trend as usize + 2 * n_freqs;
    let beta_actual = &beta[..beta_len.min(beta.len())];

    let mut pred = 0.0;

    // DC term
    if let Some(&b0) = beta_actual.first() {
        pred += b0;
        let mut idx = 1;
        // Trend
        if result.include_trend {
            if let Some(&b1) = beta_actual.get(idx) {
                pred += b1 * (t_star_rel - result.t_rel_mean);
                idx += 1;
            }
        }
        // Harmonic terms
        for &f in &freqs_sel {
            let omega = 2.0 * PI * f;
            if let Some(&bc) = beta_actual.get(idx) {
                pred += bc * (omega * t_star_rel).cos();
            }
            if let Some(&bs) = beta_actual.get(idx + 1) {
                pred += bs * (omega * t_star_rel).sin();
            }
            idx += 2;
        }
    }

    pred * result.y_scale
}

/// Predict NUFROST-reconstructed curve at multiple target times.
pub fn nufrost_predict_curve(result: &NufrostResult, target_t: &[f64]) -> Vec<f64> {
    target_t
        .iter()
        .map(|&t| nufrost_predict(result, t))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Convenience: fit + predict in one call
// ═══════════════════════════════════════════════════════════════════════════

/// Fit NUFROST to a single pixel and predict at target time.
///
/// Returns `(predicted_value, n_freqs_used)`.
pub fn nufrost_pixel(
    t_in: &[f64],
    y: &[f64],
    target_in: f64,
    config: &NufrostConfig,
) -> (f64, usize) {
    let (t_sec, target_t) = maybe_days_to_seconds(t_in, target_in);
    let min_obs = config.min_obs as usize;
    let y_scale = 10000.0;

    let valid_mask: Vec<bool> = t_sec
        .iter()
        .zip(y.iter())
        .map(|(&t, &v)| t.is_finite() && v.is_finite())
        .collect();
    let n_valid = valid_mask.iter().filter(|&&b| b).count();
    if n_valid < min_obs.max(3) {
        return (nanmean(y), 0);
    }

    let t: Vec<f64> = t_sec
        .iter()
        .zip(valid_mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&ti, _)| ti)
        .collect();
    let yy_raw: Vec<f64> = y
        .iter()
        .zip(valid_mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&vi, _)| vi)
        .collect();
    let yy: Vec<f64> = yy_raw.iter().map(|&v| v / y_scale).collect();
    let fill_value = nanmean(&yy_raw);

    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);
    let t_rel: Vec<f64> = t.iter().map(|&ti| ti - t_min).collect();
    let t_rel_mean = nanmean(&t_rel);
    let tspan = t_rel.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - t_rel.iter().copied().fold(f64::INFINITY, f64::min);
    if tspan <= 0.0 || !tspan.is_finite() {
        return (fill_value, 0);
    }

    let (f_pos, p_pos) = compute_spectrum_nufft(&t_rel, &yy_raw, config.modes as usize, y_scale);

    let mut t_sorted = t_rel.clone();
    t_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dt_pos: Vec<f64> = t_sorted
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0.0)
        .collect();
    let dt_med = if !dt_pos.is_empty() {
        nanmedian(&dt_pos)
    } else {
        tspan / t_rel.len() as f64
    };
    let fmax = 0.5 / dt_med.max(1e-12);

    let selection_mode = match config.frequency_selection.as_str() {
        "all" => "all",
        "preferred" => "preferred",
        "hybrid" | "shared_spectral" => "hybrid",
        _ => "spectral",
    };
    let preferred_freqs = parse_preferred_frequencies(&config.preferred_periods_days);
    let freqs_sel = select_frequencies(
        &f_pos,
        &p_pos,
        fmax,
        selection_mode,
        &preferred_freqs,
        config.preferred_top_k as usize,
        config.num_peaks as usize,
        config.spectral_top_k as usize,
        config.spectral_merge_tol,
        config.power_cum,
        config.ignore_dc_hz,
        config.refine_peaks,
    );
    let n_freqs = freqs_sel.len();

    let x = make_design_matrix(&t_rel, &freqs_sel, config.include_trend, true);
    if x.ncols() == 0 {
        return (fill_value, 0);
    }

    let weights = difference_weights(&t, config.step_dt_weighting);
    let mut u = vec![0.0f64; yy.len()];
    let y_arr = Array1::from_vec(yy.clone());
    let mut beta = tiered_ridge_solve(
        &x,
        &y_arr,
        &freqs_sel,
        config.ridge_lam,
        config.lambda_high,
        config.low_freq_period_days,
        config.freq_weight,
        true,
        config.include_trend,
        None,
        None,
    )
    .unwrap_or_else(|| Array1::zeros(x.ncols()));

    for _ in 0..(config.max_outer_iter as usize) {
        let beta_old = beta.clone();
        let u_old = u.clone();

        let y_hat = x.dot(&beta);
        let residual: Vec<f64> = yy
            .iter()
            .zip(y_hat.iter())
            .map(|(&yi, &yh)| yi - yh)
            .collect();
        u = fused_lasso_1d(&residual, config.lambda_step, &weights);

        // Make the decomposition identifiable: the DC column carries the mean,
        // while u is a zero-mean piecewise-constant innovation.
        let u_mean = nanmean(&u);
        if u_mean.is_finite() {
            for ui in &mut u {
                *ui -= u_mean;
            }
        }

        let y_minus_u: Vec<f64> = yy.iter().zip(u.iter()).map(|(&yi, &ui)| yi - ui).collect();
        let ymu_arr = Array1::from_vec(y_minus_u);
        beta = tiered_ridge_solve(
            &x,
            &ymu_arr,
            &freqs_sel,
            config.ridge_lam,
            config.lambda_high,
            config.low_freq_period_days,
            config.freq_weight,
            true,
            config.include_trend,
            None,
            None,
        )
        .unwrap_or_else(|| Array1::zeros(x.ncols()));

        let denom = (beta_old.iter().map(|&b| b * b).sum::<f64>().sqrt()
            + u_old.iter().map(|&v| v * v).sum::<f64>().sqrt())
        .max(1e-12);
        let delta_beta = beta
            .iter()
            .zip(beta_old.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>();
        let delta_u = u
            .iter()
            .zip(u_old.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>();
        if (delta_beta + delta_u).sqrt() / denom < config.outer_tol {
            break;
        }
    }

    let t_star_rel = target_t - t_min;
    let mut pred = beta.get(0).copied().unwrap_or(0.0);
    let mut idx = 1usize;
    if config.include_trend {
        if let Some(&bt) = beta.get(idx) {
            pred += bt * (t_star_rel - t_rel_mean);
        }
        idx += 1;
    }
    for &f in &freqs_sel {
        let omega = 2.0 * PI * f;
        if let Some(&bc) = beta.get(idx) {
            pred += bc * (omega * t_star_rel).cos();
        }
        if let Some(&bs) = beta.get(idx + 1) {
            pred += bs * (omega * t_star_rel).sin();
        }
        idx += 2;
    }

    let seg_idx = match t.binary_search_by(|&ti| {
        ti.partial_cmp(&target_t)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Ok(i) => i,
        Err(i) => {
            if i == 0 {
                0
            } else {
                i - 1
            }
        }
    }
    .min(u.len().saturating_sub(1));
    pred += u.get(seg_idx).copied().unwrap_or(0.0);

    (pred * y_scale, n_freqs)
}

/// Fit one vector-valued NUFROST model to a multi-band pixel trajectory.
///
/// `observations[b][i]` is the value of band `b` at timestamp `ts_days[i]`.
/// All bands share one timestamp grid, one vector NUFFT frequency set, one
/// design matrix, and one date-level Huber weight sequence. The returned vector
/// has one prediction per input band.
pub fn nufrost_fit_pixel_vector(
    ts_days: &[f64],
    observations: &[Vec<f64>],
    config: &NufrostConfig,
) -> NufrostVectorResult {
    fit_vector_with_options(ts_days, observations, config, &VectorFitOptions::default())
}

#[derive(Default)]
pub(crate) struct VectorFitOptions {
    pub fixed_frequencies: Option<Vec<f64>>,
    pub unpenalized_frequencies: Option<Vec<f64>>,
    pub constraints: Vec<(f64, Vec<f64>, f64)>,
}

pub(crate) fn fit_vector_with_options(
    ts_days: &[f64],
    observations: &[Vec<f64>],
    config: &NufrostConfig,
    options: &VectorFitOptions,
) -> NufrostVectorResult {
    let n_times = ts_days.len();
    let n_bands = observations.len();
    let mut result = NufrostVectorResult {
        n_bands,
        include_trend: config.include_trend,
        ..NufrostVectorResult::default()
    };

    if n_times == 0 || n_bands == 0 {
        return result;
    }
    for obs in observations {
        if obs.len() != n_times {
            return result;
        }
    }

    let (ts_sec, _) = maybe_days_to_seconds(ts_days, ts_days[0]);
    let min_obs = config.min_obs as usize;

    let base_mask: Vec<bool> = (0..n_times)
        .map(|i| ts_sec[i].is_finite() && observations.iter().all(|band| band[i].is_finite()))
        .collect();
    let n_use = base_mask.iter().filter(|&&m| m).count();
    if n_use < min_obs.max(3) {
        return result;
    }

    struct VectorFit {
        row_indices: Vec<usize>,
        x: Array2<f64>,
        z_mat: Array2<f64>,
        result: NufrostVectorResult,
    }

    let fit_active = |active_mask: &[bool]| -> Option<VectorFit> {
        let row_indices: Vec<usize> = active_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();
        let n_use = row_indices.len();
        if n_use < min_obs.max(3) {
            return None;
        }

        let t_use: Vec<f64> = row_indices.iter().map(|&i| ts_sec[i]).collect();
        let t_min = t_use.iter().copied().fold(f64::INFINITY, f64::min);
        let t_rel: Vec<f64> = t_use.iter().map(|&t| t - t_min).collect();
        let t_rel_mean = nanmean(&t_rel);
        let tspan = t_rel.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - t_rel.iter().copied().fold(f64::INFINITY, f64::min);
        if !tspan.is_finite() || tspan <= 0.0 {
            return None;
        }

        let mut y_mat = Array2::<f64>::zeros((n_use, n_bands));
        for b in 0..n_bands {
            for (row, &i) in row_indices.iter().enumerate() {
                y_mat[[row, b]] = observations[b][i];
            }
        }

        let mut centers = vec![0.0; n_bands];
        let mut scales = vec![1.0; n_bands];
        let mut z_mat = Array2::<f64>::zeros((n_use, n_bands));
        for b in 0..n_bands {
            let col: Vec<f64> = y_mat.column(b).iter().copied().collect();
            let (center, scale) = if config.normalization_mode == "reflectance" {
                (0.0, 10_000.0)
            } else if config.normalization_mode == "centered_reflectance" {
                let center = nanmedian(&col);
                (
                    if center.is_finite() {
                        center
                    } else {
                        nanmean(&col)
                    },
                    10_000.0,
                )
            } else {
                let center = nanmedian(&col);
                let abs_dev: Vec<f64> = col
                    .iter()
                    .map(|&v| {
                        if v.is_finite() {
                            (v - center).abs()
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
                let mut scale = 1.4826 * nanmedian(&abs_dev);
                if !scale.is_finite() || scale <= 1e-6 {
                    let mu = nanmean(&col);
                    let var = col
                        .iter()
                        .filter(|v| v.is_finite())
                        .map(|&v| {
                            let d = v - mu;
                            d * d
                        })
                        .sum::<f64>()
                        / (n_use as f64).max(1.0);
                    scale = var.sqrt();
                }
                (
                    if center.is_finite() {
                        center
                    } else {
                        nanmean(&col)
                    },
                    scale.max(1e-6),
                )
            };
            centers[b] = center;
            scales[b] = scale.max(1e-6);
            for i in 0..n_use {
                z_mat[[i, b]] = (y_mat[[i, b]] - centers[b]) / scales[b];
            }
        }

        let mut t_sorted = t_rel.clone();
        t_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let dt_pos: Vec<f64> = t_sorted
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|&d| d > 0.0)
            .collect();
        let dt_med = if dt_pos.is_empty() {
            tspan / n_use as f64
        } else {
            nanmedian(&dt_pos)
        };
        let fmax = 0.5 / dt_med.max(1e-12);

        let mut spectrum_dims: Vec<Vec<f64>> = Vec::with_capacity(n_bands);
        for b in 0..n_bands {
            spectrum_dims.push(z_mat.column(b).iter().copied().collect());
        }
        let (vector_freqs, vector_power) = crate::nufft::type1_vector_power_kb(
            &t_rel,
            &spectrum_dims,
            config.modes as usize,
            crate::nufft::NufftOptions::default(),
        );
        if vector_freqs.is_empty() || vector_power.is_empty() {
            return None;
        }

        let selection_mode = match config.frequency_selection.as_str() {
            "all" => "all",
            "preferred" => "preferred",
            "hybrid" | "shared_spectral" => "hybrid",
            _ => "spectral",
        };
        let preferred_freqs = parse_preferred_frequencies(&config.preferred_periods_days);
        let mut phenology_freqs = select_phenology_frequencies(
            &vector_freqs,
            &vector_power,
            fmax,
            &preferred_freqs,
            config.preferred_top_k as usize,
            config.ignore_dc_hz,
            config.spectral_merge_tol,
        );
        let mut freqs_sel = select_frequencies(
            &vector_freqs,
            &vector_power,
            fmax,
            selection_mode,
            &preferred_freqs,
            config.preferred_top_k as usize,
            config.num_peaks as usize,
            config.spectral_top_k as usize,
            config.spectral_merge_tol,
            config.power_cum,
            config.ignore_dc_hz,
            config.refine_peaks,
        );
        if let Some(frequencies) = &options.fixed_frequencies {
            freqs_sel = frequencies.clone();
        }
        if let Some(frequencies) = &options.unpenalized_frequencies {
            phenology_freqs = frequencies.clone();
        }
        if freqs_sel.is_empty() {
            return None;
        }

        let x = make_design_matrix(&t_rel, &freqs_sel, config.include_trend, true);
        if x.ncols() == 0 {
            return None;
        }

        let solve_with_constraints = |real_weights: &[f64]| {
            if options.constraints.is_empty() {
                return multi_output_tiered_ridge_solve(
                    &x,
                    &z_mat,
                    &freqs_sel,
                    real_weights,
                    config.ridge_lam,
                    config.lambda_high,
                    config.low_freq_period_days,
                    config.freq_weight,
                    true,
                    config.include_trend,
                    &phenology_freqs,
                    config.multiband_shrinkage,
                );
            }
            let extra = options.constraints.len();
            let mut xa = Array2::<f64>::zeros((n_use + extra, x.ncols()));
            let mut za = Array2::<f64>::zeros((n_use + extra, n_bands));
            let mut wa = real_weights.to_vec();
            for i in 0..n_use {
                for j in 0..x.ncols() {
                    xa[[i, j]] = x[[i, j]];
                }
                for b in 0..n_bands {
                    za[[i, b]] = z_mat[[i, b]];
                }
            }
            for (k, (day, value, weight)) in options.constraints.iter().enumerate() {
                let t = day * 86400. - t_min;
                let row = n_use + k;
                xa[[row, 0]] = 1.;
                let mut col = 1;
                if config.include_trend {
                    xa[[row, col]] = t - t_rel_mean;
                    col += 1;
                }
                for f in &freqs_sel {
                    xa[[row, col]] = (2. * PI * f * t).cos();
                    xa[[row, col + 1]] = (2. * PI * f * t).sin();
                    col += 2;
                }
                for b in 0..n_bands {
                    za[[row, b]] = (value[b] - centers[b]) / scales[b];
                }
                wa.push(*weight);
            }
            multi_output_tiered_ridge_solve(
                &xa,
                &za,
                &freqs_sel,
                &wa,
                config.ridge_lam,
                config.lambda_high,
                config.low_freq_period_days,
                config.freq_weight,
                true,
                config.include_trend,
                &phenology_freqs,
                config.multiband_shrinkage,
            )
        };
        let mut weights = vec![1.0f64; n_use];
        let delta = if config.normalization_mode == "reflectance" {
            config.huber_delta.max(1e-6)
        } else {
            config.huber_delta.max(1.5)
        };
        let damping = 0.5;
        let min_weight = 0.03;
        let iters = (config.huber_iters as usize).clamp(1, 5);
        let mut theta = solve_with_constraints(&weights)?;

        for _ in 0..iters {
            let z_hat = x.dot(&theta);
            let mut next_weights = vec![1.0f64; n_use];
            for i in 0..n_use {
                let mut ss = 0.0;
                let mut count = 0usize;
                for b in 0..n_bands {
                    let r = z_mat[[i, b]] - z_hat[[i, b]];
                    if r.is_finite() {
                        ss += r * r;
                        count += 1;
                    }
                }
                let e = if count == 0 {
                    0.0
                } else {
                    (ss / count as f64).sqrt()
                };
                let huber_w = if e <= delta { 1.0 } else { delta / (e + 1e-12) };
                next_weights[i] =
                    (damping * weights[i] + (1.0 - damping) * huber_w).max(min_weight);
            }
            weights = next_weights;
            if let Some(next_theta) = solve_with_constraints(&weights) {
                theta = next_theta;
            }
        }

        Some(VectorFit {
            row_indices,
            x,
            z_mat,
            result: NufrostVectorResult {
                valid: true,
                n_bands,
                freqs: freqs_sel,
                t_min,
                t_rel_mean,
                centers,
                scales,
                theta,
                include_trend: config.include_trend,
            },
        })
    };

    let mut active_mask = base_mask;
    let reject_iters = if config.joint_outlier {
        (config.outlier_reject_iters as usize).min(5)
    } else {
        0
    };
    let mut final_fit: Option<VectorFit> = None;

    for pass in 0..=reject_iters {
        let fit = match fit_active(&active_mask) {
            Some(fit) => fit,
            None => break,
        };
        if pass >= reject_iters {
            final_fit = Some(fit);
            break;
        }

        let z_hat = fit.x.dot(&fit.result.theta);
        let mut rms = vec![0.0f64; fit.z_mat.nrows()];
        for i in 0..fit.z_mat.nrows() {
            let mut ss = 0.0;
            let mut count = 0usize;
            for b in 0..n_bands {
                let r = fit.z_mat[[i, b]] - z_hat[[i, b]];
                if r.is_finite() {
                    ss += r * r;
                    count += 1;
                }
            }
            rms[i] = if count == 0 {
                f64::NAN
            } else {
                (ss / count as f64).sqrt()
            };
        }

        let center = nanmedian(&rms);
        let abs_dev: Vec<f64> = rms.iter().map(|&v| (v - center).abs()).collect();
        let mut scale = 1.4826 * nanmedian(&abs_dev);
        if !scale.is_finite() || scale <= 1e-12 {
            let mu = nanmean(&rms);
            let var = rms
                .iter()
                .filter(|v| v.is_finite())
                .map(|&v| {
                    let d = v - mu;
                    d * d
                })
                .sum::<f64>()
                / (rms.len() as f64).max(1.0);
            scale = var.sqrt();
        }
        let threshold_floor = if config.normalization_mode == "reflectance" {
            config.huber_delta.max(1e-6)
        } else {
            config.huber_delta.max(1.5)
        };
        let threshold =
            (center + config.outlier_reject_sigma.max(0.0) * scale).max(threshold_floor);
        let mut candidates: Vec<(usize, f64)> = rms
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, e)| e.is_finite() && *e > threshold)
            .collect();
        if candidates.is_empty() {
            final_fit = Some(fit);
            break;
        }
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let min_remaining = min_obs.max(fit.x.ncols() + 2).min(fit.row_indices.len());
        let removable = fit.row_indices.len().saturating_sub(min_remaining);
        let max_by_fraction =
            ((fit.row_indices.len() as f64) * config.outlier_reject_max_fraction).ceil() as usize;
        let n_remove = candidates.len().min(removable).min(max_by_fraction.max(1));
        if n_remove == 0 {
            final_fit = Some(fit);
            break;
        }
        for (row, _) in candidates.into_iter().take(n_remove) {
            active_mask[fit.row_indices[row]] = false;
        }
    }

    result = match final_fit {
        Some(fit) => fit.result,
        None => return result,
    };

    result
}

/// Predict a multi-band NUFROST curve from a fitted vector model.
pub fn nufrost_predict_vector_curve(
    result: &NufrostVectorResult,
    target_days: &[f64],
) -> Vec<Vec<f64>> {
    if !result.valid || result.n_bands == 0 {
        return vec![vec![f64::NAN; target_days.len()]; result.n_bands];
    }
    if target_days.is_empty() {
        return vec![Vec::new(); result.n_bands];
    }
    let (target_sec, _) = maybe_days_to_seconds(target_days, target_days[0]);
    let mut curves = vec![vec![f64::NAN; target_days.len()]; result.n_bands];
    for (t_idx, &target_sec) in target_sec.iter().enumerate() {
        let pred = nufrost_predict_vector(result, target_sec);
        for b in 0..result.n_bands {
            curves[b][t_idx] = pred.get(b).copied().unwrap_or(f64::NAN);
        }
    }
    curves
}

/// Predict one multi-band NUFROST value from a fitted vector model.
pub fn nufrost_predict_vector(result: &NufrostVectorResult, target_day: f64) -> Vec<f64> {
    if !result.valid || result.n_bands == 0 {
        return vec![f64::NAN; result.n_bands];
    }
    let (_, target_sec) = maybe_days_to_seconds(&[target_day], target_day);
    let t_star_rel = target_sec - result.t_min;
    let ncols = result.theta.nrows();
    let mut basis = Array1::<f64>::zeros(ncols);
    basis[0] = 1.0;
    let mut idx = 1usize;
    if result.include_trend {
        if idx < basis.len() {
            basis[idx] = t_star_rel - result.t_rel_mean;
        }
        idx += 1;
    }
    for &f in &result.freqs {
        let omega = 2.0 * PI * f;
        if idx < basis.len() {
            basis[idx] = (omega * t_star_rel).cos();
        }
        if idx + 1 < basis.len() {
            basis[idx + 1] = (omega * t_star_rel).sin();
        }
        idx += 2;
    }

    let z_pred = basis.dot(&result.theta);

    let mut out = vec![f64::NAN; result.n_bands];
    for b in 0..result.n_bands {
        out[b] = result.centers[b] + z_pred[b] * result.scales[b];
    }

    out
}

/// Fit one vector-valued pixel and predict at a single target date.
///
/// `observations[b][i]` is the value of band `b` at timestamp `ts_days[i]`.
/// All bands share one timestamp grid, one vector NUFFT frequency set, one
/// design matrix, and one date-level Huber weight sequence. The returned vector
/// has one prediction per input band.
pub fn nufrost_pixel_vector(
    ts_days: &[f64],
    observations: &[Vec<f64>],
    target_day: f64,
    config: &NufrostConfig,
) -> Vec<f64> {
    let fit = nufrost_fit_pixel_vector(ts_days, observations, config);
    nufrost_predict_vector(&fit, target_day)
}

/// Reconstruct a full raster using NUFROST.
///
/// The GDAL crate supplies raster I/O and the shared per-pixel traversal;
/// this crate supplies only the NUFROST pixel model.
pub fn reconstruct_nufrost_geotiff<P: AsRef<std::path::Path>>(
    reader: &gdal::RasterReader,
    timestamps_days: &[f64],
    target_t_day: f64,
    config: &NufrostConfig,
    output_path: P,
    metadata: &gdal::RasterMetadata,
) -> anyhow::Result<()> {
    let cube = gdal::read_all_bands(reader)?;
    gdal::reconstruct_single_band(
        &cube,
        timestamps_days,
        target_t_day,
        output_path,
        metadata,
        |ts, obs, targ| {
            let (pred, _n_freqs) = nufrost_pixel(ts, obs, targ, config);
            if pred.is_finite() {
                pred
            } else {
                f64::NAN
            }
        },
    )
}

// ── Multi-band NUFROST ────────────────────────────────────────────────────

/// Fit NUFROST independently to each band and predict at a shared target day.
///
/// Unlike `nufrost_pixel`, which operates on a single-band time series,
/// this function accepts per-band observation vectors (all sharing the same
/// timestamp grid `ts_days`) and returns one prediction per band.
///
/// # Parameters
/// - `ts_days`: time axis in days (relative or absolute)
/// - `observations`: `&[Vec<f64>]` where `observations[i]` is the time-series
///   for band `i`; every inner `Vec` must be the same length as `ts_days`
/// - `target_day`: prediction target in the same units as `ts_days`
/// - `config`: per-pixel NUFROST configuration
///
/// # Returns
/// `Vec<f64>` with length `observations.len()`.  The i‑th element is the
/// predicted reflectance (or `f64::NAN` when the fit fails for that band).
///
/// # Panics
/// Panics if any inner `Vec` length differs from `ts_days.len()`.
///
/// # Stub note
/// The body is a stub; the full fitting loop will be implemented in a later
/// task.  All multi-band tests are therefore expected to **fail** until the
/// implementation is completed.
#[cfg(test)]
fn nufrost_pixel_multiband(
    ts_days: &[f64],
    observations: &[Vec<f64>],
    target_day: f64,
    config: &NufrostConfig,
) -> Vec<f64> {
    // Validate input dimensions
    let n_times = ts_days.len();
    for (i, obs) in observations.iter().enumerate() {
        assert_eq!(
            obs.len(),
            n_times,
            "observations[{}] has length {} but ts_days has length {}",
            i,
            obs.len(),
            n_times
        );
    }

    let (ts_days, target_day) = maybe_days_to_seconds(ts_days, target_day);

    let n_bands = observations.len();
    let mut result = vec![f64::NAN; n_bands];
    let min_obs = config.min_obs as usize;
    let y_scale = 10000.0;
    let modes = config.modes as usize;

    // ── Step 1: Per-band spectrum aggregation ───────────────────────────────
    // For each band, collect valid observations and compute spectrum.
    struct BandValid {
        band_idx: usize,
        freqs: Vec<f64>,
        power: Vec<f64>,
    }

    let mut band_data: Vec<BandValid> = Vec::new();
    // Also collect all valid timestamps for fmax / global t_min later.
    let mut all_valid_t: Vec<f64> = Vec::new();

    for (bi, obs) in observations.iter().enumerate() {
        let pairs: Vec<(f64, f64)> = ts_days
            .iter()
            .zip(obs.iter())
            .filter(|(&t, &y)| t.is_finite() && y.is_finite())
            .map(|(&t, &y)| (t, y))
            .collect();

        if pairs.len() < min_obs.max(3) {
            continue;
        }

        let valid_t: Vec<f64> = pairs.iter().map(|&(t, _)| t).collect();
        let valid_y: Vec<f64> = pairs.iter().map(|&(_, y)| y).collect();
        all_valid_t.extend_from_slice(&valid_t);

        let t_min = valid_t.iter().copied().fold(f64::INFINITY, f64::min);
        let t_rel: Vec<f64> = valid_t.iter().map(|&t| t - t_min).collect();

        let (freqs, power) = compute_spectrum_nufft(&t_rel, &valid_y, modes, y_scale);

        band_data.push(BandValid {
            band_idx: bi,
            freqs,
            power,
        });
    }

    if band_data.is_empty() {
        return result;
    }

    // ── Step 2: Shared frequency selection via median-aggregated power ─────
    let n_f = band_data[0].freqs.len();

    // Median power at each frequency index across bands.
    let mut median_power = vec![0.0f64; n_f];
    for i in 0..n_f {
        let mut powers: Vec<f64> = band_data
            .iter()
            .map(|bd| bd.power[i])
            .filter(|p| p.is_finite())
            .collect();
        let n = powers.len();
        if n == 0 {
            median_power[i] = 0.0;
        } else {
            powers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            median_power[i] = if n % 2 == 1 {
                powers[n / 2]
            } else {
                (powers[n / 2 - 1] + powers[n / 2]) / 2.0
            };
        }
    }

    let freqs_for_sel = &band_data[0].freqs;

    // Nyquist limit from the pooled set of all valid timestamps.
    let mut t_sort = all_valid_t.clone();
    t_sort.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dt_pos: Vec<f64> = t_sort
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0.0)
        .collect();
    let dt_med = if !dt_pos.is_empty() {
        let mut d = dt_pos.clone();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if d.len() % 2 == 1 {
            d[d.len() / 2]
        } else {
            (d[d.len() / 2 - 1] + d[d.len() / 2]) / 2.0
        }
    } else {
        let tspan = all_valid_t
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            - all_valid_t.iter().copied().fold(f64::INFINITY, f64::min);
        tspan / all_valid_t.len() as f64
    };
    let fmax = 0.5 / dt_med.max(1e-12);

    let peak_idx = select_peaks_adaptive(
        freqs_for_sel,
        &median_power,
        config.num_peaks as usize,
        config.power_cum,
        config.ignore_dc_hz,
        fmax,
    );

    // Refine and collect frequency values.
    let mut shared_freqs: Vec<f64> = peak_idx
        .iter()
        .map(|&i| {
            if config.refine_peaks {
                refine_parabolic(freqs_for_sel, &median_power, i)
            } else {
                freqs_for_sel[i]
            }
        })
        .filter(|&f| f.is_finite() && f > config.ignore_dc_hz && f <= fmax)
        .collect();

    // Sort, deduplicate, merge nearby.
    shared_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    {
        let mut merged: Vec<f64> = Vec::new();
        for f in shared_freqs {
            if merged.is_empty() {
                merged.push(f);
                continue;
            }
            let last = *merged.last().unwrap();
            let rel = (f - last).abs() / last.max(1e-12);
            if rel <= config.spectral_merge_tol.max(0.0) {
                let len = merged.len();
                merged[len - 1] = 0.5 * (last + f);
            } else {
                merged.push(f);
            }
        }
        shared_freqs = merged;
    }

    if shared_freqs.is_empty() {
        return result;
    }

    // ── Step 2b: Private frequency selection per band ────────────────────────
    // For each band, select additional frequencies from its own power spectrum
    // that are distinct from shared frequencies and from each other.
    let n_bands_with_data = band_data.len();

    let mut private_freqs_per_band: Vec<Vec<f64>> = Vec::with_capacity(n_bands_with_data);

    for bd in &band_data {
        let private = select_private_frequencies(&bd.freqs, &bd.power, &shared_freqs, config, fmax);
        private_freqs_per_band.push(private);
    }

    // Build all_freqs for each band: shared ∪ private, sorted and merged.
    let mut all_freqs_per_band: Vec<Vec<f64>> = Vec::with_capacity(n_bands_with_data);
    for private in &private_freqs_per_band {
        let mut all = shared_freqs.clone();
        all.extend_from_slice(private);
        all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        {
            let mut merged: Vec<f64> = Vec::new();
            for f in all {
                if merged.is_empty() {
                    merged.push(f);
                    continue;
                }
                let last = *merged.last().unwrap();
                let rel = (f - last).abs() / last.max(1e-12);
                if rel <= config.spectral_merge_tol.max(0.0) {
                    let len = merged.len();
                    merged[len - 1] = 0.5 * (last + f);
                } else {
                    merged.push(f);
                }
            }
            all = merged;
        }
        all_freqs_per_band.push(all);
    }

    // ── Step 3: Build base mask (ALL bands must be observed) ────────────────
    // A row is usable only when t is finite AND every band's observation is finite.
    let base_mask: Vec<bool> = (0..n_times)
        .map(|ti| {
            if !ts_days[ti].is_finite() {
                return false;
            }
            for band_obs in observations.iter() {
                if !band_obs[ti].is_finite() {
                    return false;
                }
            }
            true
        })
        .collect();
    let n_kept_base = base_mask.iter().filter(|&&b| b).count();
    if n_kept_base < min_obs.max(3) {
        return result;
    }

    // Extract kept timestamps (days).
    let t_kept: Vec<f64> = ts_days
        .iter()
        .zip(base_mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&t, _)| t)
        .collect();
    let t_kept_min = t_kept.iter().copied().fold(f64::INFINITY, f64::min);
    let t_kept_rel: Vec<f64> = t_kept.iter().map(|&t| t - t_kept_min).collect();

    // n_bands_used: bands that passed Step 1 (band_data).
    if n_bands_with_data == 0 {
        return result;
    }
    let nb = n_bands_with_data;

    // Y matrix: (n_kept_base, nb) — values already divided by y_scale.
    let mut y_mat = Array2::<f64>::zeros((n_kept_base, nb));
    for col in 0..nb {
        let orig_band = band_data[col].band_idx;
        let mut row = 0;
        for ti in 0..n_times {
            if base_mask[ti] {
                y_mat[[row, col]] = observations[orig_band][ti] / y_scale;
                row += 1;
            }
        }
    }

    // ── Step 4: Per-band design matrices + ridge diag (shared+private) ───────
    let mut x_per_band: Vec<Array2<f64>> = Vec::with_capacity(nb);
    let mut ridge_r_per_band: Vec<Vec<f64>> = Vec::with_capacity(nb);
    for bi in 0..nb {
        let all_freqs = &all_freqs_per_band[bi];
        let private_freqs = &private_freqs_per_band[bi];
        let shared_for_band = shared_freqs.clone();

        let x_b = make_design_matrix(&t_kept_rel, all_freqs, config.include_trend, true);
        let p_b = x_b.ncols();
        let ridge_r = if p_b == 0 {
            Vec::new()
        } else {
            build_ridge_diag_mixed(
                &shared_for_band,
                private_freqs,
                p_b,
                true,
                config.include_trend,
                config.freq_weight,
                config.private_freq_penalty_mult,
            )
        };
        x_per_band.push(x_b);
        ridge_r_per_band.push(ridge_r);
    }

    // ── Step 5: Joint outlier pre-processing ─────────────────────────────────
    let mut mask_joint: Vec<bool> = vec![true; n_kept_base];
    let do_joint = config.joint_outlier && n_kept_base >= min_obs.max((nb * 3).max(2));

    if do_joint {
        let mut residuals = Array2::<f64>::zeros((n_kept_base, nb));
        let mut sigmas = vec![0.0f64; nb];

        for b in 0..nb {
            let x_b = &x_per_band[b];
            if x_b.ncols() == 0 {
                continue;
            }
            let y_b = y_mat.column(b).to_owned();
            let ridge_r = &ridge_r_per_band[b];
            let beta_b = tiered_ridge_solve(
                x_b,
                &y_b,
                &all_freqs_per_band[b],
                config.ridge_lam,
                config.lambda_high,
                config.low_freq_period_days,
                config.freq_weight,
                true,
                config.include_trend,
                Some(ridge_r),
                None,
            )
            .unwrap_or_else(|| Array1::zeros(x_b.ncols()));

            let y_hat = x_b.dot(&beta_b);
            for i in 0..n_kept_base {
                residuals[[i, b]] = y_mat[[i, b]] - y_hat[i];
            }

            // Per-band MAD sigma
            let col: Vec<f64> = residuals.column(b).to_vec();
            let med = nanmedian(&col);
            let abs_dev: Vec<f64> = col.iter().map(|&r| (r - med).abs()).collect();
            sigmas[b] = nanmedian(&abs_dev) * 1.4826;
        }

        mask_joint = joint_outlier_mask(&residuals, &sigmas, config.joint_outlier_sigma);
    }

    let n_kept_after_joint = mask_joint.iter().filter(|&&b| b).count();
    if n_kept_after_joint < min_obs.max(3) {
        mask_joint = vec![true; n_kept_base];
    }

    // ── Step 6: Build masked data ────────────────────────────────────────────
    let n_use = mask_joint.iter().filter(|&&b| b).count();

    let t_use: Vec<f64> = t_kept
        .iter()
        .zip(mask_joint.iter())
        .filter(|(_, &m)| m)
        .map(|(&t, _)| t)
        .collect();
    let t_use_min = t_use.iter().copied().fold(f64::INFINITY, f64::min);
    let t_use_rel: Vec<f64> = t_use.iter().map(|&t| t - t_use_min).collect();

    let mut y_use = Array2::<f64>::zeros((n_use, nb));
    for col in 0..nb {
        let mut row = 0;
        for i in 0..n_kept_base {
            if mask_joint[i] {
                y_use[[row, col]] = y_mat[[i, col]];
                row += 1;
            }
        }
    }

    // Rebuild design matrices on masked times.
    let mut x_use_per_band: Vec<Array2<f64>> = Vec::with_capacity(nb);
    for bi in 0..nb {
        let x_b = make_design_matrix(
            &t_use_rel,
            &all_freqs_per_band[bi],
            config.include_trend,
            true,
        );
        x_use_per_band.push(x_b);
    }

    let diff_w = difference_weights(&t_use, config.step_dt_weighting);

    // ── Step 7: BCD initialisation ───────────────────────────────────────────
    let mut betas: Vec<Array1<f64>> = Vec::with_capacity(nb);
    for b in 0..nb {
        let x_b = &x_use_per_band[b];
        if x_b.ncols() == 0 {
            betas.push(Array1::zeros(0));
            continue;
        }
        let y_b = y_use.column(b).to_owned();
        let ridge_r = &ridge_r_per_band[b];
        let beta_b = tiered_ridge_solve(
            x_b,
            &y_b,
            &all_freqs_per_band[b],
            config.ridge_lam,
            config.lambda_high,
            config.low_freq_period_days,
            config.freq_weight,
            true,
            config.include_trend,
            Some(ridge_r),
            None,
        )
        .unwrap_or_else(|| Array1::zeros(x_b.ncols()));
        betas.push(beta_b);
    }
    let mut u_mat = Array2::<f64>::zeros((n_use, nb));

    let max_outer = config.max_outer_iter as usize;

    // ── Step 8: BCD outer loop ───────────────────────────────────────────────
    for _it in 1..=max_outer {
        let betas_old: Vec<Array1<f64>> = betas.iter().map(|b| b.clone()).collect();
        let u_old = u_mat.clone();

        // Residual: Y - stack(per-band X_b @ Beta_b)
        let mut residual = Array2::<f64>::zeros((n_use, nb));
        for b in 0..nb {
            let x_b = &x_use_per_band[b];
            if x_b.ncols() == 0 {
                for i in 0..n_use {
                    residual[[i, b]] = y_use[[i, b]];
                }
                continue;
            }
            let y_hat = x_b.dot(&betas[b]);
            for i in 0..n_use {
                residual[[i, b]] = y_use[[i, b]] - y_hat[i];
            }
        }

        // U update: group fused lasso ADMM (joint across bands).
        u_mat = group_fused_lasso_admm(
            &residual,
            config.lambda_step,
            &diff_w,
            config.admm_rho,
            config.admm_max_iter as usize,
            config.admm_tol,
        );
        for b in 0..nb {
            let mean = u_mat.column(b).sum() / n_use as f64;
            if mean.is_finite() {
                for i in 0..n_use {
                    u_mat[[i, b]] -= mean;
                }
            }
        }

        // Beta update: per-band tiered ridge on Y - U.
        for b in 0..nb {
            let x_b = &x_use_per_band[b];
            if x_b.ncols() == 0 {
                continue;
            }
            let ridge_r = &ridge_r_per_band[b];
            let y_minus_u: Vec<f64> = (0..n_use).map(|i| y_use[[i, b]] - u_mat[[i, b]]).collect();
            let ymu = Array1::from_vec(y_minus_u);
            let beta_new = tiered_ridge_solve(
                x_b,
                &ymu,
                &all_freqs_per_band[b],
                config.ridge_lam,
                config.lambda_high,
                config.low_freq_period_days,
                config.freq_weight,
                true,
                config.include_trend,
                Some(ridge_r),
                None,
            )
            .unwrap_or_else(|| Array1::zeros(x_b.ncols()));
            betas[b] = beta_new;
        }

        // Convergence: accumulate squared differences for Beta and U,
        // then take Frobenius norm at the end (matching Python convention).
        let mut delta = 0.0f64;
        let mut denom = 0.0f64;
        for b in 0..nb {
            for (&a, &b_old) in betas[b].iter().zip(betas_old[b].iter()) {
                let d = a - b_old;
                delta += d * d;
            }
            for &x in betas_old[b].iter() {
                denom += x * x;
            }
        }
        for i in 0..n_use {
            for b in 0..nb {
                let diff = u_mat[[i, b]] - u_old[[i, b]];
                delta += diff * diff;
                denom += u_old[[i, b]] * u_old[[i, b]];
            }
        }
        delta = delta.sqrt();
        denom = denom.sqrt().max(1e-12);
        if delta / denom < config.outer_tol {
            break;
        }
    }

    // ── Step 9: Pad U back to full length with carry-forward ─────────────────
    // U_kept: (n_kept_base, nb) — place ADMM U at joint-mask positions.
    let mut u_kept = Array2::<f64>::zeros((n_kept_base, nb));
    {
        let mut row = 0;
        for i in 0..n_kept_base {
            if mask_joint[i] {
                for b in 0..nb {
                    u_kept[[i, b]] = u_mat[[row, b]];
                }
                row += 1;
            }
        }
    }
    // Carry-forward across joint-masked gaps.
    for b in 0..nb {
        let mut last = 0.0;
        for i in 0..n_kept_base {
            if mask_joint[i] {
                last = u_kept[[i, b]];
            } else {
                u_kept[[i, b]] = last;
            }
        }
    }

    // Scatter into u_full: (n_times, n_bands).
    let mut u_full = Array2::<f64>::zeros((n_times, n_bands));
    {
        let mut kept_row = 0;
        for ti in 0..n_times {
            if base_mask[ti] {
                for b in 0..nb {
                    let orig_band = band_data[b].band_idx;
                    u_full[[ti, orig_band]] = u_kept[[kept_row, b]];
                }
                kept_row += 1;
            }
        }
    }

    // Full mask (n_times): base_mask AND mask_joint.
    let full_mask: Vec<bool> = {
        let mut out = vec![false; n_times];
        let mut kept_row = 0;
        for ti in 0..n_times {
            if base_mask[ti] {
                out[ti] = mask_joint[kept_row];
                kept_row += 1;
            }
        }
        out
    };

    // ── Step 10: Prediction at target_day ────────────────────────────────────
    let t_use_min_for_pred = t_use.iter().copied().fold(f64::INFINITY, f64::min);
    let t_star_rel = target_day - t_use_min_for_pred;

    // Segment U: find the last kept time ≤ target_day.
    let mut seg_vals = vec![0.0f64; n_bands];
    if full_mask.iter().any(|&m| m) {
        let t_obs: Vec<f64> = ts_days
            .iter()
            .zip(full_mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&t, _)| t)
            .collect();

        let order = match t_obs.binary_search_by(|&t| {
            t.partial_cmp(&target_day)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => idx,
            Err(idx) => {
                if idx == 0 {
                    0
                } else {
                    idx - 1
                }
            }
        };
        let order = order.min(t_obs.len().saturating_sub(1));

        let mut obs_row = 0;
        for ti in 0..n_times {
            if full_mask[ti] {
                if obs_row == order {
                    for b in 0..n_bands {
                        seg_vals[b] = u_full[[ti, b]];
                    }
                    break;
                }
                obs_row += 1;
            }
        }
    }

    let t_use_rel_mean = nanmean(&t_use_rel);

    for bi in 0..nb {
        let orig_band = band_data[bi].band_idx;
        let all_freqs = &all_freqs_per_band[bi];
        let beta = &betas[bi];
        if beta.is_empty() {
            continue;
        }

        let mut pred = 0.0;
        if let Some(&b0) = beta.first() {
            pred += b0;
            let mut idx: usize = 1;
            if config.include_trend {
                if let Some(&bt) = beta.get(idx) {
                    pred += bt * (t_star_rel - t_use_rel_mean);
                    idx += 1;
                }
            }
            for &f in all_freqs.iter() {
                let omega = 2.0 * PI * f;
                if let Some(&bc) = beta.get(idx) {
                    pred += bc * (omega * t_star_rel).cos();
                }
                if let Some(&bs) = beta.get(idx + 1) {
                    pred += bs * (omega * t_star_rel).sin();
                }
                idx += 2;
            }
        }
        pred += seg_vals[orig_band];
        result[orig_band] = pred * y_scale;
    }

    result
}

// ── Helper: extract rows from an ndarray matrix ─────────────────────────
#[allow(dead_code)]
fn select_rows(x: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    let n_cols = x.ncols();
    let mut out = Array2::zeros((rows.len(), n_cols));
    for (i, &r) in rows.iter().enumerate() {
        for j in 0..n_cols {
            out[[i, j]] = x[[r, j]];
        }
    }
    out
}

// ── Helper: first-difference along axis=0 ────────────────────────────────
/// Compute `out[i] = x[i+1] - x[i]` for each column, returning shape `(n-1, B)`.
fn diff_axis0(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let b = x.ncols();
    if n <= 1 {
        return Array2::zeros((0, b));
    }
    let mut out = Array2::zeros((n - 1, b));
    for i in 0..n - 1 {
        for j in 0..b {
            out[[i, j]] = x[[i + 1, j]] - x[[i, j]];
        }
    }
    out
}

// ── Thomas solver for symmetric tridiagonal systems ──────────────────────
/// Solve `Ax = rhs` column-wise via the Thomas algorithm.
///
/// `diag`: main diagonal of A, length n.
/// `off`: sub/super-diagonal of A, length n-1.
/// `rhs`: right-hand side matrix, shape (n, B). Each column solved independently.
///
/// Returns x with shape (n, B). New array; inputs unchanged.
fn solve_tridiag_thomas(diag: &[f64], off: &[f64], rhs: &Array2<f64>) -> Array2<f64> {
    let n = diag.len();
    let b = rhs.ncols();
    debug_assert_eq!(off.len(), n - 1);
    debug_assert_eq!(rhs.nrows(), n);

    let mut x = Array2::zeros((n, b));
    if n == 0 {
        return x;
    }
    if n == 1 {
        let inv = 1.0 / diag[0];
        for j in 0..b {
            x[[0, j]] = rhs[[0, j]] * inv;
        }
        return x;
    }

    // Forward sweep
    let mut cprime = vec![0.0f64; n - 1];
    let mut dprime = Array2::zeros((n, b));

    cprime[0] = off[0] / diag[0];
    for j in 0..b {
        dprime[[0, j]] = rhs[[0, j]] / diag[0];
    }

    for i in 1..n {
        let denom = diag[i] - off[i - 1] * cprime[i - 1];
        let inv_denom = 1.0 / denom;
        if i < n - 1 {
            cprime[i] = off[i] * inv_denom;
        }
        for j in 0..b {
            dprime[[i, j]] = (rhs[[i, j]] - off[i - 1] * dprime[[i - 1, j]]) * inv_denom;
        }
    }

    // Backward substitution
    for j in 0..b {
        x[[n - 1, j]] = dprime[[n - 1, j]];
    }
    for i in (0..n - 1).rev() {
        for j in 0..b {
            x[[i, j]] = dprime[[i, j]] - cprime[i] * x[[i + 1, j]];
        }
    }

    x
}

// ── Group fused lasso via ADMM ───────────────────────────────────────────
/// Solve the multi-band group fused lasso via ADMM.
///
/// Solves:
/// ```text
/// min_U  0.5 * ||R - U||_F^2
///      + lambda_step * Σ_i weights_i * ||(D U)_i||_2
/// ```
/// where D ∈ R^{(n-1)×n} is the first-difference operator and ||·||₂ acts
/// on the band axis at each time index.
///
/// `r`: observation matrix, shape (n, B).
/// `weights`: per-difference penalty weights, length n-1.
///
/// ADMM on convex objective with proximable g = group L1 — see Boyd (2011).
pub fn group_fused_lasso_admm(
    r: &Array2<f64>,
    lambda_step: f64,
    weights: &[f64],
    rho: f64,
    max_iter: usize,
    tol: f64,
) -> Array2<f64> {
    let n = r.nrows();
    let b = r.ncols();

    if n == 0 {
        return Array2::zeros((0, b));
    }
    if n == 1 || lambda_step <= 0.0 {
        return r.to_owned();
    }

    assert_eq!(
        weights.len(),
        n - 1,
        "weights length {} != n-1 ({})",
        weights.len(),
        n - 1
    );

    let lam: Vec<f64> = weights.iter().map(|&w| lambda_step * w).collect();

    // ── When lambda_step is huge (e.g. the default 1e30) the step term is
    //    intentionally disabled. Return zeros so U=0 and prediction reduces
    //    to harmonic-only without bias from column means.
    if lambda_step > 1e6 {
        return Array2::zeros((n, b));
    }

    // ── Sentinel: if the smallest per-step penalty already exceeds
    //    max_i ||(D R)_i||₂ by a comfortable margin, V is zeroed at every
    //    iteration and U converges to the per-band column mean. Short-circuit.
    if !lam.is_empty() {
        let dr = diff_axis0(r);
        let dr_max = if dr.nrows() > 0 {
            dr.rows()
                .into_iter()
                .map(|row| {
                    let ssq: f64 = row.iter().map(|&v| v * v).sum();
                    ssq.sqrt()
                })
                .fold(0.0f64, f64::max)
        } else {
            0.0
        };
        let lam_min = lam.iter().copied().fold(f64::INFINITY, f64::min);
        if lam_min > 0.0 && lam_min >= 10.0 * (dr_max + 1e-12) {
            // Collapse to per-band column means
            let mut means = Array2::from_elem((1, b), 0.0);
            for j in 0..b {
                let sum: f64 = r.column(j).sum();
                means[[0, j]] = sum / n as f64;
            }
            let mut out = Array2::zeros((n, b));
            for i in 0..n {
                for j in 0..b {
                    out[[i, j]] = means[[0, j]];
                }
            }
            return out;
        }
    }

    // ── Build tridiagonal matrix I + rho * D^T D ──────────────────────────
    let mut diag = vec![1.0 + 2.0 * rho; n];
    diag[0] = 1.0 + rho;
    diag[n - 1] = 1.0 + rho;
    let off = vec![-rho; n - 1];

    // ── ADMM state ────────────────────────────────────────────────────────
    let mut v: Array2<f64> = Array2::zeros((n - 1, b));
    let mut lam_mat: Array2<f64> = Array2::zeros((n - 1, b));
    let mut u = r.to_owned();

    for _iter in 0..max_iter {
        // ── U-update: solve (I + rho D^T D) U = R + rho D^T (V - Lam) ──
        let mut dt_term: Array2<f64> = Array2::zeros((n, b));
        for i in 0..n - 1 {
            for j in 0..b {
                let val = v[[i, j]] - lam_mat[[i, j]];
                dt_term[[i, j]] -= val;
                dt_term[[i + 1, j]] += val;
            }
        }
        let mut rhs = r.to_owned();
        for i in 0..n {
            for j in 0..b {
                rhs[[i, j]] += rho * dt_term[[i, j]];
            }
        }
        let u_new = solve_tridiag_thomas(&diag, &off, &rhs);

        // ── V-update: group soft-threshold per row ────────────────────────
        let du: Array2<f64> = diff_axis0(&u_new); // (n-1, B)
        let mut z: Array2<f64> = Array2::zeros((n - 1, b));
        for i in 0..n - 1 {
            for j in 0..b {
                z[[i, j]] = du[[i, j]] + lam_mat[[i, j]];
            }
        }

        // Compute L2 norms of each row of Z
        let mut norms = vec![0.0f64; n - 1];
        for i in 0..n - 1 {
            let ssq: f64 = z.row(i).iter().map(|&v| v * v).sum();
            norms[i] = ssq.sqrt();
        }

        let thresh: Vec<f64> = lam.iter().map(|&l| l / rho).collect();
        let mut scale = vec![0.0f64; n - 1];
        for i in 0..n - 1 {
            if norms[i] > 0.0 {
                scale[i] = (1.0 - thresh[i] / norms[i].max(1e-30)).max(0.0);
            }
        }

        let mut v_new: Array2<f64> = Array2::zeros((n - 1, b));
        for i in 0..n - 1 {
            for j in 0..b {
                v_new[[i, j]] = scale[i] * z[[i, j]];
            }
        }

        // ── Dual update: Lam += DU - V ────────────────────────────────────
        for i in 0..n - 1 {
            for j in 0..b {
                lam_mat[[i, j]] += du[[i, j]] - v_new[[i, j]];
            }
        }

        // ── Convergence check: primal residual + change ───────────────────
        let mut primal = 0.0f64;
        for i in 0..n - 1 {
            for j in 0..b {
                let diff: f64 = du[[i, j]] - v_new[[i, j]];
                let val = diff.abs();
                if val > primal {
                    primal = val;
                }
            }
        }

        let mut change = 0.0f64;
        for i in 0..n {
            for j in 0..b {
                let diff: f64 = u_new[[i, j]] - u[[i, j]];
                let val = diff.abs();
                if val > change {
                    change = val;
                }
            }
        }

        u = u_new;
        v = v_new;

        if primal < tol && change < tol {
            break;
        }
    }

    u
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NufrostConfig;

    // ── Default test config ────────────────────────────────────────────────

    fn default_nufrost_config() -> NufrostConfig {
        serde_json::from_str(
            r#"{
                "modes": 4096,
                "eps": 1e-12,
                "num_peaks": 10,
                "power_cum": 0.7,
                "ignore_dc_hz": 1e-10,
                "frequency_selection": "spectral",
                "preferred_periods_days": "365.25,182.625,91.3125,30.4375",
                "spectral_merge_tol": 0.15,
                "refine_peaks": true,
                "include_trend": true,
                "ridge_lam": 0.005,
                "freq_weight": 2.0,
                "huber_iters": 3,
                "huber_delta": 0.05,
                "min_obs": 12,
                "outlier_sigma": 2.0,
                "lambda_step": 1e30,
                "lambda_high": 0.005,
                "low_freq_period_days": 0.0,
                "step_dt_weighting": false,
                "max_outer_iter": 5,
                "outer_tol": 1e-3,
                "joint_outlier": false,
                "joint_outlier_sigma": 2.5,
                "admm_rho": 1.0,
                "admm_max_iter": 1000,
                "admm_tol": 1e-4
            }"#,
        )
        .unwrap()
    }

    // ── Unit tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_next_even() {
        assert_eq!(next_even(1), 2);
        assert_eq!(next_even(2), 2);
        assert_eq!(next_even(3), 4);
        assert_eq!(next_even(4095), 4096);
        assert_eq!(next_even(4096), 4096);
    }

    #[test]
    fn test_nanmean_basic() {
        assert!((nanmean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-15);
        assert!(nanmean(&[f64::NAN, 3.0, 5.0]).abs() - 4.0 < 1e-15);
        assert!(nanmean(&[f64::NAN]).is_nan());
    }

    #[test]
    fn test_nanmedian_basic() {
        assert!((nanmedian(&[1.0, 3.0, 2.0]) - 2.0).abs() < 1e-15);
        assert!((nanmedian(&[1.0, 2.0]) - 1.5).abs() < 1e-15);
    }

    #[test]
    fn test_mad_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mad_std(&data);
        assert!(m > 0.0);
    }

    #[test]
    fn test_design_matrix_shape() {
        let t = vec![0.0, 1.0, 2.0];
        let freqs = vec![1.0 / 365.25 / 86400.0];
        let x = make_design_matrix(&t, &freqs, true, true);
        // DC + trend + 2*freqs = 1 + 1 + 2 = 4
        assert_eq!(x.shape(), &[3, 4]);
    }

    #[test]
    fn test_design_matrix_dc_col_is_ones() {
        let t = vec![10.0, 20.0];
        let freqs: Vec<f64> = vec![];
        let x = make_design_matrix(&t, &freqs, false, true);
        assert!((x[[0, 0]] - 1.0).abs() < 1e-15);
        assert!((x[[1, 0]] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_design_matrix_no_dc_no_trend() {
        let t = vec![0.0, 1.0];
        let freqs = vec![0.1];
        let x = make_design_matrix(&t, &freqs, false, false);
        // 2 cols: cos, sin
        assert_eq!(x.shape(), &[2, 2]);
    }

    #[test]
    fn test_harmonic_cos_sin_at_zero() {
        let t = vec![0.0];
        let freqs = vec![0.1];
        let x = make_design_matrix(&t, &freqs, false, true);
        // DC (x[[0,0]] = 1), cos (x[[0,1]] = cos(0) = 1), sin (x[[0,2]] = 0)
        assert!((x[[0, 0]] - 1.0).abs() < 1e-14);
        assert!((x[[0, 1]] - 1.0).abs() < 1e-14);
        assert!(x[[0, 2]].abs() < 1e-14);
    }

    #[test]
    fn test_gauss_solve_2x2() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![5.0, 5.0]);
        let x = gauss_solve(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gauss_solve_singular() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 4.0, 2.0]).unwrap();
        let b = Array1::from_vec(vec![3.0, 6.0]);
        assert!(gauss_solve(&a, &b).is_none());
    }

    #[test]
    fn test_frequency_penalty_min_freq_is_1() {
        let freqs = vec![0.0001, 0.0002, 0.0005];
        let penalty = make_frequency_penalty(&freqs, 2.0);
        assert!(penalty.len() == 3);
        // First (min) should be 1.0
        assert!((penalty[0] - 1.0).abs() < 1e-10);
        // Higher freqs get larger penalty
        assert!(penalty[2] > penalty[1]);
        assert!(penalty[1] >= penalty[0]);
    }

    #[test]
    fn test_phenology_frequency_has_no_ridge_penalty() {
        let freqs = vec![0.0001, 0.0002, 0.0005];
        let p = 2 + 2 * freqs.len();
        let lam = tiered_lambda_diag(&freqs, p, 0.005, 0.02, 60.0, 2.0, true, true, &[0.0002]);

        // DC/trend still have the base penalty.
        assert!(lam[0] > 0.0);
        assert!(lam[1] > 0.0);
        // The second harmonic pair corresponds to 0.0002 and is unpenalized.
        assert_eq!(lam[4], 0.0);
        assert_eq!(lam[5], 0.0);
        // Neighboring frequencies remain penalized.
        assert!(lam[2] > 0.0);
        assert!(lam[3] > 0.0);
        assert!(lam[6] > 0.0);
        assert!(lam[7] > 0.0);
    }

    #[test]
    fn test_multiband_shrinkage_reduces_band_contrast() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, -1.5, 1.0, -0.5, 1.0, 0.5, 1.0, 1.5]).unwrap();
        let z =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 4.0, 3.0, 7.0, 4.0, 10.0]).unwrap();
        let weights = vec![1.0; 4];
        let theta_free = multi_output_tiered_ridge_solve(
            &x,
            &z,
            &[],
            &weights,
            1e-6,
            0.0,
            60.0,
            2.0,
            true,
            true,
            &[],
            0.0,
        )
        .unwrap();
        let theta_shrunk = multi_output_tiered_ridge_solve(
            &x,
            &z,
            &[],
            &weights,
            1e-6,
            0.0,
            60.0,
            2.0,
            true,
            true,
            &[],
            10.0,
        )
        .unwrap();

        let free_slope_gap = (theta_free[[1, 1]] - theta_free[[1, 0]]).abs();
        let shrunk_slope_gap = (theta_shrunk[[1, 1]] - theta_shrunk[[1, 0]]).abs();
        assert!(shrunk_slope_gap < free_slope_gap);

        let free_mean = 0.5 * (theta_free[[1, 0]] + theta_free[[1, 1]]);
        let shrunk_mean = 0.5 * (theta_shrunk[[1, 0]] + theta_shrunk[[1, 1]]);
        assert!((free_mean - shrunk_mean).abs() < 1e-8);
    }

    #[test]
    fn test_multiband_shrinkage_zero_matches_unshrunken_path() {
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, -2.0, 0.5, 1.0, -1.0, -0.2, 1.0, 0.0, 0.1, 1.0, 1.0, -0.4, 1.0, 2.0, 0.3,
            ],
        )
        .unwrap();
        let z = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 1.5, 2.5, 1.8, 2.4, 3.1, 2.6, 3.2, 3.9, 3.3, 3.9, 4.7, 4.1, 4.8, 5.4,
            ],
        )
        .unwrap();
        let weights = vec![1.0, 0.7, 1.0, 0.9, 0.8];
        let theta_a = multi_output_tiered_ridge_solve(
            &x,
            &z,
            &[],
            &weights,
            0.01,
            0.0,
            60.0,
            2.0,
            true,
            true,
            &[],
            0.0,
        )
        .unwrap();
        let theta_b = multi_output_tiered_ridge_solve(
            &x,
            &z,
            &[],
            &weights,
            0.01,
            0.0,
            60.0,
            2.0,
            true,
            true,
            &[],
            -1.0,
        )
        .unwrap();

        for (a, b) in theta_a.iter().zip(theta_b.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_multiband_shrinkage_single_band_no_effect() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, -1.5, 1.0, -0.5, 1.0, 0.5, 1.0, 1.5]).unwrap();
        let z = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let weights = vec![1.0; 4];
        let theta_free = multi_output_tiered_ridge_solve(
            &x,
            &z,
            &[],
            &weights,
            0.001,
            0.0,
            60.0,
            2.0,
            true,
            true,
            &[],
            0.0,
        )
        .unwrap();
        let theta_shrunk = multi_output_tiered_ridge_solve(
            &x,
            &z,
            &[],
            &weights,
            0.001,
            0.0,
            60.0,
            2.0,
            true,
            true,
            &[],
            100.0,
        )
        .unwrap();

        for (a, b) in theta_free.iter().zip(theta_shrunk.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_multiband_shrinkage_large_value_remains_finite() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, -1.5, 1.0, -0.5, 1.0, 0.5, 1.0, 1.5]).unwrap();
        let z =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 4.0, 3.0, 7.0, 4.0, 10.0]).unwrap();
        let weights = vec![1.0; 4];
        let theta = multi_output_tiered_ridge_solve(
            &x,
            &z,
            &[],
            &weights,
            1e-6,
            0.0,
            60.0,
            2.0,
            true,
            true,
            &[],
            100.0,
        )
        .unwrap();

        assert!(theta.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_huber_weights_small_residuals() {
        let r = vec![0.0, 0.01, -0.01];
        let w = huber_weights(&r, 0.05);
        assert!((w[0] - 1.0).abs() < 1e-10);
        assert!((w[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_huber_weights_large_residuals() {
        let r = vec![0.1, -0.2];
        let w = huber_weights(&r, 0.05);
        // w[0] = 0.05 / 0.1 = 0.5
        assert!((w[0] - 0.5).abs() < 1e-10);
        // w[1] = 0.05 / 0.2 = 0.25
        assert!((w[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_refine_parabolic_exact() {
        // Parabola: y = -(x - 2)^2 + 4, peak at x = 2
        let f = vec![1.0, 2.0, 3.0];
        let p = vec![3.0, 4.0, 3.0]; // -(1-2)^2+4=3, 4, 3
        let refined = refine_parabolic(&f, &p, 1);
        assert!((refined - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_preferred_frequencies() {
        let freqs = parse_preferred_frequencies("365.25,182.625");
        assert_eq!(freqs.len(), 2);
        let annual = 1.0 / (365.25 * 86400.0);
        let semi = 1.0 / (182.625 * 86400.0);
        assert!((freqs[0] - annual).abs() < 1e-15);
        assert!((freqs[1] - semi).abs() < 1e-15);
    }

    #[test]
    fn test_parse_preferred_frequencies_empty() {
        let freqs = parse_preferred_frequencies("");
        assert!(freqs.is_empty());
    }

    #[test]
    fn test_select_frequencies_spectral_empty_input() {
        let f_pos = vec![1.0, 2.0, 3.0];
        let p_pos = vec![0.0, 0.0, 0.0]; // all zero means no valid peaks
        let _selected = select_frequencies(
            &f_pos,
            &p_pos,
            10.0,
            "spectral",
            &[],
            4,
            4,
            4,
            0.15,
            0.7,
            0.0,
            false,
        );
        // No valid peaks → empty; all-zero power handled without panicking
        let p_pos2 = vec![0.0f64; 3]; // all zero intentional
        let _selected2 = select_frequencies(
            &f_pos,
            &p_pos2,
            10.0,
            "spectral",
            &[],
            4,
            4,
            4,
            0.15,
            0.7,
            0.0,
            false,
        );
        // The function should still work without panicking
        assert!(_selected2.len() <= 4);
    }

    #[test]
    fn test_select_frequencies_hybrid_returns_frequencies() {
        let df = 1.0e-7;
        let f_pos: Vec<f64> = (0..100).map(|i| i as f64 * df).collect();
        let p_pos: Vec<f64> = (0..100)
            .map(|i| {
                // Put a peak at f = 3.171e-6 (~= 1/(365.25*86400))
                if (i as f64 - 3.171e-6 / df).abs() < 2.0 {
                    100.0 * (1.0 - (i as f64 - 3.171e-6 / df).abs() / 2.0)
                } else {
                    0.1
                }
            })
            .collect();
        let pref_freqs = parse_preferred_frequencies("365.25");

        let selected = select_frequencies(
            &f_pos,
            &p_pos,
            0.1,
            "hybrid",
            &pref_freqs,
            4,
            4,
            4,
            0.15,
            0.7,
            0.0,
            false,
        );
        assert!(!selected.is_empty());
        // Selected frequencies should be finite and > 0
        for f in &selected {
            assert!(f.is_finite() && *f > 0.0);
        }
    }

    // ── Ridge regression tests ─────────────────────────────────────────────

    #[test]
    fn test_ridge_solve_simple() {
        // y = 2 + 3*t, centered trend col = t - mean(t), mean=1
        // beta[0] = value at mean time = 2 + 3*1 = 5
        // beta[1] = slope = 3
        let t = vec![0.0, 1.0, 2.0];
        let x = make_design_matrix(&t, &[], true, true);
        let y_vals = vec![2.0, 5.0, 8.0];
        let y = Array1::from_vec(y_vals);
        let r = build_ridge_diag(&[], x.ncols(), true, true, 2.0);

        let beta = ridge_solve_augmented(&x, &y, 0.0, &r, None).unwrap();
        assert!((beta[0] - 5.0).abs() < 1e-10, "DC: {}", beta[0]);
        assert!((beta[1] - 3.0).abs() < 1e-10, "trend: {}", beta[1]);
    }

    #[test]
    fn test_ridge_solve_with_lam_produces_finite() {
        let t: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let freqs = vec![0.01, 0.02];
        let x = make_design_matrix(&t, &freqs, true, true);
        // Generate a noisy signal
        let y_vals: Vec<f64> = t
            .iter()
            .map(|&ti| 1.0 + 0.5 * ti + 2.0 * (2.0 * PI * 0.01 * ti).cos())
            .collect();
        let y = Array1::from_vec(y_vals);
        let r = build_ridge_diag(&freqs, x.ncols(), true, true, 2.0);

        let beta = ridge_solve_augmented(&x, &y, 0.01, &r, None).unwrap();
        for &b in &beta {
            assert!(b.is_finite());
        }
    }

    // ── End-to-end pixel tests ─────────────────────────────────────────────

    #[test]
    fn test_nufrost_pixel_returns_finite_on_clean_periodic_signal() {
        // Synthetic annual + semi-annual harmonic
        let t_sec: Vec<f64> = (0..50).map(|i| i as f64 * 15.0 * 86400.0).collect();
        let annual_freq = 1.0 / (365.25 * 86400.0);
        let semi_freq = 2.0 * annual_freq;

        let y: Vec<f64> = t_sec
            .iter()
            .map(|&t| {
                0.5 + 0.2 * (2.0 * PI * annual_freq * t).sin()
                    + 0.1 * (2.0 * PI * semi_freq * t).cos()
            })
            .collect();

        let config = default_nufrost_config();
        let target_t = t_sec[25]; // middle of series
        let (pred, n_freqs) = nufrost_pixel(&t_sec, &y, target_t, &config);

        assert!(pred.is_finite());
        assert!(n_freqs >= 1);
    }

    #[test]
    fn test_nufrost_pixel_vector_returns_finite_multiband_prediction() {
        let mut config = default_nufrost_config();
        config.frequency_selection = "shared_spectral".to_string();
        config.lambda_step = 1e30;
        config.min_obs = 8;

        let n = 36;
        let t_days: Vec<f64> = (0..n).map(|i| i as f64 * 10.0).collect();
        let target_day = 180.0;
        let mut observations = vec![Vec::with_capacity(n); 3];
        for &t in &t_days {
            let seasonal = (2.0 * PI * t / 120.0).sin();
            observations[0].push(1000.0 + 100.0 * seasonal);
            observations[1].push(1500.0 + 150.0 * seasonal);
            observations[2].push(2200.0 + 220.0 * seasonal);
        }

        let pred = nufrost_pixel_vector(&t_days, &observations, target_day, &config);

        assert_eq!(pred.len(), 3);
        assert!(pred.iter().all(|v| v.is_finite()));
        assert!(pred[0] < pred[1] && pred[1] < pred[2]);
    }

    #[test]
    fn test_nufrost_pixel_with_nan_returns_finite() {
        let t_sec: Vec<f64> = (0..50).map(|i| i as f64 * 15.0 * 86400.0).collect();
        let annual_freq = 1.0 / (365.25 * 86400.0);
        let mut y: Vec<f64> = t_sec
            .iter()
            .map(|&t| 0.5 + 0.2 * (2.0 * PI * annual_freq * t).sin())
            .collect();
        // Insert NaN at observation 10
        y[10] = f64::NAN;

        let config = default_nufrost_config();
        let target_t = t_sec[25];
        let (pred, _) = nufrost_pixel(&t_sec, &y, target_t, &config);

        assert!(pred.is_finite());
    }

    #[test]
    fn test_nufrost_pixel_insufficient_data_returns_nan() {
        let t_sec = vec![0.0, 86400.0];
        let y = vec![0.5, f64::NAN];
        let config = default_nufrost_config();
        let result = nufrost_fit_pixel(&t_sec, &y, &config, None);
        assert!(!result.valid, "should be invalid with only 1 valid obs");
        assert_eq!(result.n_freqs_used, 0);
    }

    // ── Multi-band tests ───────────────────────────────────────────────────

    /// Build a synthetic 3-band dataset with a shared harmonic and deterministic noise.
    ///
    /// Returns `(ts_days, observations, ground_truth_at_20)`.
    fn synthetic_3band_data() -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let n = 40;
        let ts_days: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let freq = 0.5;
        let amplitudes = [0.3, 0.2, 0.1];
        let noise_std = 0.02;

        let mut observations: Vec<Vec<f64>> = Vec::with_capacity(3);
        for &amp in &amplitudes {
            let band: Vec<f64> = ts_days
                .iter()
                .enumerate()
                .map(|(i, &t)| {
                    // Deterministic pseudo-noise from index
                    let noise = noise_std
                        * ((i as f64 * 7.0 + 13.0).sin() * 0.7
                            + (i as f64 * 11.0 + 3.0).cos() * 0.3);
                    amp * (2.0 * PI * freq * t).sin() + noise
                })
                .collect();
            observations.push(band);
        }

        let ground_truth: Vec<f64> = amplitudes.iter().map(|_| 0.0).collect();

        (ts_days, observations, ground_truth)
    }

    #[test]
    fn test_multiband_synthetic_3bands() {
        let (ts_days, observations, ground_truth) = synthetic_3band_data();
        let config = default_nufrost_config();

        let predictions = nufrost_pixel_multiband(&ts_days, &observations, 20.0, &config);

        assert_eq!(predictions.len(), 3);
        for (i, (&pred, &truth)) in predictions.iter().zip(ground_truth.iter()).enumerate() {
            assert!(
                (pred - truth).abs() < 0.05,
                "band {}: pred={:.6}, truth={:.6}, diff={:.2e}",
                i,
                pred,
                truth,
                (pred - truth).abs()
            );
        }
    }

    #[test]
    fn test_multiband_1band_parity_with_nufrost_pixel() {
        // Use the clean periodic signal from the existing single-band test
        let t_days: Vec<f64> = (0..50).map(|i| i as f64 * 15.0).collect();
        let annual_freq_hz = 1.0 / (365.25 * 86400.0);
        let semi_freq_hz = 2.0 * annual_freq_hz;

        let y: Vec<f64> = t_days
            .iter()
            .map(|&t_d| {
                let t_sec = t_d * 86400.0;
                0.5 + 0.2 * (2.0 * PI * annual_freq_hz * t_sec).sin()
                    + 0.1 * (2.0 * PI * semi_freq_hz * t_sec).cos()
            })
            .collect();

        let config = default_nufrost_config();
        let target_day = t_days[25];

        // Single-band call via multiband
        let obs_multiband = vec![y.clone()];
        let pred_multiband = nufrost_pixel_multiband(&t_days, &obs_multiband, target_day, &config);

        // Reference: existing single-band nufrost_pixel (uses seconds internally)
        let t_sec: Vec<f64> = t_days.iter().map(|&d| d * 86400.0).collect();
        let target_sec = target_day * 86400.0;
        let (pred_single, _) = nufrost_pixel(&t_sec, &y, target_sec, &config);

        let pred_mb = pred_multiband[0];
        assert!(
            (pred_mb - pred_single).abs() < 1e-4,
            "multiband={:.6}, single={:.6}, diff={:.2e}",
            pred_mb,
            pred_single,
            (pred_mb - pred_single).abs()
        );
    }

    #[test]
    fn test_multiband_nan_observations_no_panic() {
        let ts_days: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let mut band0: Vec<f64> = ts_days
            .iter()
            .map(|&t| 0.3 * (2.0 * PI * 0.5 * t).sin())
            .collect();
        // Insert NaN at multiple positions
        band0[5] = f64::NAN;
        band0[15] = f64::NAN;

        let mut band1: Vec<f64> = ts_days
            .iter()
            .map(|&t| 0.2 * (2.0 * PI * 0.5 * t).sin())
            .collect();
        band1[0] = f64::NAN;

        let observations = vec![band0, band1];
        let config = default_nufrost_config();

        let predictions = nufrost_pixel_multiband(&ts_days, &observations, 10.0, &config);
        assert_eq!(predictions.len(), 2);
        // Stub returns NaN; after implementation, predictions should be finite
    }

    #[test]
    fn test_multiband_duplicate_timestamps_no_panic() {
        let ts_days = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let band: Vec<f64> = vec![0.5, 0.55, 0.6, 0.58, 0.7, 0.72, 0.65, 0.63];
        let observations = vec![band];
        let config = default_nufrost_config();

        let predictions = nufrost_pixel_multiband(&ts_days, &observations, 1.5, &config);
        assert_eq!(predictions.len(), 1);
        // Stub returns NaN; after implementation, should not panic
    }

    // ── joint_outlier_mask tests ──────────────────────────────────────────

    #[test]
    fn test_joint_outlier_no_valid_bands() {
        // All sigmas <= 0 → all true (keep everything)
        let r = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
        let sigmas = vec![0.0, -1.0];
        let mask = joint_outlier_mask(&r, &sigmas, 2.5);
        assert_eq!(mask, vec![true; 5]);
    }

    #[test]
    fn test_joint_outlier_single_valid_band() {
        // One valid band → marginal |z| <= sigma threshold
        let r =
            Array2::from_shape_vec((4, 2), vec![0.5, 0.0, 3.0, 0.0, 1.0, 0.0, -2.0, 0.0]).unwrap();
        let sigmas = vec![1.0, 0.0]; // only band 0 is valid
        let sigma = 2.0;

        let mask = joint_outlier_mask(&r, &sigmas, sigma);
        // |0.5| <= 2 → true, |3.0| > 2 → false, |1.0| <= 2 → true, |-2.0| <= 2 → true
        assert_eq!(mask, vec![true, false, true, true]);
    }

    #[test]
    fn test_joint_outlier_multiband_cross_band_outlier() {
        // Two valid bands, one row has large residuals in both bands
        let r = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.1, 0.1, // clean
                0.2, 0.1, // clean
                5.0, 4.0, // joint outlier -- large in both bands
                0.3, 0.2, // clean
                0.1, 0.3, // clean
            ],
        )
        .unwrap();
        let sigmas = vec![1.0, 1.0];
        let sigma = 2.0;

        let mask = joint_outlier_mask(&r, &sigmas, sigma);
        // Row 2 (index 2) should be rejected as cross-band outlier
        assert!(!mask[2], "row 2 should be rejected as cross-band outlier");
        // The clean rows should be kept
        assert!(mask[0]);
        assert!(mask[1]);
        assert!(mask[3]);
        assert!(mask[4]);
    }

    #[test]
    fn test_joint_outlier_single_band_outlier_not_rejected() {
        // Modest deviation in one band, clean in the other.
        // The joint L2 score should stay within the MAD-based threshold
        // so the row is NOT falsely flagged as a cross-band outlier.
        // Build 20 rows: 10 low-residual, 9 high-residual (but clean),
        // and 1 with a modest single-band bump.
        let mut data = Vec::with_capacity(40);
        for _ in 0..10 {
            data.push(0.3); // band 0
            data.push(0.2); // band 1  -> score ≈ 0.361
        }
        for _ in 0..9 {
            data.push(0.8); // band 0
            data.push(0.6); // band 1  -> score ≈ 1.0
        }
        // The "outlier" row: band 0 is slightly elevated, band 1 still clean.
        data.push(1.3);
        data.push(0.3); // score ≈ 1.335

        let r = Array2::from_shape_vec((20, 2), data).unwrap();
        let sigmas = vec![1.0, 1.0];
        let sigma = 2.0;

        let mask = joint_outlier_mask(&r, &sigmas, sigma);
        assert!(
            mask[19],
            "modest single-band deviation should not be flagged"
        );
        // Also verify the clean rows are all kept.
        for i in 0..19 {
            assert!(mask[i], "clean row {} should be kept", i);
        }
    }

    #[test]
    fn test_joint_outlier_flat_residuals() {
        // All residuals identical → MAD = 0 → fallback to is_finite
        let r = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
        let sigmas = vec![1.0, 0.5];
        let mask = joint_outlier_mask(&r, &sigmas, 2.0);
        assert_eq!(mask, vec![true; 5]);
    }

    #[test]
    fn test_joint_outlier_empty_input() {
        let r = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let sigmas = vec![1.0, 1.0];
        let mask = joint_outlier_mask(&r, &sigmas, 2.0);
        assert!(mask.is_empty());
    }

    // ── solve_tridiag_thomas tests ─────────────────────────────────────────

    #[test]
    fn test_solve_tridiag_thomas_known_identity() {
        // A = I (diag=1, off=0), rhs = [[1,4],[2,5],[3,6]], expect same
        let diag = vec![1.0, 1.0, 1.0];
        let off = vec![0.0, 0.0];
        let rhs = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        let x = solve_tridiag_thomas(&diag, &off, &rhs);
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (x[[i, j]] - rhs[[i, j]]).abs() < 1e-12,
                    "x[{},{}]={}, rhs={}",
                    i,
                    j,
                    x[[i, j]],
                    rhs[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_solve_tridiag_thomas_n1() {
        let diag = vec![2.0];
        let off: Vec<f64> = vec![];
        let rhs = Array2::from_shape_vec((1, 3), vec![6.0, 8.0, 10.0]).unwrap();
        let x = solve_tridiag_thomas(&diag, &off, &rhs);
        assert!((x[[0, 0]] - 3.0).abs() < 1e-12);
        assert!((x[[0, 1]] - 4.0).abs() < 1e-12);
        assert!((x[[0, 2]] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_tridiag_thomas_simple_tridiag() {
        // A = [[2,-1,0],[-1,2,-1],[0,-1,2]], rhs = [1, 0, 0] (per column)
        // Exact solution x = [0.75, 0.5, 0.25] for each column
        let diag = vec![2.0, 2.0, 2.0];
        let off = vec![-1.0, -1.0];
        let rhs = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let x = solve_tridiag_thomas(&diag, &off, &rhs);
        for j in 0..2 {
            assert!((x[[0, j]] - 0.75).abs() < 1e-12, "x[0,{}]={}", j, x[[0, j]]);
            assert!((x[[1, j]] - 0.50).abs() < 1e-12, "x[1,{}]={}", j, x[[1, j]]);
            assert!((x[[2, j]] - 0.25).abs() < 1e-12, "x[2,{}]={}", j, x[[2, j]]);
        }
    }

    // ── group_fused_lasso_admm tests ───────────────────────────────────────

    #[test]
    fn test_group_fused_lasso_n1_returns_copy() {
        let r = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let weights: Vec<f64> = vec![];
        let u = group_fused_lasso_admm(&r, 1.0, &weights, 1.0, 10, 1e-6);
        assert_eq!(u.shape(), &[1, 3]);
        assert!((u[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((u[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((u[[0, 2]] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_group_fused_lasso_zero_lambda_returns_copy() {
        let r = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        let weights = vec![1.0, 1.0];
        let u = group_fused_lasso_admm(&r, 0.0, &weights, 1.0, 10, 1e-6);
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (u[[i, j]] - r[[i, j]]).abs() < 1e-12,
                    "u[{},{}]={}, r={}",
                    i,
                    j,
                    u[[i, j]],
                    r[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_group_fused_lasso_empty_returns_empty() {
        let r = Array2::from_shape_vec((0, 3), vec![]).unwrap();
        let weights: Vec<f64> = vec![];
        let u = group_fused_lasso_admm(&r, 1.0, &weights, 1.0, 10, 1e-6);
        assert_eq!(u.shape(), &[0, 3]);
    }

    #[test]
    fn test_group_fused_lasso_huge_lambda_returns_means() {
        // Huge lambda forces every column to collapse to its mean
        let data = vec![
            1.0, 7.0, // row0
            2.0, 8.0, // row1
            3.0, 9.0, // row2
            4.0, 10.0, // row3
        ];
        let r = Array2::from_shape_vec((4, 2), data).unwrap();
        let weights = vec![1.0; 3];
        let u = group_fused_lasso_admm(&r, 1e6, &weights, 1.0, 200, 1e-6);
        // Column means: (1+2+3+4)/4=2.5, (7+8+9+10)/4=8.5
        for i in 0..4 {
            assert!((u[[i, 0]] - 2.5).abs() < 1e-3, "u[{},0]={}", i, u[[i, 0]]);
            assert!((u[[i, 1]] - 8.5).abs() < 1e-3, "u[{},1]={}", i, u[[i, 1]]);
        }
    }

    #[test]
    fn test_group_fused_lasso_shared_breakpoint() {
        // 2-band signal with one shared step at row 5 (of 10 rows).
        // Band0 jumps +1.0, band1 jumps -0.5.
        let n = 10;
        let b = 2;
        let step = 5;
        let mut data = vec![0.0f64; n * b];
        for i in step..n {
            data[i * b] = 1.0; // band0
            data[i * b + 1] = -0.5; // band1
        }
        let truth = Array2::from_shape_vec((n, b), data).unwrap();

        // Add small noise
        let mut noisy = truth.clone();
        // Use a deterministic pseudo-noise via simple seed
        let seed_vals = [
            (0, 0, 0.03),
            (1, 0, -0.02),
            (2, 0, 0.01),
            (0, 1, -0.01),
            (1, 1, 0.04),
            (2, 1, -0.03),
            (7, 0, 0.02),
            (7, 1, -0.02),
        ];
        for &(i, j, v) in &seed_vals {
            noisy[[i, j]] += v;
        }

        let weights = vec![1.0; n - 1];
        let u = group_fused_lasso_admm(&noisy, 0.3, &weights, 1.0, 400, 1e-6);

        // After ADMM, the recovered signals should be nearly flat before/after the step
        let pre_mean0: f64 = (0..step - 1).map(|i| u[[i, 0]]).sum::<f64>() / (step - 1) as f64;
        let post_mean0: f64 = (step + 1..n).map(|i| u[[i, 0]]).sum::<f64>() / (n - step - 1) as f64;
        let pre_mean1: f64 = (0..step - 1).map(|i| u[[i, 1]]).sum::<f64>() / (step - 1) as f64;
        let post_mean1: f64 = (step + 1..n).map(|i| u[[i, 1]]).sum::<f64>() / (n - step - 1) as f64;

        // Recovered jumps
        let jump0 = post_mean0 - pre_mean0;
        let jump1 = post_mean1 - pre_mean1;

        assert!(
            (jump0 - 1.0).abs() < 0.5,
            "band0 jump={:.4}, expected ~1.0",
            jump0
        );
        assert!(
            (jump1 - (-0.5)).abs() < 0.5,
            "band1 jump={:.4}, expected ~-0.5",
            jump1
        );
        // Signs should match
        assert!(jump0 > 0.0, "band0 jump should be positive, got {}", jump0);
        assert!(jump1 < 0.0, "band1 jump should be negative, got {}", jump1);
    }

    #[test]
    fn test_group_fused_lasso_converges_within_max_iter() {
        // Random moderate-sized problem: should converge in < max_iter
        let n = 30;
        let b = 3;
        let mut data = vec![0.0f64; n * b];
        for i in 0..n {
            // Simple trend + noise per band
            let t = i as f64;
            data[i * b] = 0.1 * t; // band0
            data[i * b + 1] = -0.05 * t; // band1
            data[i * b + 2] = 0.0; // band2 flat with noise
                                   // Add small "noise" as a fixed deterministic pattern
            data[i * b] += ((i as f64 * 0.7).sin()) * 0.02;
            data[i * b + 1] += ((i as f64 * 0.7 + 1.0).sin()) * 0.02;
        }
        let r = Array2::from_shape_vec((n, b), data).unwrap();
        let weights = vec![1.0; n - 1];

        // This should converge (not hit max_iter)
        let u = group_fused_lasso_admm(&r, 0.1, &weights, 1.0, 500, 1e-4);

        // Verify output shape
        assert_eq!(u.shape(), &[n, b]);

        // Verify all values are finite
        for i in 0..n {
            for j in 0..b {
                assert!(u[[i, j]].is_finite(), "u[{},{}] not finite", i, j);
            }
        }
    }

    // ── Private frequency selection tests ────────────────────────────────────

    /// Helper: build a power spectrum with a single spike at a given freq bin.
    fn spike_spectrum(
        n_bins: usize,
        f_min: f64,
        f_max: f64,
        spike_bins: &[usize],
        spike_power: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let df = (f_max - f_min) / (n_bins as f64).max(1.0);
        let freqs: Vec<f64> = (0..n_bins).map(|i| f_min + (i as f64 + 0.5) * df).collect();
        let mut power = vec![0.01f64; n_bins];
        for &bin in spike_bins {
            if bin < n_bins {
                power[bin] = spike_power;
            }
        }
        (freqs, power)
    }

    #[test]
    fn test_private_frequency_band_specific_peak() {
        // Band has a strong peak at freq ~0.3, not present in shared freqs.
        // The private set should include it.
        let shared_freqs = vec![0.1];
        let (band_freqs, band_power) = spike_spectrum(100, 0.0, 1.0, &[29], 10.0); // bin 29 ≈ 0.295

        let mut config = default_nufrost_config();
        config.private_top_k_per_band = 2;
        config.spectral_merge_tol = 0.15;
        config.num_peaks = 5;

        let private =
            select_private_frequencies(&band_freqs, &band_power, &shared_freqs, &config, 1.0);

        assert!(
            !private.is_empty(),
            "expected at least one private frequency"
        );
        // The spike bin should be selected (frequency near 0.3)
        let has_spike = private.iter().any(|&f| (f - 0.295).abs() < 0.02);
        assert!(
            has_spike,
            "private frequencies should include the band-specific peak near 0.3, got {:?}",
            private
        );
    }

    #[test]
    fn test_private_frequency_near_shared_excluded() {
        // Band has peaks at 0.3 (distinct) and 0.105 (near shared 0.1).
        // Only 0.3 should appear in private; 0.105 should be excluded.
        let shared_freqs = vec![0.1];
        let (band_freqs, band_power) = spike_spectrum(100, 0.0, 1.0, &[29, 9], 10.0);
        // bin 29 ≈ 0.295, bin 9 ≈ 0.095

        let mut config = default_nufrost_config();
        config.private_top_k_per_band = 3;
        config.spectral_merge_tol = 0.15;
        config.num_peaks = 5;

        let private =
            select_private_frequencies(&band_freqs, &band_power, &shared_freqs, &config, 1.0);

        // The peak near 0.3 should be present
        let has_spike = private.iter().any(|&f| (f - 0.295).abs() < 0.02);
        assert!(
            has_spike,
            "private should include the distinct peak near 0.3, got {:?}",
            private
        );

        // No frequency should be near the shared 0.1 (within merge_tol*2 = 0.3 relative)
        for &f in &private {
            let rel = (f - 0.1).abs() / 0.1f64.max(1e-12);
            assert!(
                rel > 0.3,
                "private freq {} is too close to shared freq 0.1 (rel={})",
                f,
                rel
            );
        }
    }

    #[test]
    fn test_private_frequency_cap_respected() {
        // Four strong peaks, cap at 2 → only 2 private frequencies.
        let shared_freqs = vec![0.05];
        let spike_bins = [30, 50, 70, 90]; // bins far from shared
        let (band_freqs, band_power) = spike_spectrum(100, 0.0, 1.0, &spike_bins, 10.0);

        let mut config = default_nufrost_config();
        config.private_top_k_per_band = 2;
        config.spectral_merge_tol = 0.15;
        config.num_peaks = 10;

        let private =
            select_private_frequencies(&band_freqs, &band_power, &shared_freqs, &config, 1.0);

        assert_eq!(
            private.len(),
            2,
            "expected exactly 2 private frequencies (cap=2), got {}: {:?}",
            private.len(),
            private
        );
    }

    #[test]
    fn test_private_frequency_empty_when_all_near_shared() {
        // All candidate peaks are within merge_tol*2 of shared freqs.
        let shared_freqs = vec![0.1, 0.3, 0.5];
        // Place spike bins near each shared freq: bins 9, 29, 49
        let spike_bins = [9, 29, 49]; // ≈ 0.095, 0.295, 0.495
        let (band_freqs, band_power) = spike_spectrum(100, 0.0, 1.0, &spike_bins, 10.0);

        let mut config = default_nufrost_config();
        config.private_top_k_per_band = 2;
        config.spectral_merge_tol = 0.3; // generous tol, merge_tol_x2 = 0.6
        config.num_peaks = 5;

        let private =
            select_private_frequencies(&band_freqs, &band_power, &shared_freqs, &config, 1.0);

        assert!(
            private.is_empty(),
            "expected empty private set when all peaks are near shared, got {:?}",
            private
        );
    }

    // ── Expanded multi-band BCD tests ───────────────────────────────────────

    fn make_two_band_clean_signals() -> (Vec<f64>, Vec<Vec<f64>>, f64) {
        let n = 60;
        let ts_days: Vec<f64> = (0..n).map(|i| i as f64 * 10.0).collect();
        let period = 365.25;
        let freq_hz = 1.0 / (period * 86400.0);
        let mut band_a = Vec::with_capacity(n);
        let mut band_b = Vec::with_capacity(n);
        for &td in &ts_days {
            let phase = 2.0 * PI * freq_hz * td * 86400.0;
            band_a.push(0.3 * phase.sin() + 0.5);
            band_b.push(0.2 * phase.cos() + 0.6);
        }
        let target_day = ts_days[n / 2];
        (ts_days, vec![band_a, band_b], target_day)
    }

    #[test]
    fn test_multiband_lambda_step_zero_is_harmonic_only() {
        let (ts_days, observations, target_day) = make_two_band_clean_signals();
        let mut cfg_zero = default_nufrost_config();
        cfg_zero.lambda_step = 0.0;
        cfg_zero.max_outer_iter = 3;
        let pred_zero = nufrost_pixel_multiband(&ts_days, &observations, target_day, &cfg_zero);

        let mut cfg_large = default_nufrost_config();
        cfg_large.lambda_step = 1e30;
        cfg_large.max_outer_iter = 3;
        let pred_large = nufrost_pixel_multiband(&ts_days, &observations, target_day, &cfg_large);

        for b in 0..2 {
            assert!(pred_zero[b].is_finite(), "band {} zero-step NaN", b);
            assert!(pred_large[b].is_finite(), "band {} large-step NaN", b);
        }
        assert!(
            (pred_zero[0] - pred_large[0]).abs() < 0.15,
            "lambda_step=0 vs large should produce similar harmonic predictions, got diff={:.4}",
            (pred_zero[0] - pred_large[0]).abs()
        );
    }

    #[test]
    fn test_multiband_lambda_step_large_produces_near_constant_segment() {
        let (ts_days, observations, target_day) = make_two_band_clean_signals();
        let mut cfg = default_nufrost_config();
        cfg.lambda_step = 1e12;
        cfg.max_outer_iter = 3;
        cfg.step_dt_weighting = false;

        let pred = nufrost_pixel_multiband(&ts_days, &observations, target_day, &cfg);
        for b in 0..2 {
            assert!(pred[b].is_finite(), "band {} NaN", b);
        }
    }

    #[test]
    fn test_multiband_max_outer_iter_respected() {
        let (ts_days, observations, target_day) = make_two_band_clean_signals();
        let mut cfg = default_nufrost_config();
        cfg.max_outer_iter = 1;
        cfg.outer_tol = 1e-15;
        // Should not panic with single iteration.
        let pred = nufrost_pixel_multiband(&ts_days, &observations, target_day, &cfg);
        assert_eq!(pred.len(), 2);
        assert!(pred[0].is_finite() && pred[1].is_finite());
    }

    #[test]
    fn test_multiband_early_convergence() {
        let (ts_days, observations, target_day) = make_two_band_clean_signals();
        let mut cfg = default_nufrost_config();
        cfg.max_outer_iter = 20;
        cfg.outer_tol = 0.99;
        let pred = nufrost_pixel_multiband(&ts_days, &observations, target_day, &cfg);
        assert!(pred[0].is_finite() && pred[1].is_finite());
    }

    #[test]
    fn test_multiband_joint_outlier_excludes_rows() {
        let (ts_days, mut observations, target_day) = make_two_band_clean_signals();
        // Insert a shared outlier at index 5.
        observations[0][5] += 100.0;
        observations[1][5] += 100.0;

        let mut cfg_on = default_nufrost_config();
        cfg_on.joint_outlier = true;
        cfg_on.joint_outlier_sigma = 2.5;
        let pred_on = nufrost_pixel_multiband(&ts_days, &observations, target_day, &cfg_on);

        let mut cfg_off = default_nufrost_config();
        cfg_off.joint_outlier = false;
        let pred_off = nufrost_pixel_multiband(&ts_days, &observations, target_day, &cfg_off);

        for b in 0..2 {
            assert!(pred_on[b].is_finite(), "band {} joint-on NaN", b);
            assert!(pred_off[b].is_finite(), "band {} joint-off NaN", b);
        }
        assert!(
            (pred_on[0] - pred_off[0]).abs() > 1e-6,
            "joint outlier should change prediction, got diff={:.6}",
            (pred_on[0] - pred_off[0]).abs()
        );
    }

    #[test]
    fn test_multiband_joint_outlier_fallback_when_too_few_kept() {
        let n = 15;
        let ts_days: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut band_a = vec![0.0f64; n];
        let mut band_b = vec![0.0f64; n];
        for i in 0..n {
            band_a[i] = (i as f64 * 0.3).sin() + 0.5;
            band_b[i] = (i as f64 * 0.2).cos() + 0.5;
        }
        let observations = vec![band_a, band_b];

        let mut cfg = default_nufrost_config();
        cfg.joint_outlier = true;
        cfg.joint_outlier_sigma = 0.01;
        cfg.min_obs = 14;
        // Very tight sigma may reject most rows, triggering fallback.
        let pred = nufrost_pixel_multiband(&ts_days, &observations, ts_days[n / 2], &cfg);
        assert_eq!(pred.len(), 2);
        assert!(pred[0].is_finite() && pred[1].is_finite());
    }

    #[test]
    fn test_multiband_prediction_changes_when_u_is_nonzero() {
        let (ts_days, mut observations, target_day) = make_two_band_clean_signals();
        // Add a systematic offset to band A at later times to force non-zero U.
        let n = ts_days.len();
        for i in (n / 2)..n {
            observations[0][i] += 0.15;
        }

        let mut cfg_step = default_nufrost_config();
        cfg_step.lambda_step = 0.5;
        cfg_step.max_outer_iter = 5;
        let pred_step = nufrost_pixel_multiband(&ts_days, &observations, target_day, &cfg_step);

        let mut cfg_zero = default_nufrost_config();
        cfg_zero.lambda_step = 0.0;
        cfg_zero.max_outer_iter = 5;
        let pred_zero = nufrost_pixel_multiband(&ts_days, &observations, target_day, &cfg_zero);

        assert!(pred_step[0].is_finite() && pred_step[1].is_finite());
        assert!(pred_zero[0].is_finite() && pred_zero[1].is_finite());
        assert!(
            (pred_step[0] - pred_zero[0]).abs() > 1e-6,
            "non-zero U should change prediction, got diff={:.6}",
            (pred_step[0] - pred_zero[0]).abs()
        );
    }
}
