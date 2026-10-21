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
use crate::config::{FrequencySelection, NufrostConfig};

mod vector_fit;

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
    let min_obs = config.robust.min_obs as usize;

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
            include_trend: config.ridge.include_trend,
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
            include_trend: config.ridge.include_trend,
            ..Default::default()
        };
    }

    // ── Compute spectrum ──────────────────────────────────────────────────
    let y_scale = 10000.0;
    let (f_pos, p_pos) =
        compute_spectrum_nufft(&t_rel, &yy, config.spectral.modes as usize, y_scale);

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
            .filter(|&f| f.is_finite() && f > config.spectral.ignore_dc_hz && f <= fmax)
            .collect()
    } else {
        let mode = match config.spectral.selection {
            FrequencySelection::Preferred => "preferred",
            FrequencySelection::Hybrid => "hybrid",
            FrequencySelection::Spectral | FrequencySelection::All => "spectral",
        };
        let pref_freqs = parse_preferred_frequencies(&config.spectral.preferred_periods_days);
        select_frequencies(
            &f_pos,
            &p_pos,
            fmax,
            mode,
            &pref_freqs,
            config.spectral.preferred_top_k as usize,
            config.spectral.num_peaks as usize,
            config.spectral.spectral_top_k as usize,
            config.spectral.merge_tolerance,
            config.spectral.power_cum,
            config.spectral.ignore_dc_hz,
            config.spectral.refine_peaks,
        )
    };

    let n_freqs_used = freqs_sel.len();

    // ── Scale observations ─────────────────────────────────────────────────
    let yy_scaled: Vec<f64> = yy.iter().map(|&v| v / y_scale).collect();

    // ── Build design matrix ────────────────────────────────────────────────
    let x = make_design_matrix(
        &t_rel,
        &freqs_sel,
        config.ridge.include_trend,
        true, // include_dc always true
    );

    // ── Huber-Ridge fit with iterative outlier rejection ────────────────────
    let n_min_needed = (1 + config.ridge.include_trend as usize + 2 * n_freqs_used).max(min_obs);

    let beta = if config.robust.outlier_sigma > 0.0 {
        // MAD-based iterative outlier rejection
        let mut t_curr = t_rel.clone();
        let mut y_curr = yy_scaled.clone();
        let mut beta_final: Option<Vec<f64>> = None;

        for _ in 0..(t_curr.len().min(50)) {
            if y_curr.len() < n_min_needed {
                break;
            }
            let x_curr = make_design_matrix(&t_curr, &freqs_sel, config.ridge.include_trend, true);
            let y_arr = Array1::from_vec(y_curr.clone());
            let (beta_curr, y_pred) = robust_fit_freq_ridge(
                &x_curr,
                &y_arr,
                &freqs_sel,
                config.ridge.lambda,
                config.robust.huber_iters as usize,
                config.robust.huber_delta,
                true,
                config.ridge.include_trend,
                config.ridge.freq_weight,
            );

            beta_final = Some(beta_curr.clone());

            let residuals: Vec<f64> = y_curr
                .iter()
                .zip(y_pred.iter())
                .map(|(&a, &b)| a - b)
                .collect();
            let mad = mad_std(&residuals);
            let threshold = config.robust.outlier_sigma * mad.max(1e-12);

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
                    config.ridge.lambda,
                    config.robust.huber_iters as usize,
                    config.robust.huber_delta,
                    true,
                    config.ridge.include_trend,
                    config.ridge.freq_weight,
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
            config.ridge.lambda,
            config.robust.huber_iters as usize,
            config.robust.huber_delta,
            true,
            config.ridge.include_trend,
            config.ridge.freq_weight,
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
        include_trend: config.ridge.include_trend,
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
    let min_obs = config.robust.min_obs as usize;
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

    let (f_pos, p_pos) =
        compute_spectrum_nufft(&t_rel, &yy_raw, config.spectral.modes as usize, y_scale);

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

    let selection_mode = config.spectral.selection.select_mode();
    let preferred_freqs = parse_preferred_frequencies(&config.spectral.preferred_periods_days);
    let freqs_sel = select_frequencies(
        &f_pos,
        &p_pos,
        fmax,
        selection_mode,
        &preferred_freqs,
        config.spectral.preferred_top_k as usize,
        config.spectral.num_peaks as usize,
        config.spectral.spectral_top_k as usize,
        config.spectral.merge_tolerance,
        config.spectral.power_cum,
        config.spectral.ignore_dc_hz,
        config.spectral.refine_peaks,
    );
    let n_freqs = freqs_sel.len();

    let x = make_design_matrix(&t_rel, &freqs_sel, config.ridge.include_trend, true);
    if x.ncols() == 0 {
        return (fill_value, 0);
    }

    let weights = difference_weights(&t, config.step.weight_by_gap);
    let mut u = vec![0.0f64; yy.len()];
    let y_arr = Array1::from_vec(yy.clone());
    let mut beta = tiered_ridge_solve(
        &x,
        &y_arr,
        &freqs_sel,
        config.ridge.lambda,
        config.ridge.lambda_high,
        config.ridge.low_freq_period_days,
        config.ridge.freq_weight,
        true,
        config.ridge.include_trend,
        None,
        None,
    )
    .unwrap_or_else(|| Array1::zeros(x.ncols()));

    for _ in 0..(config.step.max_outer_iter as usize) {
        let beta_old = beta.clone();
        let u_old = u.clone();

        let y_hat = x.dot(&beta);
        let residual: Vec<f64> = yy
            .iter()
            .zip(y_hat.iter())
            .map(|(&yi, &yh)| yi - yh)
            .collect();
        u = fused_lasso_1d(&residual, config.step.lambda, &weights);

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
            config.ridge.lambda,
            config.ridge.lambda_high,
            config.ridge.low_freq_period_days,
            config.ridge.freq_weight,
            true,
            config.ridge.include_trend,
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
        if (delta_beta + delta_u).sqrt() / denom < config.step.outer_tol {
            break;
        }
    }

    let t_star_rel = target_t - t_min;
    let mut pred = beta.get(0).copied().unwrap_or(0.0);
    let mut idx = 1usize;
    if config.ridge.include_trend {
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
    vector_fit::fit_pixel(ts_days, observations, config)
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
#[cfg(feature = "raster")]
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

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FrequencySelection, NufrostConfig};

    // ── Default test config ────────────────────────────────────────────────

    fn default_nufrost_config() -> NufrostConfig {
        serde_json::from_str(
            r#"{
                "modes": 4096,
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
                "joint_outlier": false
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
        config.spectral.selection = FrequencySelection::Hybrid;
        config.step.lambda = 1e30;
        config.robust.min_obs = 8;

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
    fn test_nufrost_pixel_vector_rejects_uneven_bands() {
        let config = default_nufrost_config();
        let fit = nufrost_fit_pixel_vector(&[0.0, 8.0], &[vec![1.0, 2.0], vec![1.0]], &config);
        assert!(!fit.valid);
        assert!(fit.freqs.is_empty());
    }

    #[test]
    fn test_nufrost_pixel_vector_insufficient_dates_is_invalid() {
        let config = default_nufrost_config();
        let times = vec![0.0, 8.0];
        let observations = vec![vec![1000.0, f64::NAN], vec![1400.0, 1500.0]];
        let fit = nufrost_fit_pixel_vector(&times, &observations, &config);
        assert!(!fit.valid);
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
}
