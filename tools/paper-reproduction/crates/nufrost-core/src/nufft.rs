use std::f64::consts::PI;

/// Minimal complex number type for the in-crate NUFFT implementation.
#[derive(Debug, Clone, Copy, Default)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    #[inline]
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn scale(self, a: f64) -> Self {
        Self::new(self.re * a, self.im * a)
    }

    #[inline]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.abs2();
        if denom <= 0.0 {
            return Self::default();
        }
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

impl std::ops::Add for C64 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for C64 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for C64 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl std::ops::AddAssign for C64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

/// Options for the paper-style gridding NUFFT.
#[derive(Debug, Clone, Copy)]
pub struct NufftOptions {
    /// Oversampling factor for the uniform FFT grid.
    pub oversamp: f64,
    /// Compact kernel width in grid cells.
    pub kernel_width: usize,
    /// Kaiser-Bessel beta parameter.
    pub beta: f64,
}

impl Default for NufftOptions {
    fn default() -> Self {
        Self {
            oversamp: 2.0,
            kernel_width: 8,
            beta: 10.0,
        }
    }
}

/// Round up to the next power of two.
pub fn next_power_of_two_at_least(n: usize) -> usize {
    n.max(2).next_power_of_two()
}

/// Paper-style gridded type-1 NUFFT spectrum for real-valued observations.
///
/// This follows the three-stage approximation described in the manuscript:
///
/// 1. spread non-uniform impulses to an oversampled uniform grid using a
///    compact Kaiser-Bessel kernel;
/// 2. apply a radix-2 FFT to the uniform grid;
/// 3. deconvolve by the FFT of the same periodized kernel.
///
/// The returned frequency grid uses the same positive-mode convention as the
/// Python implementation after `finufft.nufft1d1(..., isign=-1)`:
/// `k = 0..M/2-1`, `freqs = k / Tspan`.
pub fn type1_spectrum_kb(
    t_rel: &[f64],
    y: &[f64],
    modes: usize,
    y_scale: f64,
    opts: NufftOptions,
) -> (Vec<f64>, Vec<f64>) {
    let n = t_rel.len();
    if n == 0 || y.len() != n {
        return (Vec::new(), Vec::new());
    }

    let t_min = t_rel.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = t_rel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let tspan = t_max - t_min;
    if tspan <= 0.0 || !tspan.is_finite() {
        return (vec![0.0], vec![0.0]);
    }

    let m = next_even(modes);
    let nf = next_power_of_two_at_least(((m as f64) * opts.oversamp.max(1.25)).ceil() as usize);
    let width = opts.kernel_width.max(2);
    let half_width = 0.5 * width as f64;

    let mut grid = vec![C64::default(); nf];
    for (&ti, &yi) in t_rel.iter().zip(y.iter()) {
        if !ti.is_finite() || !yi.is_finite() {
            continue;
        }
        let x = 2.0 * PI * (ti - t_min) / tspan - PI;
        let u = x * (nf as f64) / (2.0 * PI);
        let center = u.round() as isize;
        let c = C64::new(yi / y_scale, 0.0);

        for offset in -(width as isize)..=(width as isize) {
            let idx_i = center + offset;
            let dist = u - idx_i as f64;
            if dist.abs() > half_width {
                continue;
            }
            let idx = idx_i.rem_euclid(nf as isize) as usize;
            let w = kaiser_bessel(dist / half_width, opts.beta);
            grid[idx] += c.scale(w);
        }
    }

    let kernel_hat = kernel_fft(nf, width, opts.beta);
    fft_forward(&mut grid);

    let n_pos = m / 2;
    let mut freqs = Vec::with_capacity(n_pos);
    let mut power = Vec::with_capacity(n_pos);

    for k in 0..n_pos {
        let kh = kernel_hat[k];
        let fk = if kh.abs2() > 1e-24 {
            grid[k].div(kh)
        } else {
            C64::default()
        };
        freqs.push(k as f64 / tspan);
        power.push(fk.abs2());
    }

    (freqs, power)
}

/// Paper-style gridded type-1 NUFFT power for vector-valued observations.
///
/// `y_by_dim[d][j]` is dimension/band `d` at non-uniform time `t_rel[j]`.
/// All dimensions share the same spread indices, Kaiser-Bessel weights, FFT
/// grid size, and kernel deconvolution. The returned power is the squared
/// vector spectrum norm:
///
/// ```text
/// P_k = Σ_d |F_{k,d}|²
/// ```
pub fn type1_vector_power_kb(
    t_rel: &[f64],
    y_by_dim: &[Vec<f64>],
    modes: usize,
    opts: NufftOptions,
) -> (Vec<f64>, Vec<f64>) {
    let n = t_rel.len();
    let dims = y_by_dim.len();
    if n == 0 || dims == 0 || y_by_dim.iter().any(|y| y.len() != n) {
        return (Vec::new(), Vec::new());
    }

    let t_min = t_rel.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = t_rel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let tspan = t_max - t_min;
    if tspan <= 0.0 || !tspan.is_finite() {
        return (vec![0.0], vec![0.0]);
    }

    let m = next_even(modes);
    let nf = next_power_of_two_at_least(((m as f64) * opts.oversamp.max(1.25)).ceil() as usize);
    let width = opts.kernel_width.max(2);
    let half_width = 0.5 * width as f64;

    let mut grids = vec![vec![C64::default(); nf]; dims];
    for j in 0..n {
        let ti = t_rel[j];
        if !ti.is_finite() {
            continue;
        }
        let x = 2.0 * PI * (ti - t_min) / tspan - PI;
        let u = x * (nf as f64) / (2.0 * PI);
        let center = u.round() as isize;

        for offset in -(width as isize)..=(width as isize) {
            let idx_i = center + offset;
            let dist = u - idx_i as f64;
            if dist.abs() > half_width {
                continue;
            }
            let idx = idx_i.rem_euclid(nf as isize) as usize;
            let w = kaiser_bessel(dist / half_width, opts.beta);
            for d in 0..dims {
                let y = y_by_dim[d][j];
                if y.is_finite() {
                    grids[d][idx] += C64::new(y, 0.0).scale(w);
                }
            }
        }
    }

    let kernel_hat = kernel_fft(nf, width, opts.beta);
    for grid in &mut grids {
        fft_forward(grid);
    }

    let n_pos = m / 2;
    let mut freqs = Vec::with_capacity(n_pos);
    let mut power = vec![0.0; n_pos];

    for k in 0..n_pos {
        let kh = kernel_hat[k];
        freqs.push(k as f64 / tspan);
        if kh.abs2() <= 1e-24 {
            continue;
        }
        for grid in &grids {
            let fk = grid[k].div(kh);
            power[k] += fk.abs2();
        }
    }

    (freqs, power)
}

/// Direct type-1 non-uniform Fourier spectrum using Python/FINUFFT mode order.
///
/// Python computes:
///
/// ```text
/// x_j = 2π (t_j - min(t)) / Tspan - π
/// F_k = Σ_j c_j exp(-i k x_j),  k = -M/2 ... M/2-1
/// freqs = k / Tspan
/// keep freqs >= 0
/// ```
///
/// This function returns the same kept positive modes, but evaluates the sum
/// directly. It is intentionally kept as the correctness oracle for the
/// gridded NUFFT path and for Python parity tests.
pub fn type1_spectrum_direct(
    t_rel: &[f64],
    y: &[f64],
    modes: usize,
    y_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = t_rel.len();
    let t_min = t_rel.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = t_rel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let tspan = t_max - t_min;
    if n == 0 || tspan <= 0.0 || !tspan.is_finite() {
        return (vec![0.0], vec![0.0]);
    }

    let m = next_even(modes);
    let n_pos = m / 2;
    let x: Vec<f64> = t_rel
        .iter()
        .map(|&ti| 2.0 * PI * (ti - t_min) / tspan - PI)
        .collect();
    let c: Vec<f64> = y.iter().map(|&yi| yi / y_scale).collect();

    let mut freqs = Vec::with_capacity(n_pos);
    let mut power = Vec::with_capacity(n_pos);
    for k in 0..n_pos {
        let kf = k as f64;
        let mut re = 0.0;
        let mut im = 0.0;
        for j in 0..n {
            let phase = -kf * x[j];
            re += c[j] * phase.cos();
            im += c[j] * phase.sin();
        }
        freqs.push(kf / tspan);
        power.push(re * re + im * im);
    }
    (freqs, power)
}

#[inline]
fn next_even(n: usize) -> usize {
    ((n + 1) / 2) * 2
}

fn kernel_fft(nf: usize, width: usize, beta: f64) -> Vec<C64> {
    let half_width = 0.5 * width.max(2) as f64;
    let mut kernel = vec![C64::default(); nf];
    for offset in -(width as isize)..=(width as isize) {
        let dist = offset as f64;
        if dist.abs() > half_width {
            continue;
        }
        let idx = offset.rem_euclid(nf as isize) as usize;
        kernel[idx] = C64::new(kaiser_bessel(dist / half_width, beta), 0.0);
    }
    fft_forward(&mut kernel);
    kernel
}

fn kaiser_bessel(x: f64, beta: f64) -> f64 {
    let ax = x.abs();
    if ax > 1.0 {
        return 0.0;
    }
    let arg = beta * (1.0 - ax * ax).sqrt();
    bessel_i0(arg) / bessel_i0(beta)
}

/// Approximation to the modified Bessel function I0.
///
/// Numerical Recipes / Cephes-style polynomial approximation.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

fn fft_forward(data: &mut [C64]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    let mut len = 2usize;
    while len <= n {
        let angle = -2.0 * PI / len as f64;
        let wlen = C64::new(angle.cos(), angle.sin());
        let half = len / 2;
        for i in (0..n).step_by(len) {
            let mut w = C64::new(1.0, 0.0);
            for j in 0..half {
                let u = data[i + j];
                let v = data[i + j + half] * w;
                data[i + j] = u + v;
                data[i + j + half] = u - v;
                w = w * wlen;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_delta_is_constant() {
        let mut data = vec![C64::default(); 8];
        data[0] = C64::new(1.0, 0.0);
        fft_forward(&mut data);
        for z in data {
            assert!((z.re - 1.0).abs() < 1e-12);
            assert!(z.im.abs() < 1e-12);
        }
    }

    #[test]
    fn direct_spectrum_detects_periodic_signal() {
        let t: Vec<f64> = (0..40).map(|i| i as f64 * 86400.0 * 5.0).collect();
        let f = 5.0 / (t[t.len() - 1] - t[0]);
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| 100.0 * (2.0 * PI * f * ti).cos())
            .collect();
        let (freqs, power) = type1_spectrum_direct(&t, &y, 128, 10000.0);
        let peak_power = power
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(idx, _)| (freqs[*idx] - f).abs() <= 2.0 / (t[t.len() - 1] - t[0]))
            .map(|(_, &p)| p)
            .fold(0.0, f64::max);
        let median_power = {
            let mut p = power[1..].to_vec();
            p.sort_by(|a, b| a.partial_cmp(b).unwrap());
            p[p.len() / 2]
        };
        assert!(peak_power > 5.0 * median_power);
    }

    #[test]
    fn gridded_nufft_returns_finite_power() {
        let t: Vec<f64> = (0..32).map(|i| i as f64 * 86400.0 * 7.0).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| 2000.0 + 300.0 * (2.0 * PI * ti / (90.0 * 86400.0)).sin())
            .collect();
        let opts = NufftOptions {
            oversamp: 4.0,
            kernel_width: 16,
            beta: 18.0,
        };
        let (freqs, power) = type1_spectrum_kb(&t, &y, 128, 10000.0, opts);
        assert_eq!(freqs.len(), 64);
        assert_eq!(power.len(), 64);
        assert!(power.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gridded_deconvolved_spectrum_has_non_dc_energy() {
        let t: Vec<f64> = (0..40).map(|i| i as f64 * 86400.0 * 5.0).collect();
        let f = 5.0 / (t[t.len() - 1] - t[0]);
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| 100.0 * (2.0 * PI * f * ti).cos())
            .collect();
        let (_, power) = type1_spectrum_kb(&t, &y, 128, 10000.0, NufftOptions::default());
        let non_dc_peak = power.iter().skip(1).copied().fold(0.0, f64::max);
        assert!(non_dc_peak.is_finite());
        assert!(non_dc_peak > 0.0);
    }

    #[test]
    fn vector_power_matches_sum_of_scalar_powers() {
        let t: Vec<f64> = (0..32).map(|i| i as f64 * 86400.0 * 6.0).collect();
        let y1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * ti / (90.0 * 86400.0)).sin())
            .collect();
        let y2: Vec<f64> = t
            .iter()
            .map(|&ti| 0.5 * (2.0 * PI * ti / (120.0 * 86400.0)).cos())
            .collect();
        let opts = NufftOptions::default();
        let (fv, pv) = type1_vector_power_kb(&t, &[y1.clone(), y2.clone()], 128, opts);
        let (f1, p1) = type1_spectrum_kb(&t, &y1, 128, 1.0, opts);
        let (f2, p2) = type1_spectrum_kb(&t, &y2, 128, 1.0, opts);

        assert_eq!(fv, f1);
        assert_eq!(fv, f2);
        for i in 0..pv.len() {
            assert!((pv[i] - (p1[i] + p2[i])).abs() < 1e-8);
        }
    }
}
