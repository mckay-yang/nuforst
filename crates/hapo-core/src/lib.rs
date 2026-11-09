//! HAPO (Zhou et al., ISPRS JPRS 2022), per-band annual harmonic regression.
//! Objective and preprocessing follow the USGS HAPO.py, main branch.
//! The seven-dimensional BFGS solve uses an analytic gradient and deterministic
//! initial coefficients; these solver choices are exposed for reference checks.
use anyhow::{ensure, Result};
const D: usize = 7;
type V = [f64; D];
type Matrix = [[f64; D]; D];

#[derive(Clone, Debug)]
pub struct Fit {
    pub coefficients: V,
    pub slope: f64,
    pub center: f64,
    pub lower: f64,
    pub range: f64,
    pub objective: f64,
    pub iterations: usize,
    pub converged: bool,
    pub gradient_inf: f64,
    /// Minimum-norm subgradient residual, accounting for zero harmonic groups.
    pub stationarity_inf: f64,
}
pub fn basis(t: f64) -> V {
    let w = std::f64::consts::TAU * t / 365.25;
    [
        1.,
        w.cos(),
        w.sin(),
        (2. * w).cos(),
        (2. * w).sin(),
        (3. * w).cos(),
        (3. * w).sin(),
    ]
}
fn dot(a: &V, b: &V) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let k = q * (sorted.len() - 1) as f64;
    let i = k.floor() as usize;
    sorted[i] + (sorted[k.ceil() as usize] - sorted[i]) * k.fract()
}

/// Squared residuals are reduced to a seven-by-seven Gram matrix once per band.
/// The returned gradient uses zero subgradients at the amplitude/absolute kinks.
fn objective(c: &V, gram: &Matrix, xy: &V, yy: f64) -> (f64, V) {
    let gc: V = std::array::from_fn(|i| dot(&gram[i], c));
    let mse = (dot(c, &gc) - 2. * dot(c, xy) + yy).max(0.);
    let r: [f64; 3] = std::array::from_fn(|k| c[1 + 2 * k].hypot(c[2 + 2 * k]));
    let sum: f64 = r.iter().sum();
    let sign = if sum > 1. {
        1.
    } else if sum < 1. {
        -1.
    } else {
        0.
    };
    let penalty = r[0] + 2. * r[1] + 3. * r[2] + (1. - sum).abs();
    let grad = std::array::from_fn(|j| {
        let dp = if j == 0 {
            0.
        } else {
            let k = (j - 1) / 2;
            if r[k] > 0. {
                ((k + 1) as f64 + sign) * c[j] / r[k]
            } else {
                0.
            }
        };
        (gc[j] - xy[j]) * penalty + 0.5 * mse * dp
    });
    (0.5 * mse * penalty, grad)
}

fn stationarity(c: &V, gram: &Matrix, xy: &V, yy: f64) -> f64 {
    let gc: V = std::array::from_fn(|i| dot(&gram[i], c));
    let mse = (dot(c, &gc) - 2. * dot(c, xy) + yy).max(0.);
    let r: [f64; 3] = std::array::from_fn(|k| c[1 + 2 * k].hypot(c[2 + 2 * k]));
    let sum: f64 = r.iter().sum();
    let p = r[0] + 2. * r[1] + 3. * r[2] + (1. - sum).abs();
    let residual = |z: f64| {
        let mut v = ((gc[0] - xy[0]) * p).abs();
        for (k, &amp) in r.iter().enumerate() {
            let a = (gc[1 + 2 * k] - xy[1 + 2 * k]) * p;
            let b = (gc[2 + 2 * k] - xy[2 + 2 * k]) * p;
            let radius = 0.5 * mse * ((k + 1) as f64 + z);
            let group = if amp <= 1e-8 {
                (a.hypot(b) - radius).max(0.)
            } else {
                (a + radius * c[1 + 2 * k] / amp).hypot(b + radius * c[2 + 2 * k] / amp)
            };
            v = v.max(group);
        }
        v
    };
    if (sum - 1.).abs() > 1e-8 {
        return residual(if sum > 1. { 1. } else { -1. });
    }
    // At the absolute-value kink one shared subgradient z lies in [-1,1].
    let (mut a, mut b) = (-1., 1.);
    for _ in 0..70 {
        let u = (2. * a + b) / 3.;
        let v = (a + 2. * b) / 3.;
        if residual(u) < residual(v) {
            b = v
        } else {
            a = u
        }
    }
    residual((a + b) / 2.)
}

pub fn fit(days: &[f64], values: &[f64], initial: Option<V>) -> Result<Fit> {
    ensure!(days.len() == values.len(), "date/value length mismatch");
    let rows: Vec<_> = days
        .iter()
        .zip(values)
        .filter(|(t, y)| t.is_finite() && y.is_finite())
        .map(|(&t, &y)| (t, y))
        .collect();
    ensure!(
        rows.len() >= 8,
        "HAPO requires at least eight retained observations"
    );
    let n = rows.len() as f64;
    let center = rows.iter().map(|r| r.0).sum::<f64>() / n;
    let mean = rows.iter().map(|r| r.1).sum::<f64>() / n;
    let denominator = rows.iter().map(|r| (r.0 - center).powi(2)).sum::<f64>();
    ensure!(denominator > 0., "degenerate dates");
    let slope = rows
        .iter()
        .map(|r| (r.0 - center) * (r.1 - mean))
        .sum::<f64>()
        / denominator;
    let detrended: Vec<_> = rows.iter().map(|r| r.1 - slope * (r.0 - center)).collect();
    let mut sorted = detrended.clone();
    sorted.sort_by(f64::total_cmp);
    let lower = percentile(&sorted, 0.01);
    let range = percentile(&sorted, 0.99) - lower;
    // Explicit constant-residual limit; avoids division by zero in the reference.
    if range <= 1e-12 * mean.abs().max(1.) {
        return Ok(Fit {
            coefficients: [0.; D],
            slope,
            center,
            lower: mean,
            range: 0.,
            objective: 0.,
            iterations: 0,
            converged: true,
            gradient_inf: 0.,
            stationarity_inf: 0.,
        });
    }
    let mut gram = [[0.; D]; D];
    let mut xy = [0.; D];
    let mut yy = 0.;
    let mut ym = 0.;
    for ((t, _), y) in rows.iter().zip(&detrended) {
        let x = basis(*t);
        let y = (y - lower) / range;
        ym += y / n;
        yy += y * y / n;
        for i in 0..D {
            xy[i] += x[i] * y / n;
            for j in 0..D {
                gram[i][j] += x[i] * x[j] / n;
            }
        }
    }
    let mut c = initial.unwrap_or([ym / 2., 0.17, 0.31, 0.53, 0.71, 0.23, 0.43]);
    ensure!(
        c.iter().all(|x| x.is_finite()),
        "nonfinite initial coefficients"
    );
    let mut h: Matrix =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    let (mut f, mut g) = objective(&c, &gram, &xy, yy);
    let mut iterations = 0;
    let mut converged = false;
    for it in 0..1000 {
        if g.iter().map(|v| v.abs()).fold(0., f64::max) < 1e-5 {
            converged = true;
            break;
        }
        let mut dir: V = std::array::from_fn(|i| -dot(&h[i], &g));
        if dot(&dir, &g) >= -1e-14 * dot(&g, &g) {
            dir = g.map(|v| -v);
            h = std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
        }
        let dg = dot(&dir, &g);
        let mut step = 1.;
        let mut accepted = None;
        for _ in 0..50 {
            let next = std::array::from_fn(|j| c[j] + step * dir[j]);
            let (nf, ng) = objective(&next, &gram, &xy, yy);
            if nf.is_finite() && nf <= f + 1e-4 * step * dg {
                accepted = Some((next, nf, ng));
                break;
            }
            step *= 0.5;
        }
        let Some((next, nf, ng)) = accepted else {
            break;
        };
        let s: V = std::array::from_fn(|j| next[j] - c[j]);
        let y: V = std::array::from_fn(|j| ng[j] - g[j]);
        let sy = dot(&s, &y);
        if sy > 1e-12 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
            let hy: V = std::array::from_fn(|i| dot(&h[i], &y));
            let yhy = dot(&y, &hy);
            for i in 0..D {
                for j in 0..D {
                    h[i][j] +=
                        (1. + yhy / sy) * s[i] * s[j] / sy - (s[i] * hy[j] + hy[i] * s[j]) / sy;
                }
            }
        }
        c = next;
        f = nf;
        g = ng;
        iterations = it + 1;
    }
    let gradient_inf = g.iter().map(|v| v.abs()).fold(0., f64::max);
    converged |= gradient_inf < 1e-5;
    ensure!(
        f.is_finite() && c.iter().all(|x| x.is_finite()),
        "nonfinite fit"
    );
    let stationarity_inf = stationarity(&c, &gram, &xy, yy);
    Ok(Fit {
        coefficients: c,
        slope,
        center,
        lower,
        range,
        objective: f,
        iterations,
        converged,
        gradient_inf,
        stationarity_inf,
    })
}
impl Fit {
    pub fn predict(&self, t: f64) -> f64 {
        dot(&basis(t), &self.coefficients) * self.range
            + self.lower
            + self.slope * (t - self.center)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_usgs_example_matches_reference() {
        let v: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/hapo/official-example.json"
        ))
        .unwrap();
        let numbers = |key: &str| {
            v[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let f = fit(
            &numbers("days"),
            &numbers("values"),
            Some(numbers("initial").try_into().unwrap()),
        )
        .unwrap();
        assert!((f.objective - v["official_objective"].as_f64().unwrap()).abs() < 1e-8);
        assert!(
            (f.predict(v["query"].as_f64().unwrap()) - v["official_prediction"].as_f64().unwrap())
                .abs()
                < 0.05
        );
    }
    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let gram = std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
        for c in [
            [0.2, 0.1, 0.2, 0.04, 0.06, 0.03, -0.05],
            [0.5, 0.8, 0.7, 0.3, 0.2, -0.1, 0.2],
        ] {
            let (_, g) = objective(&c, &gram, &[0.1; D], 2.);
            for j in 0..D {
                let mut a = c;
                let mut b = c;
                a[j] += 1e-6;
                b[j] -= 1e-6;
                let fd = (objective(&a, &gram, &[0.1; D], 2.).0
                    - objective(&b, &gram, &[0.1; D], 2.).0)
                    / 2e-6;
                assert!((fd - g[j]).abs() < 1e-7);
            }
        }
    }
    #[test]
    fn linear_limit_and_missing_values() {
        let days: Vec<_> = (0..40).map(|i| i as f64 * 11.).collect();
        let mut y: Vec<_> = days.iter().map(|t| 123. + t * 0.3).collect();
        y[9] = f64::NAN;
        let f = fit(&days, &y, None).unwrap();
        assert!((f.predict(222.) - 189.6).abs() < 1e-9);
        assert!(f.converged);
    }
    #[test]
    fn zero_groups_admit_a_nonzero_penalty_subgradient() {
        let gram = std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
        let c = [0.; D];
        let mut xy = [0.; D];
        xy[3] = 0.2;
        assert!(stationarity(&c, &gram, &xy, 1.) < 1e-12);
        xy[3] = 0.7;
        assert!((stationarity(&c, &gram, &xy, 1.) - 0.2).abs() < 1e-12);
    }
}
