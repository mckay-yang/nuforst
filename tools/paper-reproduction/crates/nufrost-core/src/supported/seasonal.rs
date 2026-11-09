use crate::nufrost as diag;

pub struct SeasonalSupport<'a> {
    pub times: &'a [f64],
    pub values: &'a [Vec<f64>],
    pub train: &'a [usize],
    pub seasonal: &'a diag::NufrostVectorResult,
    pub phase_transfer: bool,
}
impl SeasonalSupport<'_> {
    fn template(&self, query: f64) -> Vec<f64> {
        let mut value = diag::nufrost_predict_vector(self.seasonal, query);
        if !self.phase_transfer {
            return value;
        }
        let mut mass = 0.;
        let mut residual = vec![0.; value.len()];
        for &i in self.train {
            let delta = self.times[i] - query;
            if delta.abs() < 182.625 {
                continue;
            }
            let phase = (delta + 182.625).rem_euclid(365.25) - 182.625;
            let w = (-0.5 * (phase / 20.).powi(2) - delta.abs() / (3. * 365.25)).exp();
            if w < 1e-8 {
                continue;
            }
            let fitted = diag::nufrost_predict_vector(self.seasonal, self.times[i]);
            mass += w;
            for b in 0..value.len() {
                residual[b] += w * (self.values[b][i] - fitted[b]);
            }
        }
        for b in 0..value.len() {
            value[b] += residual[b] / (mass + 3.);
        }
        value
    }
    pub fn constraint(&self, query: f64) -> Option<(Vec<f64>, f64)> {
        let mut left: Vec<_> = self
            .train
            .iter()
            .copied()
            .filter(|&i| self.times[i] <= query)
            .collect();
        let mut right: Vec<_> = self
            .train
            .iter()
            .copied()
            .filter(|&i| self.times[i] >= query)
            .collect();
        left.sort_by(|a, b| self.times[*b].total_cmp(&self.times[*a]));
        right.sort_by(|a, b| self.times[*a].total_cmp(&self.times[*b]));
        if left.is_empty() || right.is_empty() {
            return None;
        }
        let distance = ((query - self.times[left[0]]) * (self.times[right[0]] - query))
            .max(0.)
            .sqrt();
        let side = |ids: &[usize]| {
            let mut mass = 0.;
            let mut time = 0.;
            let mut r = vec![0.; self.values.len()];
            for &i in ids.iter().take(3) {
                let w = (-(self.times[i] - query).abs() / 30.).exp();
                let base = self.template(self.times[i]);
                mass += w;
                time += w * self.times[i];
                for b in 0..r.len() {
                    r[b] += w * (self.values[b][i] - base[b]);
                }
            }
            if mass < 1e-100 {
                return (self.times[ids[0]], vec![0.; r.len()]);
            }
            for v in &mut r {
                *v /= mass;
            }
            (time / mass, r)
        };
        let (tl, rl) = side(&left);
        let (tr, rr) = side(&right);
        let fraction = if tr > tl {
            ((query - tl) / (tr - tl)).clamp(0., 1.)
        } else {
            0.5
        };
        let mut value = self.template(query);
        for b in 0..value.len() {
            value[b] += (-distance / 60.).exp() * ((1. - fraction) * rl[b] + fraction * rr[b]);
        }
        Some((value, 1. - (-(distance / 15.).powi(2)).exp()))
    }
    #[cfg(test)]
    pub fn rows(&self, strength: f64) -> Vec<(f64, Vec<f64>, f64)> {
        let low = self
            .train
            .iter()
            .map(|&i| self.times[i])
            .fold(f64::INFINITY, f64::min);
        let high = self
            .train
            .iter()
            .map(|&i| self.times[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut rows = vec![];
        let mut q = low + 7.5;
        while q < high {
            if let Some((y, w)) = self.constraint(q) {
                if w > 1e-8 {
                    rows.push((q, y, strength * w));
                }
            }
            q += 15.;
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn constant_model() -> diag::NufrostVectorResult {
        diag::NufrostVectorResult {
            valid: true,
            n_bands: 2,
            freqs: vec![1. / (365.25 * 86400.)],
            t_min: 0.,
            t_rel_mean: 0.,
            centers: vec![100., 200.],
            scales: vec![1., 1.],
            theta: ndarray::Array2::zeros((3, 2)),
            include_trend: false,
        }
    }
    #[test]
    fn exact_observation_has_zero_prior_strength() {
        let times = [0., 10., 20., 30.];
        let values = vec![vec![100.; 4], vec![200.; 4]];
        let train = [0, 1, 2, 3];
        let model = constant_model();
        let support = SeasonalSupport {
            times: &times,
            values: &values,
            train: &train,
            seasonal: &model,
            phase_transfer: false,
        };
        assert_eq!(support.constraint(10.).unwrap().1, 0.);
        assert!(support.constraint(15.).unwrap().1 > 0.);
        assert!(support.constraint(-1.).is_none());
    }
    #[test]
    fn hidden_values_cannot_change_prior() {
        let times = [0., 10., 365.25, 375.25, 730.5];
        let clean = vec![vec![100.; 5], vec![200.; 5]];
        let mut poisoned = clean.clone();
        poisoned[0][2] = 1e100;
        poisoned[1][2] = -1e100;
        let train = [0, 1, 3, 4];
        let model = constant_model();
        for phase_transfer in [false, true] {
            let a = SeasonalSupport {
                times: &times,
                values: &clean,
                train: &train,
                seasonal: &model,
                phase_transfer,
            };
            let b = SeasonalSupport {
                times: &times,
                values: &poisoned,
                train: &train,
                seasonal: &model,
                phase_transfer,
            };
            assert_eq!(a.rows(6.), b.rows(6.));
        }
    }
}
