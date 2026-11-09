use super::seasonal::SeasonalSupport;

pub struct YearAnalog<'a> {
    pub base: SeasonalSupport<'a>,
    pub years: i32,
    pub mismatch_scale: f64,
    sorted: Vec<usize>,
}
impl<'a> YearAnalog<'a> {
    pub fn new(base: SeasonalSupport<'a>, years: i32, mismatch_scale: f64) -> Self {
        let mut sorted = base.train.to_vec();
        sorted.sort_by(|a, b| base.times[*a].total_cmp(&base.times[*b]));
        Self {
            base,
            years,
            mismatch_scale,
            sorted,
        }
    }
    fn observed_interpolation(&self, query: f64) -> Option<(Vec<f64>, f64)> {
        let times = self.base.times;
        let hi = self.sorted.partition_point(|&i| times[i] < query);
        if hi < self.sorted.len() && (times[self.sorted[hi]] - query).abs() < 1e-8 {
            return Some((
                self.base
                    .values
                    .iter()
                    .map(|v| v[self.sorted[hi]])
                    .collect(),
                0.,
            ));
        }
        if hi == 0 || hi == self.sorted.len() {
            return None;
        }
        let left = self.sorted[hi - 1];
        let right = self.sorted[hi];
        let width = times[right] - times[left];
        if width <= 0. || width > 60. {
            return None;
        }
        let fraction = (query - times[left]) / width;
        Some((
            self.base
                .values
                .iter()
                .map(|v| (1. - fraction) * v[left] + fraction * v[right])
                .collect(),
            width,
        ))
    }
    pub fn constraint(&self, query: f64) -> Option<(Vec<f64>, f64)> {
        let (fallback, strength) = self.base.constraint(query)?;
        let times = self.base.times;
        let hi = self.sorted.partition_point(|&i| times[i] < query);
        let context: Vec<_> = self.sorted[hi.saturating_sub(3)..(hi + 3).min(self.sorted.len())]
            .iter()
            .copied()
            .filter(|&i| (times[i] - query).abs() <= 120.)
            .collect();
        let mut mass = 0.;
        let mut transfer = vec![0.; fallback.len()];
        let mut analogs = vec![];
        for year in -self.years..=self.years {
            if year == 0 {
                continue;
            }
            let shift = year as f64 * 365.25;
            let Some((analog, width)) = self.observed_interpolation(query + shift) else {
                continue;
            };
            let mut paired = vec![];
            let mut offset = vec![0.; fallback.len()];
            let mut sum = 0.;
            let mut has_left = false;
            let mut has_right = false;
            for &i in &context {
                let Some((other, _)) = self.observed_interpolation(times[i] + shift) else {
                    continue;
                };
                let weight = (-(times[i] - query).abs() / 60.).exp();
                let residual: Vec<_> = other
                    .iter()
                    .enumerate()
                    .map(|(b, v)| self.base.values[b][i] - v)
                    .collect();
                for b in 0..offset.len() {
                    offset[b] += weight * residual[b];
                }
                sum += weight;
                has_left |= times[i] < query;
                has_right |= times[i] > query;
                paired.push((weight, residual));
            }
            if paired.len() < 3 || !has_left || !has_right {
                continue;
            }
            for v in &mut offset {
                *v /= sum;
            }
            let mismatch = (paired
                .iter()
                .map(|(w, r)| {
                    w * r
                        .iter()
                        .zip(&offset)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / offset.len() as f64
                })
                .sum::<f64>()
                / sum)
                .sqrt();
            let reliability =
                (-(mismatch / self.mismatch_scale).powi(2) - width / 60.).exp() / year.abs() as f64;
            let aligned: Vec<_> = analog.iter().zip(&offset).map(|(a, b)| a + b).collect();
            mass += reliability;
            for b in 0..transfer.len() {
                transfer[b] += reliability * aligned[b];
            }
            analogs.push((reliability, aligned));
        }
        let confidence = if mass > 1e-12 {
            let effective =
                mass * mass / analogs.iter().map(|(w, _)| w * w).sum::<f64>().max(1e-12);
            let variance = analogs
                .iter()
                .map(|(w, y)| {
                    w * y
                        .iter()
                        .enumerate()
                        .map(|(b, v)| (v - transfer[b] / mass).powi(2))
                        .sum::<f64>()
                        / fallback.len() as f64
                })
                .sum::<f64>()
                / mass;
            (effective - 1.).clamp(0., 1.) * (-variance / (200. * 200.)).exp()
        } else {
            0.
        };
        let value = fallback
            .iter()
            .enumerate()
            .map(|(b, v)| (0.5 * v + confidence * transfer[b]) / (0.5 + confidence * mass))
            .collect();
        Some((value, strength))
    }
    pub fn rows(&self, strength: f64) -> Vec<(f64, Vec<f64>, f64)> {
        let low = self.base.times[*self.sorted.first().unwrap()];
        let high = self.base.times[*self.sorted.last().unwrap()];
        let mut q = low + 7.5;
        let mut rows = vec![];
        while q < high {
            if let Some((v, w)) = self.constraint(q) {
                if w > 1e-8 {
                    rows.push((q, v, strength * w));
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
    use crate::nufrost as diag;
    fn season() -> diag::NufrostVectorResult {
        diag::NufrostVectorResult {
            valid: true,
            n_bands: 2,
            freqs: vec![1. / (365.25 * 86400.)],
            t_min: 0.,
            t_rel_mean: 0.,
            centers: vec![1000., 2000.],
            scales: vec![1., 1.],
            theta: ndarray::Array2::zeros((3, 2)),
            include_trend: false,
        }
    }
    fn fixture(amplitudes: [f64; 3]) -> (Vec<f64>, Vec<Vec<f64>>, Vec<usize>) {
        let times: Vec<_> = (0..=1096).map(|i| i as f64).collect();
        let centers = [200., 565.25, 930.5];
        let values: Vec<Vec<f64>> = [1000., 2000.]
            .iter()
            .map(|base| {
                times
                    .iter()
                    .map(|&t| {
                        base + centers
                            .iter()
                            .zip(amplitudes)
                            .map(|(&c, a)| a * (-((t - c) / 8.).powi(2)).exp())
                            .sum::<f64>()
                    })
                    .collect()
            })
            .collect();
        let train = (0..times.len())
            .filter(|&i| (times[i] - 565.25).abs() > 20.)
            .collect();
        (times, values, train)
    }
    #[test]
    fn consistent_other_years_supply_missing_seasonal_peak() {
        let (t, y, train) = fixture([100., 100., 100.]);
        let model = season();
        let base = SeasonalSupport {
            times: &t,
            values: &y,
            train: &train,
            seasonal: &model,
            phase_transfer: false,
        };
        let fallback = base.constraint(565.25).unwrap().0;
        let analog = YearAnalog::new(base, 1, 200.);
        let predicted = analog.constraint(565.25).unwrap().0;
        assert!(predicted[0] > fallback[0] + 40.);
    }
    #[test]
    fn conflicting_years_do_not_inject_a_large_peak() {
        let (t, y, train) = fixture([1000., 0., 0.]);
        let model = season();
        let base = SeasonalSupport {
            times: &t,
            values: &y,
            train: &train,
            seasonal: &model,
            phase_transfer: false,
        };
        let fallback = base.constraint(565.25).unwrap().0;
        let predicted = YearAnalog::new(base, 1, 200.).constraint(565.25).unwrap().0;
        assert!((predicted[0] - fallback[0]).abs() < 10.);
    }
    #[test]
    fn masked_truth_cannot_enter_year_transfer() {
        let (t, y, train) = fixture([100., 100., 100.]);
        let model = season();
        let mut poison = y.clone();
        for i in 0..t.len() {
            if !train.contains(&i) {
                poison[0][i] = 1e100;
                poison[1][i] = -1e100;
            }
        }
        let a = YearAnalog::new(
            SeasonalSupport {
                times: &t,
                values: &y,
                train: &train,
                seasonal: &model,
                phase_transfer: false,
            },
            2,
            200.,
        );
        let b = YearAnalog::new(
            SeasonalSupport {
                times: &t,
                values: &poison,
                train: &train,
                seasonal: &model,
                phase_transfer: false,
            },
            2,
            200.,
        );
        assert_eq!(a.constraint(565.25), b.constraint(565.25));
    }
}
