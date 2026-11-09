use serde_json::{json, Value};
#[derive(Default, Clone)]
pub struct Metrics {
    n: usize,
    abs: [f64; 6],
    sq: [f64; 6],
    sam4: f64,
    sam6: f64,
}
impl Metrics {
    pub fn add(&mut self, p: &[f64], y: &[f64]) {
        self.n += 1;
        for b in 0..6 {
            let d = (p[b] - y[b]) / 10000.;
            self.abs[b] += d.abs();
            self.sq[b] += d * d;
        }
        for bands in [4, 6] {
            let dot: f64 = (0..bands).map(|b| p[b] * y[b]).sum();
            let den = ((0..bands).map(|b| p[b] * p[b]).sum::<f64>()
                * (0..bands).map(|b| y[b] * y[b]).sum::<f64>())
            .sqrt();
            let angle = if den > 0. {
                (dot / den).clamp(-1., 1.).acos().to_degrees()
            } else {
                f64::NAN
            };
            if bands == 4 {
                self.sam4 += angle
            } else {
                self.sam6 += angle
            }
        }
    }
    pub fn json(&self) -> Value {
        let n = self.n as f64;
        json!({"n":self.n,"mae4":self.abs[..4].iter().sum::<f64>()/(4.*n),"rmse4":(self.sq[..4].iter().sum::<f64>()/(4.*n)).sqrt(),"mae6":self.abs.iter().sum::<f64>()/(6.*n),"rmse6":(self.sq.iter().sum::<f64>()/(6.*n)).sqrt(),"sam4_deg":self.sam4/n,"sam6_deg":self.sam6/n,"band_mae":self.abs.map(|v|v/n),"band_rmse":self.sq.map(|v|(v/n).sqrt())})
    }
}
