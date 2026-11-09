//! Prepare raw, source-derived preview panels using the manuscript stretch.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{fs, path::Path};
fn percentile(v: &[f64], q: f64) -> f64 {
    let k = q * (v.len() - 1) as f64;
    let i = k.floor() as usize;
    v[i] + (v[k.ceil() as usize] - v[i]) * k.fract()
}
fn palette(x: f64) -> [u8; 3] {
    let x = x.clamp(-1., 1.);
    let a = if x < 0. {
        [42., 137., 177.]
    } else {
        [214., 47., 39.]
    };
    let w = x.abs();
    std::array::from_fn(|i| (255. * (1. - w) + a[i] * w).round() as u8)
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let root = Path::new(a.get(1).context("data root")?);
    let scene = Path::new(a.get(2).context("scene output")?);
    let meta: Value = serde_json::from_slice(&fs::read(scene.join("manifest.json"))?)?;
    let site = meta["site"].as_u64().context("site")?;
    let old = root.join(format!(
        "tests/paper_supported_rerun_20261010/scenes/site_{site}"
    ));
    let mut arrays = vec![];
    let (mut rows, mut cols) = (0, 0);
    for name in ["truth", "nufrost", "hapo", "zhu2015", "hants"] {
        let p = if name == "hapo" {
            scene.join("hapo.tif")
        } else {
            old.join(format!("{name}.tif"))
        };
        let r = raster::RasterReader::open(p)?;
        if arrays.is_empty() {
            (rows, cols) = r.shape();
        }
        ensure!(r.shape() == (rows, cols) && r.band_count() == 6, "shape");
        arrays.push(
            (1..=6)
                .map(|b| r.read_band(b))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    let common: Vec<_> = (0..rows * cols)
        .map(|p| {
            arrays
                .iter()
                .all(|s| s.iter().all(|b| b[[p / cols, p % cols]].is_finite()))
        })
        .collect();
    let n = common.iter().filter(|&&v| v).count();
    ensure!(n > 0, "no common pixels");
    let mut stretch = vec![];
    for b in [2, 1, 0] {
        let mut v: Vec<_> = (0..rows * cols)
            .filter(|&p| common[p])
            .map(|p| arrays[0][b][[p / cols, p % cols]])
            .collect();
        v.sort_by(f64::total_cmp);
        stretch.push((percentile(&v, 0.02), percentile(&v, 0.98)));
    }
    let out = scene.join("preview-panels");
    ensure!(!out.exists(), "new preview directory required");
    fs::create_dir(&out)?;
    for (m, name) in ["Observed", "NUFROST", "HAPO", "LHR", "HANTS"]
        .iter()
        .enumerate()
    {
        let mut rgb = vec![255u8; rows * cols * 3];
        let mut b11 = rgb.clone();
        let mut b12 = rgb.clone();
        for p in 0..rows * cols {
            if !common[p] {
                continue;
            }
            let rr = p / cols;
            let cc = p % cols;
            for (k, b) in [2, 1, 0].iter().enumerate() {
                let (lo, hi) = stretch[k];
                rgb[p * 3 + k] = if hi > lo {
                    ((arrays[m][*b][[rr, cc]] - lo) / (hi - lo) * 255.)
                        .clamp(0., 255.)
                        .round() as u8
                } else {
                    128
                };
            }
            for (b, panel) in [(4, &mut b11), (5, &mut b12)] {
                let v = arrays[m][b][[rr, cc]];
                let color = if m == 0 {
                    let u = (v / 4000.).clamp(0., 1.);
                    [255, (255. * u).round() as u8, (255. * u).round() as u8]
                } else {
                    palette((v - arrays[0][b][[rr, cc]]) / 2000.)
                };
                panel[p * 3..p * 3 + 3].copy_from_slice(&color);
            }
        }
        fs::write(out.join(format!("{name}-RGB.raw")), rgb)?;
        fs::write(out.join(format!("{name}-B11.raw")), b11)?;
        fs::write(out.join(format!("{name}-B12.raw")), b12)?;
    }
    fs::write(
        out.join("meta.json"),
        serde_json::to_vec_pretty(
            &json!({"rows":rows,"cols":cols,"common_pixels":n,"rgb_stretch":stretch,"columns":["Observed","NUFROST","HAPO","LHR","HANTS"],"rows_of_panels":["RGB","B11","B12"],"residual_range_dn":[-2000,2000],"observed_swir_range_dn":[0,4000],"target":meta["target"],"site":site}),
        )?,
    )?;
    Ok(())
}
