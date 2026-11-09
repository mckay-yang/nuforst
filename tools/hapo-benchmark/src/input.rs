use anyhow::{ensure, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant,
};
#[path = "source_policy.rs"]
mod source_policy;
#[derive(Clone, Deserialize)]
struct Task {
    task_id: usize,
    sample_idx: usize,
    site: usize,
    target_index: usize,
    target_epoch: f64,
    tile: String,
    n_train: usize,
    method: String,
    #[serde(rename = "truth_B2")]
    t0: f64,
    #[serde(rename = "truth_B3")]
    t1: f64,
    #[serde(rename = "truth_B4")]
    t2: f64,
    #[serde(rename = "truth_B8")]
    t3: f64,
    #[serde(rename = "truth_B11")]
    t4: f64,
    #[serde(rename = "truth_B12")]
    t5: f64,
    #[serde(rename = "pred_B2")]
    p0: f64,
    #[serde(rename = "pred_B3")]
    p1: f64,
    #[serde(rename = "pred_B4")]
    p2: f64,
    #[serde(rename = "pred_B8")]
    p3: f64,
    #[serde(rename = "pred_B11")]
    p4: f64,
    #[serde(rename = "pred_B12")]
    p5: f64,
}
impl Task {
    fn truth(&self) -> Vec<f64> {
        vec![self.t0, self.t1, self.t2, self.t3, self.t4, self.t5]
    }
    fn old(&self) -> Vec<f64> {
        vec![self.p0, self.p1, self.p2, self.p3, self.p4, self.p5]
    }
}
struct Cache {
    meta: Value,
    data: Mmap,
    mask: Mmap,
    index: Mmap,
    times: Vec<f64>,
    sources: std::collections::BTreeMap<String, source_policy::SceneSources>,
    mt: usize,
}
fn mmap(p: impl AsRef<Path>) -> Result<Mmap> {
    Ok(unsafe { Mmap::map(&File::open(p)?)? })
}
struct Input {
    days: Vec<f64>,
    obs: Vec<Vec<f64>>,
    query: f64,
    phase_offset: f64,
}
impl Cache {
    fn open(root: &Path) -> Result<Self> {
        let p = root.join("cache/sentinel-2/100k");
        let meta: Value = serde_json::from_slice(&fs::read(p.join("meta.json"))?)?;
        let mt = meta["max_times"].as_u64().context("mt")? as usize;
        ensure!(
            meta["n_samples"] == 100000
                && meta["n_bands"] == 6
                && meta["layout"] == "sample_time_band",
            "cache layout"
        );
        let data = mmap(p.join("samples.f32.bin"))?;
        let mask = mmap(p.join("mask.u8.bin"))?;
        let index = mmap(p.join("index.u64.bin"))?;
        ensure!(
            data.len() == 100000 * mt * 24
                && mask.len() == 100000 * mt
                && index.len() == 100000 * 32,
            "cache size"
        );
        Ok(Self {
            meta,
            data,
            mask,
            index,
            times: fs::read(p.join("scene_times.f64.bin"))?
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            sources: serde_json::from_slice(&fs::read(
                root.join("tests/paper_supported_rerun_20261010/sources.json"),
            )?)?,
            mt,
        })
    }
    fn value(&self, s: usize, i: usize, b: usize) -> f64 {
        let o = (s * self.mt * 6 + i * 6 + b) * 4;
        f32::from_le_bytes(self.data[o..o + 4].try_into().unwrap()) as f64
    }
    fn input(&self, t: &Task) -> Result<Input> {
        let s = t.sample_idx;
        ensure!(s < 100000, "sample bounds");
        ensure!(
            u64::from_le_bytes(self.index[s * 32..s * 32 + 8].try_into().unwrap()) as usize
                == t.site,
            "site identity"
        );
        let sc = &self.meta["scenes"][t.site];
        let n = sc["n_times"].as_u64().unwrap() as usize;
        let off = sc["time_offset"].as_u64().unwrap() as usize;
        let ts = &self.times[off..off + n];
        let provenance = &self.sources[&t.site.to_string()];
        ensure!(
            provenance.sources.len() == n && provenance.tiles.len() == n,
            "sources length"
        );
        let valid: Vec<_> = (0..n)
            .map(|i| {
                self.mask[s * self.mt + i] != 0
                    && (0..6).all(|b| {
                        let v = self.value(s, i, b);
                        v.is_finite() && v > 0. && v < 10000.
                    })
            })
            .collect();
        let selected = provenance.select(ts, &valid);
        ensure!(
            selected.len() == t.n_train + 1 && selected.contains(&t.target_index),
            "source/retained count mismatch"
        );
        ensure!(
            ts[t.target_index] == t.target_epoch && provenance.tiles[t.target_index] == t.tile,
            "target identity"
        );
        for (b, v) in t.truth().iter().enumerate() {
            ensure!(
                (self.value(s, t.target_index, b) - v).abs() < 1e-5,
                "truth identity"
            )
        }
        let origin = ts.iter().copied().fold(f64::INFINITY, f64::min);
        let days: Vec<_> = ts.iter().map(|v| (v - origin) / 86400.).collect();
        let mut obs = vec![vec![f64::NAN; n]; 6];
        let mut retained_days = vec![];
        for i in selected {
            if ts[i] == t.target_epoch {
                continue;
            }
            let row: Vec<_> = (0..6).map(|b| self.value(s, i, b)).collect();
            for b in 0..6 {
                obs[b][i] = row[b]
            }
            retained_days.push(days[i]);
        }
        ensure!(
            retained_days.len() == t.n_train && obs.iter().all(|r| r[t.target_index].is_nan()),
            "whole-target mask"
        );
        Ok(Input {
            query: days[t.target_index],
            phase_offset: origin / 86400.,
            days,
            obs,
        })
    }
}
