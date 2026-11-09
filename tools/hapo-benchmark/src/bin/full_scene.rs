//! Full-scene HAPO using the paper's frozen per-pixel source and target policy.
use anyhow::{ensure, Context, Result};
use memmap2::Mmap;
use ndarray::Array2;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::Path,
    time::Instant,
};
#[path = "../metrics.rs"]
mod metrics;
#[path = "../source_policy.rs"]
mod source_policy;
use metrics::Metrics;
fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    let temp = path.with_extension("partial");
    fs::write(&temp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temp, path)?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(args.get(1).context("data root")?);
    let out = Path::new(args.get(2).context("new output root")?);
    let sites: Vec<usize> = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("0,5,15")
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let maxrows: usize = args
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(usize::MAX);
    ensure!(!out.exists(), "refuse an existing output root");
    fs::create_dir_all(out)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(10)
        .build_global()?;
    let sample: Value =
        serde_json::from_slice(&fs::read(root.join("cache/sentinel-2/100k/meta.json"))?)?;
    let sources: BTreeMap<String, source_policy::SceneSources> = serde_json::from_slice(
        &fs::read(root.join("tests/paper_supported_rerun_20261010/sources.json"))?,
    )?;
    let mut summaries = vec![];
    let total_start = Instant::now();
    for site in sites {
        let sc = &sample["scenes"][site];
        let lon = sc["lon"].as_f64().context("lon")?;
        let lat = sc["lat"].as_f64().context("lat")?;
        let cache = root.join(format!("cache/sentinel-2/16-sites/lon{lon:.6}_lat{lat:.6}"));
        let old = root.join(format!(
            "tests/paper_supported_rerun_20261010/scenes/site_{site}"
        ));
        let meta: Value = serde_json::from_slice(&fs::read(cache.join("meta.json"))?)?;
        let paper: Value = serde_json::from_slice(&fs::read(old.join("manifest.json"))?)?;
        let rows = meta["rows"].as_u64().context("rows")? as usize;
        let cols = meta["cols"].as_u64().context("cols")? as usize;
        let nt = sc["n_times"].as_u64().context("times")? as usize;
        let target = paper["target"].as_str().context("paper target")?;
        ensure!(
            paper["rows"] == rows && paper["cols"] == cols && paper["times"] == nt,
            "paper grid/time mismatch"
        );
        let bands: Vec<String> = serde_json::from_value(meta["bands"].clone())?;
        ensure!(bands == ["B2", "B3", "B4", "B8", "B11", "B12"]);
        let bm = meta["band_meta"].as_array().context("band metadata")?;
        ensure!(bm.len() == 6, "band count");
        let provenance = &sources[&site.to_string()];
        ensure!(
            nt == provenance.sources.len() && nt == provenance.tiles.len(),
            "source lengths"
        );
        let labels: Vec<String> = serde_json::from_value(bm[0]["timestamps"].clone())?;
        ensure!(labels.len() == nt, "time count");
        for b in bm {
            ensure!(b["timestamps"] == bm[0]["timestamps"], "unaligned bands");
        }
        for (i, l) in labels.iter().enumerate() {
            ensure!(provenance.sources[i].starts_with(l), "source order");
        }
        let epochs: Vec<f64> = labels
            .iter()
            .map(|t| raster::parse_iso8601_to_epoch_seconds(t).context("timestamp"))
            .collect::<Result<_>>()?;
        let targetepoch = raster::parse_iso8601_to_epoch_seconds(target).context("target")?;
        ensure!(epochs.contains(&targetepoch), "target absent");
        let days: Vec<_> = epochs.iter().map(|t| t / 86400. + 719163.).collect();
        let query = targetepoch / 86400. + 719163.;
        let offsets: Vec<_> = bm
            .iter()
            .map(|b| b["offset_values"].as_u64().unwrap() as usize)
            .collect();
        let data = unsafe { Mmap::map(&File::open(cache.join("cube.f32.bin"))?)? };
        ensure!(
            data.len() == meta["total_values"].as_u64().context("total values")? as usize * 4,
            "cache size"
        );
        let geo: [f64; 6] = serde_json::from_value(meta["geo_transform"].clone())?;
        let truth_reader = raster::RasterReader::open(old.join("truth.tif"))?;
        ensure!(
            truth_reader.shape() == (rows, cols) && truth_reader.band_count() == 6,
            "truth shape"
        );
        ensure!(truth_reader.geo_transform() == Some(geo), "truth affine");
        let truth: Vec<_> = (1..=6)
            .map(|b| truth_reader.read_band(b))
            .collect::<Result<_>>()?;
        let truth_wkt = truth_reader.crs_wkt();
        let crs = meta["crs_wkt"].as_str().map(str::to_string);
        ensure!(truth_wkt == crs, "truth CRS");
        let siteout = out.join(format!("site_{site}"));
        fs::create_dir(&siteout)?;
        atomic_json(
            &siteout.join("manifest.json"),
            &json!({"site":site,"rows":rows,"cols":cols,"times":nt,"bands":bands,"target":target,"source_cache":cache,"paper_reference":old,"threads":10,"algorithm":"HAPO alpha=1 three annual harmonics; same core and solver as 100k run","source_policy":"max unique jointly-valid dates per tile; lexical ties; frozen before target holdout","eligibility":"target belongs to selected source; at least 12 retained six-band observations","mask":"all acquisitions at target epoch excluded across six bands","units":"raw DN; divide by 10000 for reflectance; no clipping"}),
        )?;
        let start = Instant::now();
        let nr = rows.min(maxrows);
        let mut arrays: Vec<_> = (0..6)
            .map(|_| Array2::<f64>::from_elem((rows, cols), f64::NAN))
            .collect();
        let mut eligible = 0usize;
        let mut failed = 0usize;
        let mut strict = 0usize;
        let mut loose = 0usize;
        let mut cap = 0usize;
        let mut paper_target_count = 0usize;
        for row in (0..nr).step_by(8) {
            let end = (row + 8).min(nr);
            let pixels = (end - row) * cols;
            let mut strip = vec![f32::NAN; pixels * nt * 6];
            for b in 0..6 {
                for t in 0..nt {
                    let offset = (offsets[b] + t * rows * cols + row * cols) * 4;
                    for p in 0..pixels {
                        strip[(p * nt + t) * 6 + b] = f32::from_le_bytes(
                            data[offset + p * 4..offset + p * 4 + 4].try_into().unwrap(),
                        );
                    }
                }
            }
            let mut records = vec![[f64::NAN; 9]; pixels];
            records
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(p, dst)| -> Result<()> {
                    let v = &strip[p * nt * 6..(p + 1) * nt * 6];
                    let valid: Vec<_> = (0..nt)
                        .map(|i| {
                            (0..6).all(|b| {
                                let x = v[i * 6 + b];
                                x.is_finite() && x > 0. && x < 10000.
                            })
                        })
                        .collect();
                    let selected = provenance.select(&epochs, &valid);
                    let rr = row + p / cols;
                    let cc = p % cols;
                    let was_valid = (0..6).all(|b| truth[b][[rr, cc]].is_finite());
                    let targetindex = selected
                        .iter()
                        .find(|&&i| epochs[i] == targetepoch)
                        .copied();
                    let available = selected.len() >= 13 && targetindex.is_some();
                    ensure!(
                        available == was_valid,
                        "eligibility differs from paper site {site} row {rr} col {cc}"
                    );
                    if !available {
                        return Ok(());
                    }
                    let ti = targetindex.unwrap();
                    for b in 0..6 {
                        ensure!(
                            v[ti * 6 + b] as f64 == truth[b][[rr, cc]],
                            "truth source mismatch"
                        );
                    }
                    let train: Vec<_> = selected
                        .into_iter()
                        .filter(|&i| epochs[i] != targetepoch)
                        .collect();
                    let times: Vec<_> = train.iter().map(|&i| days[i]).collect();
                    ensure!(times.len() >= 12, "retained count");
                    dst[6] = 0.;
                    dst[7] = 0.;
                    dst[8] = 0.;
                    for b in 0..6 {
                        let values: Vec<_> = train.iter().map(|&i| v[i * 6 + b] as f64).collect();
                        match hapo_core::fit(&times, &values, None) {
                            Ok(f) => {
                                dst[b] = f.predict(query);
                                if f.stationarity_inf < 1e-5 {
                                    dst[6] += 1.;
                                }
                                if f.stationarity_inf < 1e-4 {
                                    dst[7] += 1.;
                                }
                                if f.iterations == 1000 {
                                    dst[8] += 1.;
                                }
                            }
                            Err(_) => {
                                dst[b] = f64::NAN;
                            }
                        }
                    }
                    Ok(())
                })?;
            let mut bytes = Vec::with_capacity(pixels * 9 * 4);
            for (p, r) in records.iter().enumerate() {
                if r[6].is_finite() {
                    eligible += 1;
                    if r[..6].iter().any(|v| !v.is_finite()) {
                        failed += 1;
                    }
                    strict += r[6] as usize;
                    loose += r[7] as usize;
                    cap += r[8] as usize;
                }
                if truth[0][[row + p / cols, p % cols]].is_finite() {
                    paper_target_count += 1;
                }
                for b in 0..6 {
                    arrays[b][[row + p / cols, p % cols]] = r[b] as f32 as f64;
                }
                for v in r {
                    bytes.extend_from_slice(&(*v as f32).to_le_bytes());
                }
            }
            let file = siteout.join(format!("rows_{row:04}_{end:04}.f32"));
            let tmp = file.with_extension("partial");
            fs::write(&tmp, &bytes)?;
            fs::rename(tmp, file)?;
            let progress = json!({"site":site,"completed_rows":end,"rows":rows,"eligible_pixels":eligible,"failed_pixels":failed,"bands_stationary_1e_5":strict,"bands_stationary_1e_4":loose,"bands_at_iteration_cap":cap,"elapsed_seconds":start.elapsed().as_secs_f64()});
            atomic_json(&siteout.join("progress.json"), &progress)?;
            if end % 64 == 0 || end == nr {
                eprintln!("{progress}");
            }
        }
        if nr < rows {
            summaries.push(json!({"site":site,"smoke_only":true,"rows":nr,"eligible":eligible,"failed":failed}));
            continue;
        }
        let raster_meta = raster::RasterMetadata {
            geo_transform: geo,
            crs_wkt: crs,
            nodata: Some(f64::NAN),
        };
        let maps: BTreeMap<_, _> = bands
            .iter()
            .enumerate()
            .map(|(b, name)| (name.clone(), arrays[b].clone()))
            .collect();
        raster::full_scene::write_band_stack(
            &siteout.join("hapo.partial.tif"),
            &maps,
            &bands,
            &raster_meta,
        )?;
        fs::rename(siteout.join("hapo.partial.tif"), siteout.join("hapo.tif"))?;
        let mut preds = BTreeMap::new();
        preds.insert("hapo".to_string(), arrays);
        for method in ["nufrost", "hants", "zhu2015"] {
            let reader = raster::RasterReader::open(old.join(format!("{method}.tif")))?;
            ensure!(
                reader.shape() == (rows, cols)
                    && reader.geo_transform() == Some(geo)
                    && reader.crs_wkt() == raster_meta.crs_wkt,
                "baseline grid/CRS mismatch"
            );
            preds.insert(
                method.to_string(),
                (1..=6)
                    .map(|b| reader.read_band(b))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        let mut common = BTreeMap::<String, Metrics>::new();
        let mut common_pixels = 0;
        for r in 0..rows {
            for c in 0..cols {
                let y: Vec<_> = (0..6).map(|b| truth[b][[r, c]]).collect();
                if y.iter().any(|v| !v.is_finite())
                    || preds
                        .values()
                        .any(|v| v.iter().any(|b| !b[[r, c]].is_finite()))
                {
                    continue;
                }
                common_pixels += 1;
                for (method, values) in &preds {
                    let p: Vec<_> = values.iter().map(|a| a[[r, c]]).collect();
                    common.entry(method.clone()).or_default().add(&p, &y);
                }
            }
        }
        let output = raster::RasterReader::open(siteout.join("hapo.tif"))?;
        ensure!(
            output.shape() == (rows, cols)
                && output.band_count() == 6
                && output.geo_transform() == Some(geo)
                && output.crs_wkt() == raster_meta.crs_wkt,
            "output roundtrip grid/CRS"
        );
        for b in 0..6 {
            let band = output.read_band(b + 1)?;
            ensure!(
                band.iter()
                    .zip(preds["hapo"][b].iter())
                    .all(|(a, b)| (a.is_nan() && b.is_nan()) || a == b),
                "output pixel roundtrip"
            );
        }
        let stats: BTreeMap<_, _> = common.iter().map(|(k, v)| (k.clone(), v.json())).collect();
        let summary = json!({"site":site,"complete":true,"target":target,"rows":rows,"cols":cols,"eligible_pixels":eligible,"paper_target_pixels":paper_target_count,"failed_pixels":failed,"common_pixels":common_pixels,"bands_stationary_1e_5":strict,"bands_stationary_1e_4":loose,"bands_at_iteration_cap":cap,"elapsed_seconds":start.elapsed().as_secs_f64(),"metrics_common_support":stats,"geotiff_roundtrip_verified":true});
        atomic_json(&siteout.join("summary.json"), &summary)?;
        eprintln!("FINISHED SITE {site} {}s", start.elapsed().as_secs_f64());
        summaries.push(summary);
    }
    atomic_json(
        &out.join("summary.json"),
        &json!({"complete":maxrows==usize::MAX,"smoke":maxrows!=usize::MAX,"threads":10,"elapsed_seconds":total_start.elapsed().as_secs_f64(),"scenes":summaries}),
    )?;
    Ok(())
}
