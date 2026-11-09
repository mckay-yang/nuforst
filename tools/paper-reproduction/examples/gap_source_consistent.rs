use anyhow::{ensure, Result};
use nufrost_core::nufrost as diag;
pub use nufrost_core::{config, nufft};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};
#[derive(Clone)]
struct Task {
    site: usize,
    sample: usize,
    target: usize,
    t: Vec<f64>,
    epoch: Vec<f64>,
    y: Vec<Vec<f64>>,
    valid: Vec<usize>,
}
struct Mask {
    gap: usize,
    kind: String,
    rep: usize,
    removed: Vec<usize>,
}
fn random(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn observations(t: &Task, removed: &[usize]) -> Vec<Vec<f64>> {
    let mut y = vec![vec![f64::NAN; t.t.len()]; 6];
    for &i in &t.valid {
        if !removed.contains(&i) {
            for b in 0..6 {
                y[b][i] = t.y[b][i]
            }
        }
    }
    y
}
#[path = "support/paper_inputs.rs"]
mod paper_inputs;

fn main() -> Result<()> {
    let data_root = paper_inputs::data_root()?;
    let args: Vec<_> = std::env::args().collect();
    let out = Path::new(&args[1]);
    let source_indices: Value = serde_json::from_slice(&fs::read(data_root.join("tests/gap_source_consistent_20261010/cohort/retained_indices.json"))?)?;
    let offset: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0);
    let take: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(20);
    ensure!(!out.exists(), "output exists");
    fs::create_dir_all(out)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()?;
    let root = data_root.join("cache/sentinel-2/100k");
    let meta: Value = serde_json::from_slice(&fs::read(root.join("meta.json"))?)?;
    let mt = meta["max_times"].as_u64().unwrap() as usize;
    let times: Vec<f64> = fs::read(root.join("scene_times.f64.bin"))?
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let masks = fs::read(root.join("mask.u8.bin"))?;
    let mut data = File::open(root.join("samples.f32.bin"))?;
    let manifest = fs::read_to_string(
        data_root.join("tests/gap_untouched_quality_20261010/manifest.csv"),
    )?;
    let mut counts = [0; 16];
    let mut tasks = vec![];
    let mut selected = String::from("sample_idx,site_id,target_index\n");
    for line in manifest.lines().skip(1) {
        let f: Vec<_> = line.split(',').collect();
        let sample: usize = f[0].parse()?;
        if source_indices.get(sample.to_string()).is_none() {continue;}
        let site: usize = f[2].parse()?;
        let target: usize = f[3].parse()?;
        let rank = counts[site];
        counts[site] += 1;
        if rank < offset || rank >= offset + take {
            continue;
        }
        let scene = &meta["scenes"][site];
        let nt = scene["n_times"].as_u64().unwrap() as usize;
        let offset = scene["time_offset"].as_u64().unwrap() as usize;
        let start = times[offset..offset + nt]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let t: Vec<_> = times[offset..offset + nt]
            .iter()
            .map(|x| (x - start) / 86400.)
            .collect();
        let mut raw = vec![0; mt * 6 * 4];
        data.seek(SeekFrom::Start((sample * mt * 6 * 4) as u64))?;
        data.read_exact(&mut raw)?;
        let vals: Vec<_> = raw
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()) as f64)
            .collect();
        let y: Vec<Vec<f64>> = (0..6)
            .map(|b| (0..nt).map(|i| vals[i * 6 + b]).collect())
            .collect();
        let valid: Vec<usize> = source_indices[sample.to_string()].as_array().unwrap().iter().map(|v|v.as_u64().unwrap() as usize).collect();
        ensure!(valid.iter().all(|&i|i<nt && masks[sample*mt+i] != 0));
        let mut unique=std::collections::HashSet::new();
        ensure!(valid.iter().all(|&i|unique.insert(times[offset+i].to_bits())));
        selected += &format!("{sample},{site},{target}\n");
        tasks.push(Task {
            site,
            sample,
            target,
            t,
            y,
            valid,
            epoch: times[offset..offset + nt].to_vec(),
        });
    }
    ensure!(tasks.len() == source_indices.as_object().unwrap().len());
    fs::write(out.join("selected.csv"), selected)?;
    let mut all_masks = vec![];
    let mut maskfile = File::create(out.join("masks.csv"))?;
    writeln!(maskfile, "sample_idx,gap,kind,rep,removed_indices")?;
    for task in &tasks {
        let mut group = vec![];
        let target = task.t[task.target];
        let baseline: Vec<_> = task
            .valid
            .iter()
            .copied()
            .filter(|i| task.t[*i] == target)
            .collect();
        let before = *task
            .valid
            .iter()
            .filter(|i| task.t[**i] < target)
            .max_by(|a, b| task.t[**a].total_cmp(&task.t[**b]))
            .unwrap();
        let after = *task
            .valid
            .iter()
            .filter(|i| task.t[**i] > target)
            .min_by(|a, b| task.t[**a].total_cmp(&task.t[**b]))
            .unwrap();
        for gap in [0, 30, 40, 60] {
            let removed: Vec<_> = task
                .valid
                .iter()
                .copied()
                .filter(|i| {
                    (task.epoch[*i] - task.epoch[task.target]).abs() / 86400. <= gap as f64 / 2.
                })
                .collect();
            let count = removed.len();
            group.push(Mask {
                gap,
                kind: "continuous".into(),
                rep: 0,
                removed,
            });
            if false && gap > 0 {
                for rep in 0..3 {
                    let mut pool: Vec<_> = task
                        .valid
                        .iter()
                        .copied()
                        .filter(|i| !baseline.contains(i) && *i != before && *i != after)
                        .collect();
                    let mut seed =
                        20261010u64 ^ (task.sample as u64 * 7919) ^ (gap as u64 * 101) ^ rep as u64;
                    for k in (1..pool.len()).rev() {
                        let j = random(&mut seed) as usize % (k + 1);
                        pool.swap(k, j)
                    }
                    let mut removed = baseline.clone();
                    ensure!(count - baseline.len() <= pool.len());
                    removed.extend_from_slice(&pool[..count - baseline.len()]);
                    removed.sort_unstable();
                    ensure!(removed.len() == count);
                    ensure!(!removed.contains(&before) && !removed.contains(&after));
                    group.push(Mask {
                        gap,
                        kind: "dispersed_local_kept".into(),
                        rep,
                        removed,
                    });
                }
            }
        }
        for m in &group {
            writeln!(
                maskfile,
                "{},{},{},{},{}",
                task.sample,
                m.gap,
                m.kind,
                m.rep,
                m.removed
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(";")
            )?;
        }
        all_masks.push(group);
    }
    drop(maskfile);
    let cfg = config::NufrostConfig::from_json(paper_inputs::CONFIG)?;
    let start = std::time::Instant::now();
    let lines: Result<Vec<String>> = tasks
        .par_iter()
        .zip(all_masks.par_iter())
        .map(|(task, masks)| -> Result<String> {
            let mut out = String::new();
            let target = task.t[task.target];
            for m in masks {
                let y = observations(task, &m.removed);
                let ntrain = task.valid.len() - m.removed.len();
                let original = diag::nufrost_fit_pixel_vector(&task.t, &y, &cfg);
                ensure!(original.valid);
                let supported = nufrost_core::supported::fit_supported_vector(&task.t, &y, &cfg)?;
                ensure!(supported.model.valid && supported.training_dates == ntrain);
                let independent: Vec<_> = (0..6)
                    .map(|b| {
                        nufrost_core::supported::fit_supported_vector(
                            &task.t,
                            &[y[b].clone()],
                            &cfg,
                        )
                    })
                    .collect::<std::result::Result<_, _>>()?;
                ensure!(independent.iter().all(|m| m.model.valid));
                let mut queries: Vec<_> = task
                    .valid
                    .iter()
                    .copied()
                    .filter(|&i| (task.t[i] - target).abs() <= 180.)
                    .map(|i| (task.t[i], Some(i)))
                    .collect();
                queries.extend((-180..=180).step_by(2).map(|d| (target + d as f64, None)));
                queries.sort_by(|a, b| a.0.total_cmp(&b.0));
                for (query, observation) in queries {
                    let predictions = vec![
                        ("original", diag::nufrost_predict_vector(&original, query)),
                        (
                            "supported_v1",
                            diag::nufrost_predict_vector(&supported.model, query),
                        ),
                        (
                            "independent_v1",
                            independent
                                .iter()
                                .map(|m| diag::nufrost_predict_vector(&m.model, query)[0])
                                .collect(),
                        ),
                        (
                            "zhu2015",
                            y.iter()
                                .map(|v| {
                                    zhu2015_core::fit_predict_pixel(&task.t, v, query, 0.1)
                                        .prediction
                                })
                                .collect(),
                        ),
                        (
                            "hants",
                            y.iter()
                                .map(|v| {
                                    hants_core::hants_pixel(
                                        &task.t, v, query, 3, "high", None, None, 500., 5, 365.25,
                                    )
                                })
                                .collect(),
                        ),
                    ];
                    for (name, prediction) in predictions {
                        ensure!(prediction.iter().all(|x| x.is_finite()));
                        for band in 0..6 {
                            let truth = observation
                                .map(|i| task.y[band][i] / 10000.)
                                .unwrap_or(f64::NAN);
                            let hidden =
                                observation.map(|i| m.removed.contains(&i)).unwrap_or(false);
                            out += &format!(
                                "{},{},{},{},{},{},{},{},{},{:.12},{:.12}\n",
                                task.sample,
                                task.site,
                                m.gap,
                                name,
                                band,
                                ntrain,
                                query - target,
                                observation.map(|i| i as i64).unwrap_or(-1),
                                hidden,
                                truth,
                                prediction[band] / 10000.
                            );
                        }
                    }
                }
            }
            Ok(out)
        })
        .collect();
    let mut file = File::create(out.join("predictions.csv"))?;
    writeln!(
        file,
        "sample_idx,site,gap,variant,band,ntrain,dt,observation_index,hidden,truth,prediction"
    )?;
    for text in lines? {
        file.write_all(text.as_bytes())?;
    }
    fs::write(
        out.join("metadata.json"),
        serde_json::to_vec_pretty(
            &json!({"completed":true,"tasks":tasks.len(),"offset":offset,"take":take,"elapsed_seconds":start.elapsed().as_secs_f64(),"model_version":nufrost_core::supported::MODEL_VERSION,"candidate_rule":"fixed source tile per pixel, unique exact timestamps, original targets, frozen model without retuning","selection":"manifest rank, no method results"}),
        )?,
    )?;
    println!(
        "Completed {} in {:.1}s",
        tasks.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
