use anyhow::{ensure,Result};
use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::{Value,json};
use std::{collections::BTreeMap,fs::{self,File},io::Write,path::Path,time::Instant};
use ndarray::Array2;
#[path="support/paper_source_policy.rs"] mod source;
#[path = "support/paper_inputs.rs"]
mod paper_inputs;

fn main()->Result<()> {
    let data_root = paper_inputs::data_root()?;
 let a:Vec<_>=std::env::args().collect();let site:usize=a[1].parse()?;let out=Path::new(&a[2]);let maxrows:usize=a.get(3).map(|s|s.parse().unwrap()).unwrap_or(usize::MAX);
 fs::create_dir_all(out)?;let threads=std::env::var("RAYON_NUM_THREADS").ok().map(|s|s.parse::<usize>()).transpose()?.unwrap_or(std::thread::available_parallelism()?.get());ensure!(threads>0,"thread count must be positive");rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;eprintln!("scene worker threads: {threads}");
 let base=&data_root;let sample:Value=serde_json::from_slice(&fs::read(base.join("cache/sentinel-2/100k/meta.json"))?)?;
 let scene=&sample["scenes"][site];let lon=scene["lon"].as_f64().unwrap();let lat=scene["lat"].as_f64().unwrap();
 let cache=base.join(format!("cache/sentinel-2/16-sites/lon{lon:.6}_lat{lat:.6}"));let meta:Value=serde_json::from_slice(&fs::read(cache.join("meta.json"))?)?;
 let rows=meta["rows"].as_u64().unwrap() as usize;let cols=meta["cols"].as_u64().unwrap() as usize;let nt=scene["n_times"].as_u64().unwrap() as usize;
 let sources:BTreeMap<String,source::SceneSources>=serde_json::from_slice(&fs::read(base.join("tests/paper_supported_rerun_20261010/sources.json"))?)?;let provenance=&sources[&site.to_string()];
 let bm=meta["band_meta"].as_array().unwrap();let bands:Vec<String>=meta["bands"].as_array().unwrap().iter().map(|x|x.as_str().unwrap().to_string()).collect();
 ensure!(bands==["B2","B3","B4","B8","B11","B12"]);ensure!(nt==provenance.sources.len());
 let labels:Vec<_>=bm[0]["timestamps"].as_array().unwrap().iter().map(|x|x.as_str().unwrap()).collect();
 for b in bm {ensure!(b["timestamps"]==bm[0]["timestamps"]);}
 for (i,l) in labels.iter().enumerate(){ensure!(provenance.sources[i].starts_with(l));}
 let epochs:Vec<f64>=labels.iter().map(|t|nufrost_core::parse_iso8601_to_epoch_seconds(t).unwrap()).collect();
 let targetjson:Value=serde_json::from_slice(&fs::read(cache.join("target_timestamps.json"))?)?;
 let target=targetjson["targets"]["min_valid_ratio=0.900000;late_fraction=0.250000"]["target_time"].as_str().unwrap();
 let targetepoch=nufrost_core::parse_iso8601_to_epoch_seconds(target).ok_or_else(||anyhow::anyhow!("invalid target"))?;ensure!(epochs.contains(&targetepoch));
 let t0=epochs.iter().copied().fold(f64::INFINITY,f64::min);let days:Vec<_>=epochs.iter().map(|t|(t-t0)/86400.).collect();let query=(targetepoch-t0)/86400.;
 let offsets:Vec<_>=bm.iter().map(|b|b["offset_values"].as_u64().unwrap() as usize).collect();
 let data=unsafe{Mmap::map(&File::open(cache.join("cube.f32.bin"))?)?};
 let cfg=nufrost_core::NufrostConfig::from_json(paper_inputs::CONFIG)?;
 let signature=json!({"site":site,"rows":rows,"cols":cols,"times":nt,"target":target,"model":"nufrost-supported-v1","source_policy":"max unique valid dates; lexical ties; target excluded","channels":["truth","nufrost","original","hants","zhu2015"],"bands":bands});
 let mp=out.join("manifest.json");if mp.exists(){ensure!(serde_json::from_slice::<Value>(&fs::read(&mp)?)?==signature,"incompatible existing output");}else{fs::write(&mp,serde_json::to_vec_pretty(&signature)?)?;}
 let start=Instant::now();
 for row in (0..rows.min(maxrows)).step_by(4) {
  let end=(row+4).min(rows).min(maxrows);let path=out.join(format!("rows_{row:04}_{end:04}.f32"));let pixels=(end-row)*cols;
  if path.exists(){ensure!(fs::metadata(&path)?.len()==(pixels*30*4) as u64);continue;}
  // Transpose one strip to keep each pixel's time series contiguous during fitting.
  let mut strip=vec![f32::NAN;pixels*nt*6];
  for b in 0..6 {for t in 0..nt {let off=(offsets[b]+t*rows*cols+row*cols)*4;for p in 0..pixels {strip[(p*nt+t)*6+b]=f32::from_le_bytes(data[off+p*4..off+p*4+4].try_into().unwrap());}}}
  let mut output=vec![f32::NAN;pixels*30];
  output.par_chunks_mut(30).enumerate().try_for_each(|(p,dst)|->Result<()> {
   let v=&strip[p*nt*6..(p+1)*nt*6];let valid:Vec<_>=(0..nt).map(|i|(0..6).all(|b|v[i*6+b].is_finite()&&v[i*6+b]>0.&&v[i*6+b]<10000.)).collect();
   let selected=provenance.select(&epochs,&valid);if selected.len()<13{return Ok(());}
   let Some(&targetindex)=selected.iter().find(|&&i|epochs[i]==targetepoch)else{return Ok(());};
   let mut obs=vec![vec![f64::NAN;nt];6];for &i in &selected {for b in 0..6{if i==targetindex{dst[b]=v[i*6+b];}else{obs[b][i]=v[i*6+b] as f64;}}}
   let m=nufrost_core::supported::fit_supported_vector(&days,&obs,&cfg)?;
   let new=nufrost_core::nufrost_predict_vector(&m.model,query);let old=nufrost_core::nufrost_pixel_vector(&days,&obs,query,&cfg);
   for b in 0..6 {dst[6+b]=new[b] as f32;dst[12+b]=old[b] as f32;dst[18+b]=hants_core::hants_pixel(&days,&obs[b],query,3,"high",None,None,500.,5,365.25) as f32;dst[24+b]=zhu2015_core::fit_predict_pixel(&days,&obs[b],query,0.1).prediction as f32;}Ok(())
  })?;
  let temp=path.with_extension("partial");let mut f=File::create(&temp)?;for v in &output{f.write_all(&v.to_le_bytes())?;}f.sync_all()?;fs::rename(temp,path)?;
  eprintln!("site{site} rows {end}/{rows}, {:.1}s, valid targets {}/{}",start.elapsed().as_secs_f64(),output.chunks_exact(30).filter(|x|x[0].is_finite()).count(),pixels);
 }
 if maxrows<rows{return Ok(());}
 let mut arrays:Vec<_>=(0..30).map(|_|Array2::<f64>::from_elem((rows,cols),f64::NAN)).collect();
 for row in (0..rows).step_by(4){let end=(row+4).min(rows);let bytes=fs::read(out.join(format!("rows_{row:04}_{end:04}.f32")))?;for (p,values) in bytes.chunks_exact(120).enumerate(){for b in 0..30{arrays[b][[row+p/cols,p%cols]]=f32::from_le_bytes(values[b*4..b*4+4].try_into().unwrap()) as f64;}}}
 let geo: [f64;6]=serde_json::from_value(meta["geo_transform"].clone())?;let raster=gdal::RasterMetadata{geo_transform:geo,crs_wkt:meta["crs_wkt"].as_str().map(str::to_string),nodata:Some(f64::NAN)};
 for (i,name) in ["truth","nufrost","original","hants","zhu2015"].iter().enumerate(){let maps:BTreeMap<_,_>=bands.iter().enumerate().map(|(b,n)|(n.clone(),arrays[i*6+b].clone())).collect();gdal::full_scene::write_band_stack(&out.join(format!("{name}.tif")),&maps,&bands,&raster)?;}
 fs::write(out.join("complete.json"),serde_json::to_vec_pretty(&json!({"finished":true,"elapsed_this_run":start.elapsed().as_secs_f64()}))?)?;Ok(())
}
