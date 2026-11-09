use anyhow::{ensure,Result};
use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::{Value,json};
use std::{collections::BTreeMap,fs::{self,File},io::Write,path::Path,sync::{Mutex,atomic::{AtomicUsize,Ordering}},time::Instant};
#[path="support/paper_source_policy.rs"] mod source;
fn rng(s:&mut u64)->u64{*s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);*s}
fn mmap(p:impl AsRef<Path>)->Result<Mmap>{Ok(unsafe{Mmap::map(&File::open(p)?)?})}
#[path = "support/paper_inputs.rs"]
mod paper_inputs;

fn main()->Result<()> {
    let data_root = paper_inputs::data_root()?;
 let args:Vec<_>=std::env::args().collect();let out=Path::new(&args[1]);let count:usize=args.get(2).map(|s|s.parse().unwrap()).unwrap_or(100000);
 ensure!(!out.exists(),"output must be new");fs::create_dir_all(out)?;
 rayon::ThreadPoolBuilder::new().num_threads(8).build_global()?;
 let root=data_root.join("cache/sentinel-2/100k");
 let meta:Value=serde_json::from_slice(&fs::read(root.join("meta.json"))?)?;
 let sources:BTreeMap<String,source::SceneSources>=serde_json::from_slice(&fs::read(data_root.join("tests/paper_supported_rerun_20261010/sources.json"))?)?;
 let mt=meta["max_times"].as_u64().unwrap() as usize;let n=meta["n_samples"].as_u64().unwrap() as usize;
 let data=mmap(root.join("samples.f32.bin"))?;let mask=mmap(root.join("mask.u8.bin"))?;let index=mmap(root.join("index.u64.bin"))?;
 let times:Vec<f64>=fs::read(root.join("scene_times.f64.bin"))?.chunks_exact(8).map(|x|f64::from_le_bytes(x.try_into().unwrap())).collect();
 let cfg=nufrost_core::NufrostConfig::from_json(paper_inputs::CONFIG)?;
 let mut seed=20260609u64;let tasks:Vec<_>=(0..count).map(|id|{let sample=rng(&mut seed) as usize%n;let hs=rng(&mut seed);(id,sample,hs)}).collect();
 let mut writer=File::create(out.join("predictions.csv"))?;
 writeln!(writer,"task_id,sample_idx,holdout_seed,site,target_index,target_epoch,tile,n_train,method,truth_B2,truth_B3,truth_B4,truth_B8,truth_B11,truth_B12,pred_B2,pred_B3,pred_B4,pred_B8,pred_B11,pred_B12")?;
 let writer=Mutex::new(writer);let rejected=Mutex::new(File::create(out.join("excluded.csv"))?);writeln!(rejected.lock().unwrap(),"task_id,sample_idx,reason")?;
 let done=AtomicUsize::new(0);let start=Instant::now();
 tasks.par_iter().try_for_each(|&(id,sample,hs)|->Result<()> {
  let off=sample*32;let site=u64::from_le_bytes(index[off..off+8].try_into().unwrap()) as usize;
  let scene=&meta["scenes"][site];let nt=scene["n_times"].as_u64().unwrap() as usize;let to=scene["time_offset"].as_u64().unwrap() as usize;let ts=&times[to..to+nt];
  let provenance=&sources[&site.to_string()];ensure!(provenance.sources.len()==nt);
  let valid:Vec<_>=(0..nt).map(|i|mask[sample*mt+i]!=0 && (0..6).all(|b| {let o=(sample*mt*6+i*6+b)*4;let v=f32::from_le_bytes(data[o..o+4].try_into().unwrap());v.is_finite() && v>0. && v<10000.})).collect();let selected=provenance.select(ts,&valid);
  if selected.len()<13 {writeln!(rejected.lock().unwrap(),"{id},{sample},fewer_than_13_source_consistent_dates")?;} else {
   let mut local=hs;let target=selected[rng(&mut local) as usize%selected.len()];
   let time0=ts.iter().copied().fold(f64::INFINITY,f64::min);let days:Vec<_>=ts.iter().map(|t|(t-time0)/86400.).collect();
   let mut obs=vec![vec![f64::NAN;nt];6];let mut truth=vec![0.;6];
   for &i in &selected {for b in 0..6 {let o=(sample*mt*6+i*6+b)*4;let v=f32::from_le_bytes(data[o..o+4].try_into().unwrap()) as f64;ensure!(v.is_finite());if i==target {truth[b]=v;}else{obs[b][i]=v;}}}
   let model=nufrost_core::supported::fit_supported_vector(&days,&obs,&cfg)?;
   let supported=nufrost_core::nufrost_predict_vector(&model.model,days[target]);
   let original=nufrost_core::nufrost_pixel_vector(&days,&obs,days[target],&cfg);
   let hants:Vec<_>=obs.iter().map(|b|hants_core::hants_pixel(&days,b,days[target],3,"high",None,None,500.,5,365.25)).collect();
   let zhu:Vec<_>=obs.iter().map(|b|zhu2015_core::fit_predict_pixel(&days,b,days[target],0.1).prediction).collect();
   let mut lines=String::new();for (name,pred) in [("nufrost",supported),("original",original),("hants",hants),("zhu2015",zhu)] {
    lines+=&format!("{id},{sample},{hs},{site},{target},{},{},{},{}",ts[target],provenance.tiles[target],selected.len()-1,name);
    for v in truth.iter().chain(pred.iter()){lines+=&format!(",{v:.10}");}lines.push('\n');
   }writer.lock().unwrap().write_all(lines.as_bytes())?;
  }
  let d=done.fetch_add(1,Ordering::Relaxed)+1;if d%1000==0||d==count{eprintln!("{d}/{count} elapsed {:.1}s",start.elapsed().as_secs_f64());}Ok(())
 })?;
 fs::write(out.join("metadata.json"),serde_json::to_vec_pretty(&json!({"attempted":count,"elapsed_seconds":start.elapsed().as_secs_f64(),"seed":20260609,"method":"nufrost-supported-v1","methods":["nufrost","original","hants","zhu2015"],"scale":10000,"finished":true}))?)?;Ok(())
}
