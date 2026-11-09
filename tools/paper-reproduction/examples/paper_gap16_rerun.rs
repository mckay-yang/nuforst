use anyhow::{ensure,Result};
use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::{Value,json};
use std::{collections::BTreeMap,fs::{self,File},io::Write,path::Path,time::Instant};
#[path="support/paper_source_policy.rs"] mod source;
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

 let manifest=fs::read_to_string(data_root.join("products/reconstruction_evaluation/controlled_gap_v01/manifest.csv"))?;
 let tasks:Vec<(usize,usize,usize)>=manifest.lines().skip(1).take(count.min(16000)).enumerate().map(|(id,line)|{let f:Vec<_>=line.split(',').collect();(id,f[0].parse().unwrap(),f[3].parse().unwrap())}).collect();
 fs::write(out.join("original_manifest.csv"),&manifest)?;
 let start=Instant::now();
 for (chunk,group) in tasks.chunks(100).enumerate(){
  let results:Result<Vec<(String,String,String)>>=group.par_iter().map(|&(id,sample,original_target)|->Result<_>{
   let off=sample*32;let site=u64::from_le_bytes(index[off..off+8].try_into().unwrap()) as usize;let scene=&meta["scenes"][site];let nt=scene["n_times"].as_u64().unwrap() as usize;let to=scene["time_offset"].as_u64().unwrap() as usize;let ts=&times[to..to+nt];let provenance=&sources[&site.to_string()];
   let val=|i:usize,b:usize|{let o=(sample*mt*6+i*6+b)*4;f32::from_le_bytes(data[o..o+4].try_into().unwrap()) as f64};
   let valid:Vec<_>=(0..nt).map(|i|mask[sample*mt+i]!=0&&(0..6).all(|b|{let v=val(i,b);v.is_finite()&&v>0.&&v<10000.})).collect();let selected=provenance.select(ts,&valid);
   let epoch=ts[original_target];
   let Some(&target)=selected.iter().find(|&&i|ts[i]==epoch) else{return Ok((String::new(),String::new(),format!("{id},{sample},{site},{original_target},,target_unavailable_in_fixed_source\n")));};
   let before=selected.iter().filter(|&&i|ts[i]<epoch-30.*86400.).count();let after=selected.iter().filter(|&&i|ts[i]>epoch+30.*86400.).count();
   if before+after<12||before<3||after<3{return Ok((String::new(),String::new(),format!("{id},{sample},{site},{original_target},{target},insufficient_source_consistent_support_at60\n")));}
   let time0=ts.iter().copied().fold(f64::INFINITY,f64::min);let days:Vec<_>=ts.iter().map(|t|(t-time0)/86400.).collect();let truth:Vec<_>=(0..6).map(|b|val(target,b)).collect();let mut lines=String::new();let mut masks=String::new();
   for gap in [0,10,20,30,40,50,60] {
    let removed:Vec<_>=selected.iter().copied().filter(|&i|(ts[i]-epoch).abs()<=gap as f64/2.*86400.).collect();let mut obs=vec![vec![f64::NAN;nt];6];
    for &i in &selected {if !removed.contains(&i){for b in 0..6{obs[b][i]=val(i,b);}}}
    ensure!(removed.contains(&target));
    masks+=&format!("{id},{sample},{site},{gap},{target},{},{},{}\n",provenance.tiles[target],selected.iter().map(|x|x.to_string()).collect::<Vec<_>>().join(";"),removed.iter().map(|x|x.to_string()).collect::<Vec<_>>().join(";"));
    let m=nufrost_core::supported::fit_supported_vector(&days,&obs,&cfg)?;let supported=nufrost_core::nufrost_predict_vector(&m.model,days[target]);let original=nufrost_core::nufrost_pixel_vector(&days,&obs,days[target],&cfg);
    let hants:Vec<_>=obs.iter().map(|v|hants_core::hants_pixel(&days,v,days[target],3,"high",None,None,500.,5,365.25)).collect();let zhu:Vec<_>=obs.iter().map(|v|zhu2015_core::fit_predict_pixel(&days,v,days[target],0.1).prediction).collect();
    for (method,pred) in [("supported_v1",supported),("original",original),("hants",hants),("zhu2015",zhu)] {
     lines+=&format!("{id},{sample},{site},{gap},{target},{epoch},{},{},{}",provenance.tiles[target],selected.len()-removed.len(),method);
     for v in truth.iter().chain(pred.iter()){lines+=&format!(",{v:.10}");}lines.push('\n');
    }
   }
   Ok((lines,masks,format!("{id},{sample},{site},{original_target},{target},eligible\n")))
  }).collect();
  let dir=out.join(format!("chunk_{chunk:04}"));fs::create_dir(&dir)?;
  let mut predictions=File::create(dir.join("predictions.csv"))?;writeln!(predictions,"task_id,sample_idx,site,gap,target_index,target_epoch,tile,n_train,method,truth_B2,truth_B3,truth_B4,truth_B8,truth_B11,truth_B12,pred_B2,pred_B3,pred_B4,pred_B8,pred_B11,pred_B12")?;
  let mut masks=File::create(dir.join("masks.csv"))?;writeln!(masks,"task_id,sample_idx,site,gap,target_index,tile,selected_indices,removed_indices")?;
  let mut status=File::create(dir.join("status.csv"))?;writeln!(status,"task_id,sample_idx,site,original_target_index,resolved_target_index,status")?;
  for (p,m,s) in results? {predictions.write_all(p.as_bytes())?;masks.write_all(m.as_bytes())?;status.write_all(s.as_bytes())?;}
  predictions.sync_all()?;masks.sync_all()?;status.sync_all()?;fs::write(dir.join("complete.json"),"{\"finished\":true}")?;
  eprintln!("{}/{} tasks saved, {:.1}s",((chunk+1)*100).min(tasks.len()),tasks.len(),start.elapsed().as_secs_f64());
 }
 fs::write(out.join("metadata.json"),serde_json::to_vec_pretty(&json!({"attempted":tasks.len(),"elapsed_seconds":start.elapsed().as_secs_f64(),"model":"nufrost-supported-v1 frozen","gap_days":[0,10,20,30,40,50,60],"fixed_target_date":true,"source_policy":"max unique jointly valid dates, lexical tie; exact timestamps deduplicated; strict0<DN<10000","common_cohort":"target present and >=12 training dates with >=3 per flank at60days; exclusions saved","finished":true}))?)?;Ok(())
}
