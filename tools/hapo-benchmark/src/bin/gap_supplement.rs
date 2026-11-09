include!("../input.rs");
#[derive(Deserialize)]
struct SavedMask {
    task_id: usize,
    gap: usize,
    removed_indices: String,
}
#[derive(Deserialize)]
struct GapRow {
    #[serde(flatten)]
    t: Task,
    gap: usize,
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let root = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    ensure!(!out.exists(), "new output required");
    fs::create_dir_all(out)?;
    let cache = Cache::open(root)?;
    let mut tasks = vec![];
    let mut counts = std::collections::BTreeMap::new();
    let mut masks = std::collections::BTreeMap::new();
    for k in 0..160 {
        for r in csv::Reader::from_path(
            root.join(format!("tests/gap16_fig10_20261010/chunk_{k:04}/masks.csv")),
        )?
        .deserialize::<SavedMask>()
        {
            let r = r?;
            let removed: Vec<usize> = r
                .removed_indices
                .split(';')
                .filter(|s| !s.is_empty())
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()?;
            masks.insert((r.task_id, r.gap), removed);
        }
        let p = root.join(format!(
            "tests/gap16_fig10_20261010/chunk_{k:04}/predictions.csv"
        ));
        for row in csv::Reader::from_path(p)?.deserialize::<GapRow>() {
            let r = row?;
            if r.t.method != "supported_v1" {
                continue;
            }
            counts.insert((r.t.task_id, r.gap), r.t.n_train);
            if r.gap == 0 {
                tasks.push(r.t);
            }
        }
    }
    ensure!(tasks.len() == 15431, "cohort count {}", tasks.len());
    rayon::ThreadPoolBuilder::new()
        .num_threads(10)
        .build_global()?;
    let writer = Mutex::new(File::create(out.join("predictions.jsonl"))?);
    let start = Instant::now();
    let done = AtomicUsize::new(0);
    tasks.par_iter().try_for_each(|t|->Result<()>{let i=cache.input(t)?;let days:Vec<_>=i.days.iter().map(|x|x+i.phase_offset+719163.).collect();for gap in [0,10,20,30,40,50,60] {let train:Vec<_>=(0..days.len()).filter(|&j|i.obs[0][j].is_finite()&&!masks[&(t.task_id,gap)].contains(&j)).collect();ensure!(train.len()==counts[&(t.task_id,gap)],"mask count");let td:Vec<_>=train.iter().map(|&j|days[j]).collect();let mut pred=vec![];let mut stat=vec![];for b in 0..6{let y:Vec<_>=train.iter().map(|&j|i.obs[b][j]).collect();let f=hapo_core::fit(&td,&y,None)?;pred.push(f.predict(i.query+i.phase_offset+719163.)/10000.);stat.push(f.stationarity_inf);}ensure!(pred.iter().all(|x|x.is_finite()),"finite");let row=json!({"task_id":t.task_id,"sample_idx":t.sample_idx,"site":t.site,"gap":gap,"truth":t.truth().iter().map(|x|x/10000.).collect::<Vec<_>>(),"prediction":pred,"stationarity_inf":stat});writeln!(writer.lock().unwrap(),"{row}")?;}let n=done.fetch_add(1,Ordering::Relaxed)+1;if n%1000==0{eprintln!("{n}/15431 {}s",start.elapsed().as_secs_f64());}Ok(())})?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"tasks":tasks.len(),"fits":tasks.len()*7,"threads":10,"seconds":start.elapsed().as_secs_f64(),"mask_matches_existing_counts":true}),
        )?,
    )?;
    Ok(())
}
