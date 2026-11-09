include!("input.rs");
#[derive(Deserialize)]
struct FitRequest {
    days: Vec<f64>,
    values: Vec<f64>,
    query: f64,
    initial: Option<[f64; 7]>,
}
fn fit_json(r: &FitRequest) -> Result<Value> {
    let f = hapo_core::fit(&r.days, &r.values, r.initial)?;
    Ok(
        json!({"prediction":f.predict(r.query),"coefficients":f.coefficients,"slope":f.slope,"center":f.center,"lower":f.lower,"range":f.range,"objective":f.objective,"iterations":f.iterations,"converged":f.converged,"gradient_inf":f.gradient_inf,"stationarity_inf":f.stationarity_inf}),
    )
}
fn predict(i: &Input) -> Result<(Vec<f64>, Vec<bool>, Vec<usize>, Vec<f64>)> {
    let days: Vec<_> = i
        .days
        .iter()
        .map(|t| t + i.phase_offset + 719163.)
        .collect();
    let mut pred = vec![];
    let mut converged = vec![];
    let mut iterations = vec![];
    let mut stationary = vec![];
    for values in &i.obs {
        let f = hapo_core::fit(&days, values, None)?;
        let p = f.predict(i.query + i.phase_offset + 719163.);
        ensure!(p.is_finite(), "nonfinite prediction");
        pred.push(p);
        converged.push(f.converged);
        iterations.push(f.iterations);
        stationary.push(f.stationarity_inf);
    }
    Ok((pred, converged, iterations, stationary))
}
mod metrics;
use metrics::Metrics;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--fit") {
        let r: FitRequest = serde_json::from_reader(std::io::stdin())?;
        println!("{}", fit_json(&r)?);
        return Ok(());
    }
    let root = Path::new(args.get(1).context("data root")?);
    let out = Path::new(args.get(2).context("new output directory")?);
    ensure!(!out.exists(), "refuse existing output directory");
    let limit: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(99969);
    let cache = Cache::open(root)?;
    let mut all =
        std::collections::BTreeMap::<usize, std::collections::BTreeMap<String, Task>>::new();
    for row in csv::Reader::from_path(
        root.join("tests/paper_supported_rerun_20261010/paired_holdout_v2/predictions.csv"),
    )?
    .deserialize::<Task>()
    {
        let t = row?;
        ensure!(
            all.entry(t.task_id)
                .or_default()
                .insert(t.method.clone(), t)
                .is_none(),
            "duplicate task/method"
        );
    }
    let tasks: Vec<_> = all
        .values()
        .filter_map(|m| m.get("nufrost").cloned())
        .collect();
    ensure!(tasks.len() == 99969, "cohort count");
    // Deterministically span the full cohort for small checks; full run uses every task.
    let tasks: Vec<_> = if limit < tasks.len() {
        (0..limit)
            .map(|j| tasks[j * (tasks.len() - 1) / (limit - 1).max(1)].clone())
            .collect()
    } else {
        tasks
    };
    fs::create_dir_all(out)?;
    let mut seen = std::collections::BTreeSet::new();
    let mut oracle = vec![];
    for t in &tasks {
        if !seen.insert(t.site) {
            continue;
        }
        let i = cache.input(t)?;
        let ix: Vec<_> = i.obs[0]
            .iter()
            .enumerate()
            .filter_map(|(j, v)| v.is_finite().then_some(j))
            .collect();
        oracle.push(json!({"site":t.site,"task_id":t.task_id,"days":ix.iter().map(|&j|i.days[j]+i.phase_offset+719163.).collect::<Vec<_>>(),"values":ix.iter().map(|&j|i.obs[0][j]).collect::<Vec<_>>(),"query":i.query+i.phase_offset+719163.,"initial":[0.25,0.17,0.31,0.53,0.71,0.23,0.43]}));
    }
    fs::write(
        out.join("oracle-requests.json"),
        serde_json::to_vec(&oracle)?,
    )?;
    let first = cache.input(&tasks[0])?;
    let valid: Vec<_> = first.obs[0]
        .iter()
        .enumerate()
        .filter_map(|(j, y)| y.is_finite().then_some(j))
        .collect();
    let parity = json!({"days":valid.iter().map(|&j|first.days[j]+first.phase_offset+719163.).collect::<Vec<_>>(),"values":valid.iter().map(|&j|first.obs[0][j]).collect::<Vec<_>>(),"query":first.query+first.phase_offset+719163.,"initial":[0.25,0.17,0.31,0.53,0.71,0.23,0.43]});
    fs::write(
        out.join("real-parity-input.json"),
        serde_json::to_vec(&parity)?,
    )?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(10).build()?;
    let writer = Mutex::new(File::create(out.join("predictions.jsonl"))?);
    let done = AtomicUsize::new(0);
    let errors = AtomicUsize::new(0);
    let nc = AtomicUsize::new(0);
    let metrics = Mutex::new(std::collections::BTreeMap::<String, Metrics>::new());
    let site_metrics = Mutex::new(std::collections::BTreeMap::<
        usize,
        std::collections::BTreeMap<String, Metrics>,
    >::new());
    let elapsed = Instant::now();
    pool.install(||tasks.par_iter().try_for_each(|t|->Result<()>{let now=Instant::now();let result=cache.input(t).and_then(|i|predict(&i));let mut row=json!({"task_id":t.task_id,"sample_idx":t.sample_idx,"site":t.site,"target_epoch":t.target_epoch,"n_train":t.n_train,"truth":t.truth().iter().map(|x|x/10000.).collect::<Vec<_>>()});
 match result {Ok((p,c,it,st))=>{let count=c.iter().filter(|&&v|!v).count();nc.fetch_add(count,Ordering::Relaxed);row["prediction"]=json!(p.iter().map(|x|x/10000.).collect::<Vec<_>>());row["band_converged"]=json!(c);row["iterations"]=json!(it);row["stationarity_inf"]=json!(st);row["valid"]=json!(true);let mut m=metrics.lock().unwrap();let mut sm=site_metrics.lock().unwrap();m.entry("hapo".into()).or_default().add(&p,&t.truth());sm.entry(t.site).or_default().entry("hapo".into()).or_default().add(&p,&t.truth());for (name,b) in &all[&t.task_id] {ensure!(b.sample_idx==t.sample_idx&&b.target_index==t.target_index&&b.truth()==t.truth(),"baseline identity mismatch");let bp=b.old();if bp.iter().all(|v|v.is_finite()){m.entry(name.clone()).or_default().add(&bp,&t.truth());sm.entry(t.site).or_default().entry(name.clone()).or_default().add(&bp,&t.truth());}}},Err(e)=>{errors.fetch_add(1,Ordering::Relaxed);row["valid"]=json!(false);row["error"]=json!(e.to_string());}}
 row["seconds"]=json!(now.elapsed().as_secs_f64());writeln!(writer.lock().unwrap(),"{row}")?;let n=done.fetch_add(1,Ordering::Relaxed)+1;if n%1000==0||n==tasks.len(){let progress=json!({"completed":n,"total":tasks.len(),"errors":errors.load(Ordering::Relaxed),"nonconverged_bands":nc.load(Ordering::Relaxed),"elapsed_seconds":elapsed.elapsed().as_secs_f64(),"threads":10});fs::write(out.join("progress.json"),serde_json::to_vec_pretty(&progress)?)?;eprintln!("{progress}");}Ok(())}))?;
    let metrics: std::collections::BTreeMap<_, _> = metrics
        .into_inner()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.json()))
        .collect();
    let sites: std::collections::BTreeMap<_, _> = site_metrics
        .into_inner()
        .unwrap()
        .iter()
        .map(|(site, m)| {
            (
                *site,
                m.iter()
                    .map(|(k, v)| (k.clone(), v.json()))
                    .collect::<std::collections::BTreeMap<_, _>>(),
            )
        })
        .collect();
    let summary = json!({"complete":true,"attempted":done.load(Ordering::Relaxed),"errors":errors.load(Ordering::Relaxed),"nonconverged_bands":nc.load(Ordering::Relaxed),"elapsed_seconds":elapsed.elapsed().as_secs_f64(),"threads":10,"metrics":metrics,"sites":sites,"protocol":"same complete series and source selection; entire target vector hidden; existing predictions reused, only HAPO fitted; raw un-clipped predictions; nonconvergence retained and reported separately","solver":"analytic-gradient BFGS; Armijo line search; gtol=1e-5; maxiter=1000; alpha=1; fixed initial coefficients; annual harmonics 1,2,3"});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{summary}");
    Ok(())
}
