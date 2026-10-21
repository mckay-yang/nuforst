// nufrost-cli — command-line entrypoint for NUFROST, HANTS, and Zhu2015
// reconstruction algorithms.  Supports single-pixel fixture NPZ input and
// raster GeoTIFF input via gdal.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Arg, ArgAction, Command};
use memmap2::Mmap;
use ndarray::Array3;
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::Deserialize;

use gdal::{
    collapse_duplicate_timestamps, extract_raw_band_descriptions,
    full_scene::{
        self, build_ground_truth_output_path, build_scene_stack_output_path,
        choose_shared_target_timestamp, discover_sentinel_band_stacks, make_masked_time_series,
        mask_invalid_sentinel2, sorted_band_names, write_band_stack,
    },
    read_all_bands_window_offset, sample_cache, scene_cache, RasterMetadata, RasterReader,
};
use hants_core::{
    hants_fit, hants_pixel, hants_predict, hants_predict_curve, reconstruct_hants_geotiff,
    HantsConfig,
};
use nufrost_core::{
    nufrost_fit_pixel_vector, nufrost_pixel, nufrost_pixel_vector, nufrost_predict_vector,
    nufrost_predict_vector_curve, parse_iso8601_to_epoch_seconds, reconstruct_nufrost_geotiff,
    FrequencySelection, NufrostConfig,
};
use zhu2015_core::{
    fit_predict_pixel, fit_segment, reconstruct_zhu2015_geotiff, segment_predict, Zhu2015Config,
};

const SAMPLE_CACHE_EVAL_PROGRESS_INTERVAL: usize = 10_000;
const FULL_SCENE_VALID_RATIO_SUBSAMPLE_STEP: usize = 8;
const SENTINEL2_VALID_MIN: f64 = 0.0;
const SENTINEL2_VALID_MAX: f64 = 10000.0;

// ═══════════════════════════════════════════════════════════════════════════
//  CLI definition
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
struct Cli {
    algorithm: Algorithm,
}

#[derive(Debug)]
enum Algorithm {
    Nufrost(NufrostArgs),
    Hants(HantsArgs),
    Zhu2015(Zhu2015Args),
    FullScene(FullSceneArgs),
    BuildSceneCache(BuildSceneCacheArgs),
    BuildSampleCache(BuildSampleCacheArgs),
    EvalSampleCache(EvalSampleCacheArgs),
    SamplePixelCurve(SamplePixelCurveArgs),
    BatchFullScene(BatchFullSceneArgs),
    PixelBench(PixelBenchArgs),
}

/// Paths, methods, and validity rules shared by one scene run and a batch of scenes.
#[derive(Debug, Clone)]
struct SceneRun {
    source_name: String,
    output_root: PathBuf,
    data_root: PathBuf,
    methods: String,
    n_jobs: Option<usize>,
    window_size: Option<usize>,
    min_valid_ratio: f64,
    late_fraction: f64,
}

#[derive(Debug)]
struct FullSceneArgs {
    run: SceneRun,
    lon: f64,
    lat: f64,
    window_lon: Option<f64>,
    window_lat: Option<f64>,
    scene_cache: Option<PathBuf>,
    nufrost_config: Option<PathBuf>,
    frequency_selection: Option<String>,
}

#[derive(Debug)]
struct BuildSceneCacheArgs {
    source_name: String,
    lon: f64,
    lat: f64,
    data_root: PathBuf,
    cache_root: PathBuf,
    output: Option<PathBuf>,
}

#[derive(Debug)]
struct BuildSampleCacheArgs {
    source_name: String,
    scene_cache_root: PathBuf,
    output: PathBuf,
    n_samples: usize,
    min_joint_valid: usize,
    seed: u64,
    max_attempts_multiplier: usize,
    limit_scenes: Option<usize>,
}

#[derive(Debug)]
struct EvalSampleCacheArgs {
    method: String,
    cache_dir: PathBuf,
    n_eval: usize,
    seed: u64,
    min_joint_valid: usize,
    config: Option<PathBuf>,
    output_json: Option<PathBuf>,
    output_gap_csv: Option<PathBuf>,
    output_prediction_csv: Option<PathBuf>,
    gap_bins: usize,
    threads: Option<usize>,
}

#[derive(Debug)]
struct SamplePixelCurveArgs {
    cache_dir: PathBuf,
    sample_idx: usize,
    holdout_seed: Option<u64>,
    min_joint_valid: usize,
    nufrost_config: Option<PathBuf>,
    hants_config: Option<PathBuf>,
    zhu2015_config: Option<PathBuf>,
    output_csv: PathBuf,
}

#[derive(Debug)]
struct BatchFullSceneArgs {
    run: SceneRun,
    limit: Option<usize>,
    continue_on_error: bool,
}

#[derive(Debug)]
struct PixelBenchArgs {
    source_name: String,
    lon: f64,
    lat: f64,
    data_root: PathBuf,
    row: Option<usize>,
    col: Option<usize>,
    pixel_lon: Option<f64>,
    pixel_lat: Option<f64>,
    repeats: usize,
    min_valid_ratio: f64,
    late_fraction: f64,
}

#[derive(Debug)]
struct NufrostArgs {
    config: Option<PathBuf>,
    shared: SharedArgs,
}

#[derive(Debug)]
struct HantsArgs {
    config: Option<PathBuf>,
    shared: SharedArgs,
}

#[derive(Debug)]
struct Zhu2015Args {
    config: Option<PathBuf>,
    shared: SharedArgs,
}

#[derive(Debug)]
struct SharedArgs {
    data: Option<PathBuf>,
    input_geotiff: Option<PathBuf>,
    target_time: Option<f64>,
    output: Option<PathBuf>,
    #[allow(dead_code)]
    threads: usize,
}

impl Cli {
    fn parse() -> Self {
        Self::try_parse_from(std::env::args_os()).unwrap_or_else(|err| err.exit())
    }

    fn try_parse_from<I, T>(args: I) -> std::result::Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        let matches = Self::command().try_get_matches_from(args)?;
        let (name, sub) = matches
            .subcommand()
            .expect("subcommand_required ensures a subcommand is present");
        let algorithm = match name {
            "nufrost" => Algorithm::Nufrost(NufrostArgs {
                config: sub.get_one::<PathBuf>("config").cloned(),
                shared: parse_shared_args(sub),
            }),
            "hants" => Algorithm::Hants(HantsArgs {
                config: sub.get_one::<PathBuf>("config").cloned(),
                shared: parse_shared_args(sub),
            }),
            "zhu2015" => Algorithm::Zhu2015(Zhu2015Args {
                config: sub.get_one::<PathBuf>("config").cloned(),
                shared: parse_shared_args(sub),
            }),
            "full-scene" => Algorithm::FullScene(FullSceneArgs {
                run: scene_run_from_matches(sub),
                lon: *sub.get_one::<f64>("lon").unwrap(),
                lat: *sub.get_one::<f64>("lat").unwrap(),
                window_lon: sub.get_one::<f64>("window_lon").copied(),
                window_lat: sub.get_one::<f64>("window_lat").copied(),
                scene_cache: sub.get_one::<PathBuf>("scene_cache").cloned(),
                nufrost_config: sub.get_one::<PathBuf>("nufrost_config").cloned(),
                frequency_selection: sub.get_one::<String>("frequency_selection").cloned(),
            }),
            "build-scene-cache" => Algorithm::BuildSceneCache(BuildSceneCacheArgs {
                source_name: sub.get_one::<String>("source_name").cloned().unwrap(),
                lon: *sub.get_one::<f64>("lon").unwrap(),
                lat: *sub.get_one::<f64>("lat").unwrap(),
                data_root: sub.get_one::<PathBuf>("data_root").cloned().unwrap(),
                cache_root: sub.get_one::<PathBuf>("cache_root").cloned().unwrap(),
                output: sub.get_one::<PathBuf>("output").cloned(),
            }),
            "build-sample-cache" => Algorithm::BuildSampleCache(BuildSampleCacheArgs {
                source_name: sub.get_one::<String>("source_name").cloned().unwrap(),
                scene_cache_root: sub.get_one::<PathBuf>("scene_cache_root").cloned().unwrap(),
                output: sub.get_one::<PathBuf>("output").cloned().unwrap(),
                n_samples: *sub.get_one::<usize>("n_samples").unwrap(),
                min_joint_valid: *sub.get_one::<usize>("min_joint_valid").unwrap(),
                seed: *sub.get_one::<u64>("seed").unwrap(),
                max_attempts_multiplier: *sub.get_one::<usize>("max_attempts_multiplier").unwrap(),
                limit_scenes: sub.get_one::<usize>("limit_scenes").copied(),
            }),
            "eval-sample-cache" => Algorithm::EvalSampleCache(EvalSampleCacheArgs {
                method: sub.get_one::<String>("method").cloned().unwrap(),
                cache_dir: sub.get_one::<PathBuf>("cache_dir").cloned().unwrap(),
                n_eval: *sub.get_one::<usize>("n_eval").unwrap(),
                seed: *sub.get_one::<u64>("seed").unwrap(),
                min_joint_valid: *sub.get_one::<usize>("min_joint_valid").unwrap(),
                config: sub.get_one::<PathBuf>("config").cloned(),
                output_json: sub.get_one::<PathBuf>("output_json").cloned(),
                output_gap_csv: sub.get_one::<PathBuf>("output_gap_csv").cloned(),
                output_prediction_csv: sub.get_one::<PathBuf>("output_prediction_csv").cloned(),
                gap_bins: *sub.get_one::<usize>("gap_bins").unwrap(),
                threads: sub.get_one::<usize>("threads").copied(),
            }),
            "sample-pixel-curve" => Algorithm::SamplePixelCurve(SamplePixelCurveArgs {
                cache_dir: sub.get_one::<PathBuf>("cache_dir").cloned().unwrap(),
                sample_idx: *sub.get_one::<usize>("sample_idx").unwrap(),
                holdout_seed: sub.get_one::<u64>("holdout_seed").copied(),
                min_joint_valid: *sub.get_one::<usize>("min_joint_valid").unwrap(),
                nufrost_config: sub.get_one::<PathBuf>("nufrost_config").cloned(),
                hants_config: sub.get_one::<PathBuf>("hants_config").cloned(),
                zhu2015_config: sub.get_one::<PathBuf>("zhu2015_config").cloned(),
                output_csv: sub.get_one::<PathBuf>("output_csv").cloned().unwrap(),
            }),
            "batch-full-scene" => Algorithm::BatchFullScene(BatchFullSceneArgs {
                run: scene_run_from_matches(sub),
                limit: sub.get_one::<usize>("limit").copied(),
                continue_on_error: sub.get_flag("continue_on_error"),
            }),
            "pixel-bench" => Algorithm::PixelBench(PixelBenchArgs {
                source_name: sub.get_one::<String>("source_name").cloned().unwrap(),
                lon: *sub.get_one::<f64>("lon").unwrap(),
                lat: *sub.get_one::<f64>("lat").unwrap(),
                data_root: sub.get_one::<PathBuf>("data_root").cloned().unwrap(),
                row: sub.get_one::<usize>("row").copied(),
                col: sub.get_one::<usize>("col").copied(),
                pixel_lon: sub.get_one::<f64>("pixel_lon").copied(),
                pixel_lat: sub.get_one::<f64>("pixel_lat").copied(),
                repeats: *sub.get_one::<usize>("repeats").unwrap(),
                min_valid_ratio: *sub.get_one::<f64>("min_valid_ratio").unwrap(),
                late_fraction: *sub.get_one::<f64>("late_fraction").unwrap(),
            }),
            _ => unreachable!("clap accepted an unknown subcommand: {name}"),
        };
        Ok(Self { algorithm })
    }

    fn command() -> Command {
        Command::new("nufrost-cli")
            .version(env!("CARGO_PKG_VERSION"))
            .about("NUFROST / HANTS / Zhu2015 time-series reconstruction CLI")
            .after_help(
                "Examples:\n  \
                 nufrost-cli nufrost --data fixture.npz --target-time 372.7\n  \
                 nufrost-cli hants --config hants.json --data fixture.npz\n  \
                 nufrost-cli zhu2015 --data fixture.npz -t 372.7 -o pred.txt\n  \
                 nufrost-cli nufrost --input-geotiff input.tif --output pred.tif\n  \
                 nufrost-cli hants --input-geotiff input.tif -o pred.tif\n  \
                 nufrost-cli zhu2015 --input-geotiff input.tif -o pred.tif",
            )
            .subcommand_required(true)
            .arg_required_else_help(true)
            .subcommand(algorithm_command("nufrost", "Run NUFROST reconstruction."))
            .subcommand(algorithm_command("hants", "Run HANTS reconstruction."))
            .subcommand(algorithm_command("zhu2015", "Run Zhu2015 reconstruction."))
            .subcommand(full_scene_command())
            .subcommand(build_scene_cache_command())
            .subcommand(build_sample_cache_command())
            .subcommand(eval_sample_cache_command())
            .subcommand(sample_pixel_curve_command())
            .subcommand(batch_full_scene_command())
            .subcommand(pixel_bench_command())
    }
}

fn parse_shared_args(matches: &clap::ArgMatches) -> SharedArgs {
    SharedArgs {
        data: matches.get_one::<PathBuf>("data").cloned(),
        input_geotiff: matches.get_one::<PathBuf>("input_geotiff").cloned(),
        target_time: matches.get_one::<f64>("target_time").copied(),
        output: matches.get_one::<PathBuf>("output").cloned(),
        threads: matches.get_one::<usize>("threads").copied().unwrap_or(1),
    }
}

fn scene_run_from_matches(matches: &clap::ArgMatches) -> SceneRun {
    SceneRun {
        source_name: matches.get_one::<String>("source_name").cloned().unwrap(),
        output_root: matches.get_one::<PathBuf>("output_root").cloned().unwrap(),
        data_root: matches.get_one::<PathBuf>("data_root").cloned().unwrap(),
        methods: matches.get_one::<String>("methods").cloned().unwrap(),
        n_jobs: matches.get_one::<usize>("n_jobs").copied(),
        window_size: matches.get_one::<usize>("window_size").copied(),
        min_valid_ratio: *matches.get_one::<f64>("min_valid_ratio").unwrap(),
        late_fraction: *matches.get_one::<f64>("late_fraction").unwrap(),
    }
}

fn algorithm_command(name: &'static str, about: &'static str) -> Command {
    Command::new(name)
        .about(about)
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .args(shared_cli_args())
}

fn shared_cli_args() -> Vec<Arg> {
    vec![
        Arg::new("data")
            .short('d')
            .long("data")
            .value_name("PATH")
            .value_parser(clap::value_parser!(PathBuf)),
        Arg::new("input_geotiff")
            .long("input-geotiff")
            .value_name("PATH")
            .value_parser(clap::value_parser!(PathBuf)),
        Arg::new("target_time")
            .short('t')
            .long("target-time")
            .value_name("DAYS")
            .value_parser(clap::value_parser!(f64)),
        Arg::new("output")
            .short('o')
            .long("output")
            .value_name("PATH")
            .value_parser(clap::value_parser!(PathBuf)),
        Arg::new("threads")
            .long("threads")
            .value_name("N")
            .default_value("1")
            .value_parser(clap::value_parser!(usize)),
    ]
}

fn full_scene_command() -> Command {
    Command::new("full-scene")
        .about("Run full-scene reconstruction for one location.")
        .arg(
            Arg::new("source_name")
                .long("source-name")
                .required(true)
                .value_name("NAME")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("lon")
                .long("lon")
                .required(true)
                .value_name("LON")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("lat")
                .long("lat")
                .required(true)
                .value_name("LAT")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("output_root")
                .long("output-root")
                .value_name("PATH")
                .default_value("data/products/reconstruction")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("data_root")
                .long("data-root")
                .value_name("PATH")
                .default_value("data")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("methods")
                .long("methods")
                .value_name("LIST")
                .default_value("nufrost")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("n_jobs")
                .long("n-jobs")
                .value_name("N")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("window_size")
                .long("window-size")
                .value_name("PX")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("window_lon")
                .long("window-lon")
                .value_name("LON")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("window_lat")
                .long("window-lat")
                .value_name("LAT")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("min_valid_ratio")
                .long("min-valid-ratio")
                .value_name("RATIO")
                .default_value("0.9")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("late_fraction")
                .long("late-fraction")
                .value_name("RATIO")
                .default_value("0.25")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("scene_cache")
                .long("scene-cache")
                .value_name("DIR")
                .help("Read full-scene cubes from a prebuilt mmap scene cache directory.")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("nufrost_config")
                .long("nufrost-config")
                .value_name("PATH")
                .help("NUFROST config JSON used by full-scene reconstruction.")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("frequency_selection")
                .long("frequency-selection")
                .value_name("MODE")
                .action(ArgAction::Set),
        )
}

fn build_scene_cache_command() -> Command {
    Command::new("build-scene-cache")
        .about("Build a mmap-ready full-scene cache from source GeoTIFF stacks.")
        .arg(
            Arg::new("source_name")
                .long("source-name")
                .required(true)
                .value_name("NAME")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("lon")
                .long("lon")
                .required(true)
                .value_name("LON")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("lat")
                .long("lat")
                .required(true)
                .value_name("LAT")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("data_root")
                .long("data-root")
                .value_name("PATH")
                .default_value("data")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("cache_root")
                .long("cache-root")
                .value_name("PATH")
                .default_value("data/cache/sentinel-2/16-sites")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .value_name("DIR")
                .help("Explicit cache directory. Overrides --cache-root derived path.")
                .value_parser(clap::value_parser!(PathBuf)),
        )
}

fn build_sample_cache_command() -> Command {
    Command::new("build-sample-cache")
        .about("Build a mmap-ready random time-series sample cache from scene caches.")
        .arg(
            Arg::new("source_name")
                .long("source-name")
                .required(true)
                .value_name("NAME")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("scene_cache_root")
                .long("scene-cache-root")
                .value_name("PATH")
                .default_value("data/cache/sentinel-2/16-sites")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .value_name("DIR")
                .default_value("data/cache/sentinel-2/100k")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("n_samples")
                .long("n-samples")
                .value_name("N")
                .default_value("1000000")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("min_joint_valid")
                .long("min-joint-valid")
                .value_name("N")
                .default_value("12")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .value_name("U64")
                .default_value("20260608")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("max_attempts_multiplier")
                .long("max-attempts-multiplier")
                .value_name("N")
                .default_value("20")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("limit_scenes")
                .long("limit-scenes")
                .value_name("N")
                .help("Debug/smoke option: only use the first N aligned scene caches.")
                .value_parser(clap::value_parser!(usize)),
        )
}

fn eval_sample_cache_command() -> Command {
    Command::new("eval-sample-cache")
        .about("Evaluate holdout RMSE on a mmap-ready sample cache.")
        .arg(
            Arg::new("method")
                .long("method")
                .value_name("NAME")
                .default_value("nufrost")
                .help("Evaluation method: nufrost, hants, or zhu2015.")
                .value_parser(["nufrost", "hants", "zhu2015"]),
        )
        .arg(
            Arg::new("cache_dir")
                .long("cache-dir")
                .value_name("DIR")
                .default_value("data/cache/sentinel-2/100k")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("n_eval")
                .long("n-eval")
                .value_name("N")
                .default_value("10000")
                .help("Number of cached sample sequences to evaluate. Use the cache size for a full pass.")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .value_name("U64")
                .default_value("20260609")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("min_joint_valid")
                .long("min-joint-valid")
                .value_name("N")
                .default_value("12")
                .help("Minimum joint-valid dates required after one holdout is removed.")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output_json")
                .long("output-json")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output_gap_csv")
                .long("output-gap-csv")
                .value_name("PATH")
                .help("Optional CSV path for RMSE/MAE curves binned by gap indices.")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output_prediction_csv")
                .long("output-prediction-csv")
                .value_name("PATH")
                .help("Optional per-sample holdout prediction CSV with truth and prediction bands.")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("gap_bins")
                .long("gap-bins")
                .value_name("N")
                .default_value("20")
                .help("Number of value-range bins used for each gap-index RMSE curve.")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .value_name("N")
                .value_parser(clap::value_parser!(usize)),
        )
}

fn sample_pixel_curve_command() -> Command {
    Command::new("sample-pixel-curve")
        .about("Export a single-fit dense reconstruction curve for one cached multi-band sample.")
        .arg(
            Arg::new("cache_dir")
                .long("cache-dir")
                .value_name("DIR")
                .default_value("data/cache/sentinel-2/100k")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("sample_idx")
                .long("sample-idx")
                .value_name("N")
                .required(true)
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("holdout_seed")
                .long("holdout-seed")
                .value_name("U64")
                .help(
                    "Use the exact holdout seed from eval-sample-cache for a traceable case curve.",
                )
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("min_joint_valid")
                .long("min-joint-valid")
                .value_name("N")
                .default_value("12")
                .help(
                    "Minimum joint-valid dates required after one leave-one-out target is removed.",
                )
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("nufrost_config")
                .long("nufrost-config")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("hants_config")
                .long("hants-config")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("zhu2015_config")
                .long("zhu2015-config")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output_csv")
                .long("output-csv")
                .value_name("PATH")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf)),
        )
}

fn batch_full_scene_command() -> Command {
    Command::new("batch-full-scene")
        .about("Run full-scene reconstruction for every complete location.")
        .arg(
            Arg::new("source_name")
                .long("source-name")
                .required(true)
                .value_name("NAME")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("output_root")
                .long("output-root")
                .value_name("PATH")
                .default_value("data/products/reconstruction")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("data_root")
                .long("data-root")
                .value_name("PATH")
                .default_value("data")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("methods")
                .long("methods")
                .value_name("LIST")
                .default_value("nufrost")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("n_jobs")
                .long("n-jobs")
                .value_name("N")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("window_size")
                .long("window-size")
                .value_name("PX")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("min_valid_ratio")
                .long("min-valid-ratio")
                .value_name("RATIO")
                .default_value("0.9")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("late_fraction")
                .long("late-fraction")
                .value_name("RATIO")
                .default_value("0.25")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("limit")
                .long("limit")
                .value_name("N")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("continue_on_error")
                .long("continue-on-error")
                .action(ArgAction::SetTrue),
        )
}

fn pixel_bench_command() -> Command {
    Command::new("pixel-bench")
        .about("Benchmark one real multi-band pixel time series with vector NUFROST.")
        .arg(
            Arg::new("source_name")
                .long("source-name")
                .required(true)
                .value_name("NAME")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("lon")
                .long("lon")
                .required(true)
                .value_name("LON")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("lat")
                .long("lat")
                .required(true)
                .value_name("LAT")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("data_root")
                .long("data-root")
                .value_name("PATH")
                .default_value("data")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("row")
                .long("row")
                .value_name("ROW")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("col")
                .long("col")
                .value_name("COL")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("pixel_lon")
                .long("pixel-lon")
                .value_name("LON")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("pixel_lat")
                .long("pixel-lat")
                .value_name("LAT")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("repeats")
                .long("repeats")
                .value_name("N")
                .default_value("10")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("min_valid_ratio")
                .long("min-valid-ratio")
                .value_name("RATIO")
                .default_value("0.9")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("late_fraction")
                .long("late-fraction")
                .value_name("RATIO")
                .default_value("0.25")
                .value_parser(clap::value_parser!(f64)),
        )
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fixture loading (single-pixel NPZ mode)
// ═══════════════════════════════════════════════════════════════════════════

/// Raw data extracted from an NPZ fixture.
#[derive(Debug)]
struct FixtureData {
    timestamps_days: Vec<f64>,
    observations: Vec<f64>,
    target_time_day: f64,
}

/// Load a single-pixel NPZ fixture.
fn load_fixture_npz(path: &std::path::Path) -> Result<FixtureData> {
    use ndarray::Array1;

    let file =
        fs::File::open(path).with_context(|| format!("Cannot open fixture: {}", path.display()))?;
    let mut npz = ndarray_npy::NpzReader::new(file)
        .with_context(|| format!("Cannot parse NPZ: {}", path.display()))?;

    let timestamps_days: Array1<f64> = npz
        .by_name("timestamps_days.npy")
        .context("Key 'timestamps_days' not found in fixture NPZ")?;
    let observations: Array1<f64> = npz
        .by_name("observations.npy")
        .context("Key 'observations' not found in fixture NPZ")?;
    let target_time_day_arr: ndarray::Array0<f64> = npz
        .by_name("target_time_day.npy")
        .context("Key 'target_time_day' not found in fixture NPZ")?;

    if timestamps_days.len() != observations.len() {
        bail!(
            "timestamps_days (len={}) and observations (len={}) must have the same length",
            timestamps_days.len(),
            observations.len()
        );
    }

    Ok(FixtureData {
        timestamps_days: timestamps_days.to_vec(),
        observations: observations.to_vec(),
        target_time_day: target_time_day_arr[()],
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Config loading
// ═══════════════════════════════════════════════════════════════════════════

/// Default NUFROST config matching Python `config/nufrost.json`.
fn default_nufrost_config() -> NufrostConfig {
    let full_json = r#"{
        "nufrost":{"modes":64,"num_peaks":10,"power_cum":0.7,"ignore_dc_hz":1e-10,"frequency_selection":"all","preferred_periods_days":"365.25,182.625,91.3125,30.4375","preferred_top_k":4,"spectral_top_k":8,"spectral_merge_tol":0.15,"refine_peaks":true,"include_trend":true,"ridge_lam":2.0,"freq_weight":256.0,"huber_iters":3,"huber_delta":0.05,"min_obs":12,"outlier_sigma":2.5,"outlier_reject_iters":2,"outlier_reject_sigma":2.5,"outlier_reject_max_fraction":0.35,"lambda_step":0.05,"lambda_high":0.005,"low_freq_period_days":60.0,"step_dt_weighting":true,"max_outer_iter":5,"outer_tol":0.001,"joint_outlier":true},
        "hants":{"nof":3,"sf":"high","fet":500.0,"dod":5,"period":365.25,"valid_min":null,"valid_max":null},
        "zhu2015":{"lasso_alpha":0.1}
    }"#;
    parse_grouped_config_section(full_json.as_bytes(), "nufrost")
        .expect("hardcoded grouped default config must be valid")
}

/// Default HANTS config matching Python `config/hants.json`.
fn default_hants_config() -> HantsConfig {
    serde_json::from_str(
        r#"{"nof":3,"sf":"high","fet":500.0,"dod":5,"period":365.25,"valid_min":null,"valid_max":null}"#,
    )
    .expect("hardcoded default HANTS config must be valid")
}

/// Default Zhu2015 config matching Python `config/zhu2015.json`.
fn default_zhu2015_config() -> Zhu2015Config {
    serde_json::from_str(r#"{"lasso_alpha":0.1}"#)
        .expect("hardcoded default Zhu2015 config must be valid")
}

fn parse_grouped_config_section<T: DeserializeOwned>(
    bytes: &[u8],
    section: &str,
) -> serde_json::Result<T> {
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    serde_json::from_value(value[section].clone())
}

#[allow(deprecated)]
fn synthetic_geotiff_timestamps(n_bands: usize) -> (Vec<f64>, f64) {
    gdal::synthetic_timestamps_from_bands(n_bands)
}

fn load_nufrost_config(path: Option<&std::path::Path>) -> Result<NufrostConfig> {
    match path {
        Some(p) => {
            let bytes =
                fs::read(p).with_context(|| format!("Cannot read config: {}", p.display()))?;
            parse_grouped_config_section(&bytes, "nufrost")
                .or_else(|_| NufrostConfig::from_json(&bytes))
                .with_context(|| format!("Invalid NUFROST config: {}", p.display()))
        }
        None => Ok(default_nufrost_config()),
    }
}

fn load_hants_config(path: Option<&std::path::Path>) -> Result<HantsConfig> {
    match path {
        Some(p) => {
            let bytes =
                fs::read(p).with_context(|| format!("Cannot read config: {}", p.display()))?;
            parse_grouped_config_section(&bytes, "hants")
                .or_else(|_| HantsConfig::from_json(&bytes))
                .with_context(|| format!("Invalid HANTS config: {}", p.display()))
        }
        None => Ok(default_hants_config()),
    }
}

fn load_zhu2015_config(path: Option<&std::path::Path>) -> Result<Zhu2015Config> {
    match path {
        Some(p) => {
            let bytes =
                fs::read(p).with_context(|| format!("Cannot read config: {}", p.display()))?;
            parse_grouped_config_section(&bytes, "zhu2015")
                .or_else(|_| Zhu2015Config::from_json(&bytes))
                .with_context(|| format!("Invalid Zhu2015 config: {}", p.display()))
        }
        None => Ok(default_zhu2015_config()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Input mode detection
// ═══════════════════════════════════════════════════════════════════════════

enum InputMode {
    /// Single-pixel NPZ fixture.
    NpzFixture(FixtureData),
    /// Raster GeoTIFF with per-pixel reconstruction.
    GeoTiff {
        reader: RasterReader,
        output: PathBuf,
    },
}

fn detect_input_mode(shared: &SharedArgs) -> Result<InputMode> {
    match (&shared.data, &shared.input_geotiff) {
        (Some(npz_path), None) => {
            let fixture = load_fixture_npz(npz_path)?;
            Ok(InputMode::NpzFixture(fixture))
        }
        (None, Some(tif_path)) => {
            let reader = RasterReader::open(tif_path)
                .with_context(|| format!("Cannot open GeoTIFF: {}", tif_path.display()))?;
            if reader.band_count() == 0 {
                bail!("GeoTIFF has no bands");
            }
            let (rows, cols) = reader.shape();
            if rows == 0 || cols == 0 {
                bail!("GeoTIFF has zero spatial dimensions ({rows}r × {cols}c)");
            }
            let output = shared
                .output
                .clone()
                .ok_or_else(|| anyhow::anyhow!("--output <PATH> is required in GeoTIFF mode"))?;
            Ok(InputMode::GeoTiff { reader, output })
        }
        (Some(_), Some(_)) => {
            bail!("--data and --input-geotiff are mutually exclusive")
        }
        (None, None) => {
            bail!(
                "No input data provided.  Use --data <fixture.npz> for single-pixel mode \
                 or --input-geotiff <input.tif> for raster mode."
            )
        }
    }
}

/// Resolve target time: explicit CLI arg beats fixture-embedded value.
fn resolve_target_time(fixture: &FixtureData, cli_target: Option<f64>) -> f64 {
    cli_target.unwrap_or(fixture.target_time_day)
}

/// Write or print a single scalar result (NPZ mode).
fn output_result(value: f64, output_path: Option<&PathBuf>, label: &str) -> Result<()> {
    let text = if label.is_empty() {
        format!("{value}\n")
    } else {
        format!("{label}: {value}\n")
    };
    match output_path {
        Some(p) => {
            fs::write(p, &text).with_context(|| format!("Cannot write output: {}", p.display()))?;
        }
        None => {
            print!("{text}");
        }
    }
    Ok(())
}

// ── Metadata extraction from reader ──────────────────────────────────────

fn metadata_from_reader(reader: &RasterReader) -> RasterMetadata {
    RasterMetadata {
        geo_transform: reader
            .geo_transform()
            .unwrap_or([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]),
        crs_wkt: reader.crs_wkt(),
        nodata: reader.nodata(1),
    }
}

fn run_build_scene_cache(args: &BuildSceneCacheArgs) -> Result<()> {
    let cache_dir = scene_cache::build_scene_cache(
        &args.data_root,
        &args.cache_root,
        &args.source_name,
        args.lon,
        args.lat,
        args.output.as_deref(),
    )?;
    eprintln!("Scene cache written to {}", cache_dir.display());
    eprintln!("  data: {}", cache_dir.join("cube.f32.bin").display());
    eprintln!("  meta: {}", cache_dir.join("meta.json").display());
    Ok(())
}

fn run_build_sample_cache(args: &BuildSampleCacheArgs) -> Result<()> {
    let start = Instant::now();
    let meta = sample_cache::build_sample_cache(&sample_cache::SampleCacheBuildOptions {
        scene_cache_root: args.scene_cache_root.clone(),
        source_name: args.source_name.clone(),
        output_dir: args.output.clone(),
        n_samples: args.n_samples,
        min_joint_valid: args.min_joint_valid,
        seed: args.seed,
        max_attempts_multiplier: args.max_attempts_multiplier,
        limit_scenes: args.limit_scenes,
    })?;
    eprintln!("Sample cache written to {}", args.output.display());
    eprintln!(
        "  samples: {}",
        args.output.join(&meta.sample_file).display()
    );
    eprintln!("  mask: {}", args.output.join(&meta.mask_file).display());
    eprintln!(
        "  scene times: {}",
        args.output.join(&meta.scene_time_file).display()
    );
    eprintln!("  index: {}", args.output.join(&meta.index_file).display());
    eprintln!("  meta: {}", args.output.join("meta.json").display());
    eprintln!(
        "  shape: samples=[{}, {}, {}], scenes={}, elapsed={:.3}s",
        meta.n_samples,
        meta.max_times,
        meta.n_bands,
        meta.scenes.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
//  NPZ mode — single-pixel reconstruction
// ═══════════════════════════════════════════════════════════════════════════

fn run_nufrost_npz(args: &NufrostArgs, fixture: &FixtureData) -> Result<()> {
    let config = load_nufrost_config(args.config.as_deref())?;
    let target_t = resolve_target_time(fixture, args.shared.target_time);

    let (pred, n_freqs) = nufrost_pixel(
        &fixture.timestamps_days,
        &fixture.observations,
        target_t,
        &config,
    );

    if !pred.is_finite() {
        bail!("NUFROST prediction is non-finite (NaN or Inf)");
    }

    output_result(
        pred,
        args.shared.output.as_ref(),
        &format!("nufrost_prediction (n_freqs={n_freqs})"),
    )?;

    eprintln!("NUFROST completed: pred={pred:.6}, n_freqs={n_freqs}");
    Ok(())
}

fn run_hants_npz(args: &HantsArgs, fixture: &FixtureData) -> Result<()> {
    let config = load_hants_config(args.config.as_deref())?;
    let target_t = resolve_target_time(fixture, args.shared.target_time);

    let pred = hants_pixel(
        &fixture.timestamps_days,
        &fixture.observations,
        target_t,
        config.nof,
        &config.sf,
        config.valid_min,
        config.valid_max,
        config.fet,
        config.dod,
        config.period,
    );

    if !pred.is_finite() {
        bail!("HANTS prediction is non-finite (NaN or Inf)");
    }

    output_result(pred, args.shared.output.as_ref(), "hants_prediction")?;
    eprintln!("HANTS completed: pred={pred:.6}");
    Ok(())
}

fn run_zhu2015_npz(args: &Zhu2015Args, fixture: &FixtureData) -> Result<()> {
    let config = load_zhu2015_config(args.config.as_deref())?;
    let target_t = resolve_target_time(fixture, args.shared.target_time);

    let result = fit_predict_pixel(
        &fixture.timestamps_days,
        &fixture.observations,
        target_t,
        config.lasso_alpha,
    );

    if !result.prediction.is_finite() {
        bail!("Zhu2015 prediction is non-finite (NaN or Inf)");
    }

    output_result(
        result.prediction,
        args.shared.output.as_ref(),
        "zhu2015_prediction",
    )?;

    eprintln!("Zhu2015 completed: pred={:.6}", result.prediction);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
//  GeoTIFF mode — full raster reconstruction
// ═══════════════════════════════════════════════════════════════════════════

fn run_nufrost_geotiff(args: &NufrostArgs, reader: &RasterReader, output: &PathBuf) -> Result<()> {
    let config = load_nufrost_config(args.config.as_deref())?;
    let meta = metadata_from_reader(reader);
    let (timestamps_days, target_t) = synthetic_geotiff_timestamps(reader.band_count());
    let target_t = args.shared.target_time.unwrap_or(target_t);

    reconstruct_nufrost_geotiff(reader, &timestamps_days, target_t, &config, output, &meta)
        .with_context(|| format!("NUFROST GeoTIFF reconstruction failed"))?;

    eprintln!(
        "NUFROST GeoTIFF reconstruction written to {}",
        output.display()
    );
    Ok(())
}

fn run_hants_geotiff(args: &HantsArgs, reader: &RasterReader, output: &PathBuf) -> Result<()> {
    let config = load_hants_config(args.config.as_deref())?;
    let meta = metadata_from_reader(reader);
    let (timestamps_days, target_t) = synthetic_geotiff_timestamps(reader.band_count());
    let target_t = args.shared.target_time.unwrap_or(target_t);

    reconstruct_hants_geotiff(
        reader,
        &timestamps_days,
        target_t,
        config.nof,
        &config.sf,
        config.valid_min,
        config.valid_max,
        config.fet,
        config.dod,
        config.period,
        output,
        &meta,
    )
    .with_context(|| "HANTS GeoTIFF reconstruction failed")?;

    eprintln!(
        "HANTS GeoTIFF reconstruction written to {}",
        output.display()
    );
    Ok(())
}

fn run_zhu2015_geotiff(args: &Zhu2015Args, reader: &RasterReader, output: &PathBuf) -> Result<()> {
    let config = load_zhu2015_config(args.config.as_deref())?;
    let meta = metadata_from_reader(reader);
    let (timestamps_days, target_t) = synthetic_geotiff_timestamps(reader.band_count());
    let target_t = args.shared.target_time.unwrap_or(target_t);

    reconstruct_zhu2015_geotiff(
        reader,
        &timestamps_days,
        target_t,
        config.lasso_alpha,
        output,
        &meta,
    )
    .with_context(|| "Zhu2015 GeoTIFF reconstruction failed")?;

    eprintln!(
        "Zhu2015 GeoTIFF reconstruction written to {}",
        output.display()
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
//  Full-scene reconstruction handler
// ═══════════════════════════════════════════════════════════════════════════

fn full_scene_window_offset(
    reader: &RasterReader,
    window_size: usize,
    window_lon: Option<f64>,
    window_lat: Option<f64>,
) -> Result<(usize, usize)> {
    let (rows, cols) = reader.shape();
    let ws = window_size.min(rows).min(cols);
    if let (Some(lon), Some(lat)) = (window_lon, window_lat) {
        let gt = reader
            .geo_transform()
            .context("--window-lon/--window-lat require a georeferenced raster")?;
        let det = gt[1] * gt[5] - gt[2] * gt[4];
        if det.abs() <= 1e-15 {
            bail!("Cannot invert raster geotransform for --window-lon/--window-lat");
        }
        let dx = lon - gt[0];
        let dy = lat - gt[3];
        let col = (gt[5] * dx - gt[2] * dy) / det;
        let row = (-gt[4] * dx + gt[1] * dy) / det;
        let center_col = col.floor() as isize;
        let center_row = row.floor() as isize;
        let half = (ws / 2) as isize;
        let max_row_off = rows.saturating_sub(ws) as isize;
        let max_col_off = cols.saturating_sub(ws) as isize;
        let row_off = (center_row - half).clamp(0, max_row_off) as usize;
        let col_off = (center_col - half).clamp(0, max_col_off) as usize;
        Ok((row_off, col_off))
    } else {
        let row_off = if rows > ws { (rows - ws) / 2 } else { 0 };
        let col_off = if cols > ws { (cols - ws) / 2 } else { 0 };
        Ok((row_off, col_off))
    }
}

fn reconstruct_nufrost_vector_scene(
    ordered_bands: &[String],
    masked_cubes: &BTreeMap<String, Array3<f64>>,
    masked_ts: &BTreeMap<String, Vec<String>>,
    target_epoch: f64,
    config: &NufrostConfig,
) -> Result<BTreeMap<String, ndarray::Array2<f64>>> {
    let first_band = ordered_bands
        .first()
        .context("NUFROST vector reconstruction requires at least one band")?;
    let ref_ts = masked_ts
        .get(first_band)
        .with_context(|| format!("Timestamps missing for band {first_band}"))?;
    let first_cube = masked_cubes
        .get(first_band)
        .with_context(|| format!("Cube missing for band {first_band}"))?;
    let (_, rows, cols) = first_cube.dim();

    let ts_sets: BTreeMap<&str, BTreeSet<&str>> = ordered_bands
        .iter()
        .map(|band_name| {
            let ts = masked_ts
                .get(band_name)
                .with_context(|| format!("Timestamps missing for band {band_name}"))?;
            Ok((band_name.as_str(), ts.iter().map(String::as_str).collect()))
        })
        .collect::<Result<_>>()?;
    let common_ts: Vec<String> = ref_ts
        .iter()
        .filter(|ts| {
            ts_sets
                .values()
                .all(|band_ts| band_ts.contains(ts.as_str()))
        })
        .cloned()
        .collect();
    if common_ts.is_empty() {
        bail!("NUFROST vector reconstruction found no common masked timestamps across bands");
    }
    if common_ts.len() != ref_ts.len() {
        eprintln!(
            "NUFROST vector scene: using {} common masked timestamps out of {} in reference band {first_band}",
            common_ts.len(),
            ref_ts.len()
        );
    }
    let n_times = common_ts.len();

    let band_inputs: Vec<(&Array3<f64>, Vec<usize>)> = ordered_bands
        .iter()
        .map(|band_name| {
            let cube = masked_cubes
                .get(band_name)
                .with_context(|| format!("Cube missing for band {band_name}"))?;
            let ts = masked_ts
                .get(band_name)
                .with_context(|| format!("Timestamps missing for band {band_name}"))?;
            if cube.dim().1 != rows || cube.dim().2 != cols {
                bail!(
                    "NUFROST vector reconstruction requires aligned spatial shapes; band {band_name} has {:?}, expected rows={rows}, cols={cols}",
                    cube.dim()
                );
            }
            if cube.dim().0 != ts.len() {
                bail!(
                    "NUFROST vector reconstruction timestamp count mismatch for band {band_name}: {} timestamps for {:?}",
                    ts.len(),
                    cube.dim()
                );
            }
            let index_by_ts: BTreeMap<&str, usize> = ts
                .iter()
                .enumerate()
                .map(|(idx, value)| (value.as_str(), idx))
                .collect();
            let indices = common_ts
                .iter()
                .map(|value| {
                    index_by_ts
                        .get(value.as_str())
                        .copied()
                        .with_context(|| format!("Timestamp {value} missing for band {band_name}"))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((cube, indices))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut epochs: Vec<f64> = common_ts
        .iter()
        .filter_map(|s| parse_iso8601_to_epoch_seconds(s))
        .collect();
    epochs.push(target_epoch);
    epochs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ts_days: Vec<f64> = epochs.iter().map(|&e| (e - epochs[0]) / 86400.0).collect();
    let target_day = (target_epoch - epochs[0]) / 86400.0;
    let masked_days: Vec<f64> = ts_days
        .iter()
        .copied()
        .filter(|&d| (d - target_day).abs() > 1e-9)
        .collect();
    if masked_days.len() != n_times {
        bail!(
            "NUFROST vector reconstruction timestamp count mismatch: {} masked days for {} cube slices",
            masked_days.len(),
            n_times
        );
    }

    let n_bands = ordered_bands.len();
    let mut pred_cube = Array3::<f64>::from_elem((n_bands, rows, cols), f64::NAN);
    eprintln!("NUFROST vector scene: {rows} rows x {cols} cols x {n_bands} bands");
    let progress = AtomicUsize::new(0);
    let progress_step = (rows / 10).max(1);
    pred_cube
        .axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(r, mut row_out)| {
            let mut obs_by_band: Vec<Vec<f64>> =
                (0..n_bands).map(|_| Vec::with_capacity(n_times)).collect();
            for c in 0..cols {
                for obs in obs_by_band.iter_mut() {
                    obs.clear();
                }
                for (bi, (cube, time_indices)) in band_inputs.iter().enumerate() {
                    for &ti in time_indices {
                        obs_by_band[bi].push(cube[[ti, r, c]]);
                    }
                }
                let pred = nufrost_pixel_vector(&masked_days, &obs_by_band, target_day, config);
                for (bi, &val) in pred.iter().enumerate() {
                    row_out[[bi, c]] = if val.is_finite() { val } else { f64::NAN };
                }
            }
            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done == rows || done % progress_step == 0 {
                eprintln!("  NUFROST rows: {done}/{rows}");
            }
        });

    Ok(ordered_bands
        .iter()
        .enumerate()
        .map(|(bi, band_name)| {
            (
                band_name.clone(),
                pred_cube.index_axis(ndarray::Axis(0), bi).to_owned(),
            )
        })
        .collect())
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct CliSceneCacheMeta {
    version: u32,
    source_name: String,
    lon: f64,
    lat: f64,
    rows: usize,
    cols: usize,
    layout: String,
    dtype: String,
    bands: Vec<String>,
    band_meta: Vec<CliSceneCacheBandMeta>,
    total_values: usize,
    geo_transform: [f64; 6],
    crs_wkt: Option<String>,
    nodata: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct CliSceneCacheBandMeta {
    name: String,
    timestamps: Vec<String>,
    time_len: usize,
    offset_values: usize,
}

struct CliSceneCacheView {
    meta: CliSceneCacheMeta,
    mmap: Mmap,
    band_index: BTreeMap<String, usize>,
}

impl CliSceneCacheView {
    fn open(cache_dir: &Path) -> Result<Self> {
        let meta_path = cache_dir.join("meta.json");
        let bin_path = cache_dir.join("cube.f32.bin");
        let meta: CliSceneCacheMeta =
            serde_json::from_slice(&fs::read(&meta_path).with_context(|| {
                format!("Cannot read scene cache meta {}", meta_path.display())
            })?)?;
        if meta.version != 1 || meta.layout != "band_time_row_col" || meta.dtype != "f32_le" {
            bail!(
                "Unsupported scene cache metadata: version={}, layout={}, dtype={}",
                meta.version,
                meta.layout,
                meta.dtype
            );
        }
        let file = File::open(&bin_path)
            .with_context(|| format!("Cannot open scene cache data {}", bin_path.display()))?;
        let expected_bytes = meta
            .total_values
            .checked_mul(4)
            .context("Scene cache byte size overflow")?;
        let actual_bytes = file.metadata()?.len() as usize;
        if actual_bytes != expected_bytes {
            bail!(
                "Scene cache data size mismatch: expected {expected_bytes} bytes, got {actual_bytes}"
            );
        }
        let mmap = unsafe { Mmap::map(&file)? };
        let band_index = meta
            .band_meta
            .iter()
            .enumerate()
            .map(|(idx, band)| (band.name.clone(), idx))
            .collect();
        Ok(Self {
            meta,
            mmap,
            band_index,
        })
    }

    fn raster_meta(&self) -> RasterMetadata {
        RasterMetadata {
            geo_transform: self.meta.geo_transform,
            crs_wkt: self.meta.crs_wkt.clone(),
            nodata: self.meta.nodata,
        }
    }

    fn band(&self, band_name: &str) -> Result<(usize, &CliSceneCacheBandMeta)> {
        let idx = self
            .band_index
            .get(band_name)
            .copied()
            .with_context(|| format!("Band {band_name} missing from scene cache"))?;
        Ok((idx, &self.meta.band_meta[idx]))
    }

    fn value_at(&self, band_idx: usize, time_idx: usize, row: usize, col: usize) -> f64 {
        let band = &self.meta.band_meta[band_idx];
        let spatial = self.meta.rows * self.meta.cols;
        let value_idx = band.offset_values + time_idx * spatial + row * self.meta.cols + col;
        let byte_idx = value_idx * 4;
        if byte_idx + 4 > self.mmap.len() {
            return f64::NAN;
        }
        let raw = [
            self.mmap[byte_idx],
            self.mmap[byte_idx + 1],
            self.mmap[byte_idx + 2],
            self.mmap[byte_idx + 3],
        ];
        let value = f32::from_le_bytes(raw) as f64;
        if is_valid_sentinel2_value(value) {
            value
        } else {
            f64::NAN
        }
    }
}

#[derive(Debug, Clone)]
struct DedupBandPlan {
    band_name: String,
    band_idx: usize,
    timestamps: Vec<String>,
    groups: Vec<Vec<usize>>,
    target_idx: usize,
}

impl DedupBandPlan {
    fn value_at(&self, view: &CliSceneCacheView, dedup_idx: usize, row: usize, col: usize) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for &time_idx in &self.groups[dedup_idx] {
            let value = view.value_at(self.band_idx, time_idx, row, col);
            if value.is_finite() {
                sum += value;
                count += 1;
            }
        }
        if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        }
    }
}

fn is_valid_sentinel2_value(value: f64) -> bool {
    value.is_finite() && value > SENTINEL2_VALID_MIN && value < SENTINEL2_VALID_MAX
}

fn dedup_plan_for_band(
    view: &CliSceneCacheView,
    band_name: &str,
    target_time: &str,
) -> Result<DedupBandPlan> {
    let (band_idx, band) = view.band(band_name)?;
    let mut group_map: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (idx, ts) in band.timestamps.iter().enumerate() {
        group_map.entry(ts.as_str()).or_default().push(idx);
    }
    let timestamps: Vec<String> = group_map.keys().map(|ts| (*ts).to_string()).collect();
    let groups: Vec<Vec<usize>> = group_map.values().cloned().collect();
    let target_idx = timestamps
        .iter()
        .position(|ts| ts == target_time)
        .with_context(|| format!("Target timestamp {target_time} missing for band {band_name}"))?;
    Ok(DedupBandPlan {
        band_name: band_name.to_string(),
        band_idx,
        timestamps,
        groups,
        target_idx,
    })
}

fn choose_target_timestamp_from_cache(
    view: &CliSceneCacheView,
    min_valid_ratio: f64,
    late_fraction: f64,
) -> Result<String> {
    let mut shared: Option<BTreeSet<String>> = None;
    for band in &view.meta.band_meta {
        let current: BTreeSet<String> = band.timestamps.iter().cloned().collect();
        shared = match shared {
            None => Some(current),
            Some(existing) => Some(existing.intersection(&current).cloned().collect()),
        };
    }
    let candidates: Vec<String> = shared.unwrap_or_default().into_iter().collect();
    if candidates.is_empty() {
        bail!("No shared timestamps exist across selected bands.");
    }

    let sampled_rows: Vec<usize> = (0..view.meta.rows)
        .step_by(FULL_SCENE_VALID_RATIO_SUBSAMPLE_STEP)
        .collect();
    let sampled_cols: Vec<usize> = (0..view.meta.cols)
        .step_by(FULL_SCENE_VALID_RATIO_SUBSAMPLE_STEP)
        .collect();
    let total = sampled_rows.len() * sampled_cols.len();
    let mut completeness_by_band: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    for (band_idx, band) in view.meta.band_meta.iter().enumerate() {
        let mut index_by_ts = BTreeMap::new();
        for (idx, ts) in band.timestamps.iter().enumerate() {
            index_by_ts.insert(ts.as_str(), idx);
        }
        let mut scores = BTreeMap::new();
        for candidate in &candidates {
            if let Some(&time_idx) = index_by_ts.get(candidate.as_str()) {
                let mut valid = 0usize;
                for &row in &sampled_rows {
                    for &col in &sampled_cols {
                        if view.value_at(band_idx, time_idx, row, col).is_finite() {
                            valid += 1;
                        }
                    }
                }
                scores.insert(
                    candidate.clone(),
                    if total == 0 {
                        0.0
                    } else {
                        valid as f64 / total as f64
                    },
                );
            }
        }
        completeness_by_band.insert(band.name.clone(), scores);
    }

    full_scene::select_shared_target_timestamp(
        &candidates,
        &completeness_by_band,
        min_valid_ratio,
        late_fraction,
    )
}

fn time_axis_days(masked_timestamps: &[String], target_epoch: f64) -> Result<(Vec<f64>, f64)> {
    let mut epochs: Vec<f64> = masked_timestamps
        .iter()
        .map(|s| {
            parse_iso8601_to_epoch_seconds(s)
                .with_context(|| format!("Failed to parse timestamp {s}"))
        })
        .collect::<Result<_>>()?;
    epochs.push(target_epoch);
    epochs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let origin = epochs[0];
    let target_day = (target_epoch - origin) / 86400.0;
    let masked_days: Vec<f64> = epochs
        .iter()
        .copied()
        .filter(|&epoch| (epoch - target_epoch).abs() > 1e-6)
        .map(|epoch| (epoch - origin) / 86400.0)
        .collect();
    if masked_days.len() != masked_timestamps.len() {
        bail!(
            "Timestamp count mismatch: {} masked days for {} timestamps",
            masked_days.len(),
            masked_timestamps.len()
        );
    }
    Ok((masked_days, target_day))
}

fn run_full_scene_from_cache_streaming(args: &FullSceneArgs, cache_dir: &Path) -> Result<()> {
    let methods = validate_full_scene_methods(&args.run.methods)?;
    let load_start = Instant::now();
    eprintln!("Opening scene cache for streaming: {}", cache_dir.display());
    let view = Arc::new(CliSceneCacheView::open(cache_dir)?);
    let scene_load_seconds = load_start.elapsed().as_secs_f64();
    eprintln!("Scene cache opened in {scene_load_seconds:.3}s");

    let ordered_bands = view.meta.bands.clone();
    let meta = view.raster_meta();
    let rows = view.meta.rows;
    let cols = view.meta.cols;

    let target_time_str = if let Some(target) = scene_cache::load_cached_target_timestamp(
        cache_dir,
        args.run.min_valid_ratio,
        args.run.late_fraction,
    )? {
        eprintln!("Using cached shared target timestamp: {target}");
        target
    } else {
        eprintln!("Selecting shared target timestamp from mmap cache...");
        let target =
            choose_target_timestamp_from_cache(&view, args.run.min_valid_ratio, args.run.late_fraction)?;
        scene_cache::store_cached_target_timestamp(
            cache_dir,
            args.run.min_valid_ratio,
            args.run.late_fraction,
            &target,
        )?;
        target
    };
    eprintln!("Selected target timestamp: {target_time_str}");
    let target_epoch = parse_iso8601_to_epoch_seconds(&target_time_str)
        .context("Failed to parse target timestamp")?;

    let band_plans: Vec<DedupBandPlan> = ordered_bands
        .iter()
        .map(|band_name| {
            let raw_len = view.band(band_name)?.1.timestamps.len();
            let plan = dedup_plan_for_band(&view, band_name, &target_time_str)?;
            eprintln!(
                "Band {band_name}: collapsed {} timestamp slices to {}",
                raw_len,
                plan.timestamps.len()
            );
            Ok(plan)
        })
        .collect::<Result<_>>()?;

    let mut ground_truths: BTreeMap<String, ndarray::Array2<f64>> = BTreeMap::new();
    for plan in &band_plans {
        let mut gt = ndarray::Array2::<f64>::from_elem((rows, cols), f64::NAN);
        gt.axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(row, mut row_out)| {
                for col in 0..cols {
                    row_out[col] = plan.value_at(&view, plan.target_idx, row, col);
                }
            });
        ground_truths.insert(plan.band_name.clone(), gt);
    }

    let nufrost_config_path = args
        .nufrost_config
        .as_deref()
        .unwrap_or_else(|| std::path::Path::new("config/nufrost.json"));
    let mut nufrost_conf =
        load_nufrost_config(Some(nufrost_config_path)).unwrap_or_else(|_| default_nufrost_config());
    if let Some(mode) = &args.frequency_selection {
        nufrost_conf.spectral.selection = FrequencySelection::from_name(mode)
            .with_context(|| format!("unknown frequency selection mode: {mode}"))?;
    }
    let hants_conf = default_hants_config();
    let zhu2015_conf = default_zhu2015_config();

    let source_name = &args.run.source_name;
    let output_root = &args.run.output_root;
    let loc_token = full_scene::location_token(args.lon, args.lat);

    let mut prediction_outputs = serde_json::Map::new();
    let mut metrics_by_method = serde_json::Map::new();

    for method in &methods {
        let method_str = method.as_str();
        eprintln!("Reconstructing with {method_str} using mmap streaming...");
        let mut predictions: BTreeMap<String, ndarray::Array2<f64>> = ordered_bands
            .iter()
            .map(|band| {
                (
                    band.clone(),
                    ndarray::Array2::<f64>::from_elem((rows, cols), f64::NAN),
                )
            })
            .collect();

        match method_str {
            "nufrost" => {
                let ts_sets: Vec<BTreeSet<&str>> = band_plans
                    .iter()
                    .map(|plan| {
                        plan.timestamps
                            .iter()
                            .enumerate()
                            .filter(|(idx, _)| *idx != plan.target_idx)
                            .map(|(_, ts)| ts.as_str())
                            .collect()
                    })
                    .collect();
                let common_ts: Vec<String> = band_plans[0]
                    .timestamps
                    .iter()
                    .enumerate()
                    .filter(|(idx, ts)| {
                        *idx != band_plans[0].target_idx
                            && ts_sets.iter().all(|set| set.contains(ts.as_str()))
                    })
                    .map(|(_, ts)| ts.clone())
                    .collect();
                if common_ts.is_empty() {
                    bail!("NUFROST vector reconstruction found no common masked timestamps across bands");
                }
                let common_indices: Vec<Vec<usize>> = band_plans
                    .iter()
                    .map(|plan| {
                        let index_by_ts: BTreeMap<&str, usize> = plan
                            .timestamps
                            .iter()
                            .enumerate()
                            .map(|(idx, ts)| (ts.as_str(), idx))
                            .collect();
                        common_ts
                            .iter()
                            .map(|ts| {
                                index_by_ts.get(ts.as_str()).copied().with_context(|| {
                                    format!("Timestamp {ts} missing for band {}", plan.band_name)
                                })
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<_>>()?;
                let (masked_days, target_day) = time_axis_days(&common_ts, target_epoch)?;
                eprintln!(
                    "NUFROST vector scene: {rows} rows x {cols} cols x {} bands",
                    ordered_bands.len()
                );
                let progress = AtomicUsize::new(0);
                let progress_step = (rows / 10).max(1);
                let mut pred_cube =
                    Array3::<f64>::from_elem((ordered_bands.len(), rows, cols), f64::NAN);
                pred_cube
                    .axis_iter_mut(ndarray::Axis(1))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(row, mut row_out)| {
                        let mut obs_by_band: Vec<Vec<f64>> = band_plans
                            .iter()
                            .map(|_| Vec::with_capacity(common_ts.len()))
                            .collect();
                        for col in 0..cols {
                            for obs in obs_by_band.iter_mut() {
                                obs.clear();
                            }
                            for (bi, plan) in band_plans.iter().enumerate() {
                                for &dedup_idx in &common_indices[bi] {
                                    obs_by_band[bi].push(plan.value_at(&view, dedup_idx, row, col));
                                }
                            }
                            let pred = nufrost_pixel_vector(
                                &masked_days,
                                &obs_by_band,
                                target_day,
                                &nufrost_conf,
                            );
                            for (bi, &value) in pred.iter().enumerate() {
                                row_out[[bi, col]] =
                                    if value.is_finite() { value } else { f64::NAN };
                            }
                        }
                        let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                        if done == rows || done % progress_step == 0 {
                            eprintln!("  NUFROST rows: {done}/{rows}");
                        }
                    });
                for (bi, band_name) in ordered_bands.iter().enumerate() {
                    predictions.insert(
                        band_name.clone(),
                        pred_cube.index_axis(ndarray::Axis(0), bi).to_owned(),
                    );
                }
            }
            "hants" | "zhu2015" => {
                for (band_pos, plan) in band_plans.iter().enumerate() {
                    let masked_timestamps: Vec<String> = plan
                        .timestamps
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| *idx != plan.target_idx)
                        .map(|(_, ts)| ts.clone())
                        .collect();
                    let masked_indices: Vec<usize> = (0..plan.timestamps.len())
                        .filter(|idx| *idx != plan.target_idx)
                        .collect();
                    let (masked_days, target_day) =
                        time_axis_days(&masked_timestamps, target_epoch)?;
                    let n_times = masked_indices.len();
                    let pred = predictions
                        .get_mut(&plan.band_name)
                        .with_context(|| format!("Prediction missing for {}", plan.band_name))?;
                    eprintln!(
                        "  {method_str}/{}: {rows} rows x {cols} cols, {n_times} time slices",
                        plan.band_name
                    );
                    let progress = AtomicUsize::new(0);
                    let progress_step = (rows / 10).max(1);
                    pred.axis_iter_mut(ndarray::Axis(0))
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(row, mut row_out)| {
                            let mut ts_buf = Vec::with_capacity(n_times);
                            let mut obs_buf = Vec::with_capacity(n_times);
                            for col in 0..cols {
                                ts_buf.clear();
                                obs_buf.clear();
                                for (time_pos, &dedup_idx) in masked_indices.iter().enumerate() {
                                    let value = plan.value_at(&view, dedup_idx, row, col);
                                    if value.is_finite() {
                                        ts_buf.push(masked_days[time_pos]);
                                        obs_buf.push(value);
                                    }
                                }
                                if !ts_buf.is_empty() {
                                    row_out[col] = match method_str {
                                        "hants" => hants_pixel(
                                            &ts_buf,
                                            &obs_buf,
                                            target_day,
                                            hants_conf.nof,
                                            &hants_conf.sf,
                                            hants_conf.valid_min,
                                            hants_conf.valid_max,
                                            hants_conf.fet,
                                            hants_conf.dod,
                                            hants_conf.period,
                                        ),
                                        "zhu2015" => {
                                            let result = fit_predict_pixel(
                                                &ts_buf,
                                                &obs_buf,
                                                target_day,
                                                zhu2015_conf.lasso_alpha,
                                            );
                                            result.prediction
                                        }
                                        _ => f64::NAN,
                                    };
                                }
                            }
                            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                            if done == rows || done % progress_step == 0 {
                                eprintln!(
                                    "    {method_str}/{} rows: {done}/{rows}",
                                    ordered_bands[band_pos]
                                );
                            }
                        });
                }
            }
            _ => bail!("Unknown full-scene method {method_str}"),
        }

        let output_path = build_scene_stack_output_path(
            output_root,
            method_str,
            source_name,
            args.lon,
            args.lat,
            &target_time_str,
            "prediction",
        );
        write_band_stack(&output_path, &predictions, &ordered_bands, &meta)?;
        eprintln!("  Wrote {}", output_path.display());

        let metrics = prediction_metrics_json(&predictions, &ground_truths, &ordered_bands)?;
        if let Some(rmse) = metrics.get("overall_rmse").and_then(|v| v.as_f64()) {
            eprintln!("  Overall RMSE: {rmse:.6}");
        }
        prediction_outputs.insert(
            method.clone(),
            serde_json::Value::String(output_path.display().to_string()),
        );
        metrics_by_method.insert(method.clone(), metrics);
    }

    let gt_path = build_ground_truth_output_path(
        output_root,
        source_name,
        args.lon,
        args.lat,
        &target_time_str,
    );
    write_band_stack(&gt_path, &ground_truths, &ordered_bands, &meta)?;
    eprintln!("Ground truth written to {}", gt_path.display());

    write_full_scene_summary(
        args,
        source_name,
        &loc_token,
        &ordered_bands,
        &target_time_str,
        target_epoch,
        scene_load_seconds,
        Some(cache_dir),
        prediction_outputs,
        metrics_by_method,
        &gt_path,
    )?;
    eprintln!("Full-scene reconstruction complete.");
    Ok(())
}

fn validate_full_scene_methods(methods: &str) -> Result<Vec<String>> {
    let parsed: Vec<String> = methods
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if parsed.is_empty() {
        bail!("--methods must contain at least one method");
    }
    for method in &parsed {
        match method.as_str() {
            "nufrost" | "hants" | "zhu2015" => {}
            _ => bail!(
                "Unknown full-scene method '{method}'. Expected one of: nufrost,hants,zhu2015"
            ),
        }
    }
    Ok(parsed)
}

fn source_data_dir(data_root: &Path, source_name: &str) -> PathBuf {
    let preferred = data_root.join("raw").join(source_name).join("16-sites");
    if preferred.is_dir() {
        preferred
    } else {
        data_root.join(source_name)
    }
}

fn default_scene_cache_root(data_root: &Path, source_name: &str) -> PathBuf {
    data_root.join("cache").join(source_name).join("16-sites")
}

fn default_scene_vrt_dir(data_root: &Path, source_name: &str) -> PathBuf {
    default_scene_cache_root(data_root, source_name).join("vrt")
}

fn resolve_full_scene_band_stacks(
    data_root: &Path,
    source_name: &str,
    lon: f64,
    lat: f64,
) -> Result<BTreeMap<String, Vec<PathBuf>>> {
    let source_dir = source_data_dir(data_root, source_name);
    let mut band_stacks = discover_sentinel_band_stacks(&source_dir, lon, lat)?;
    if band_stacks.is_empty() {
        bail!(
            "No band stacks found for lon={}, lat={} in {}",
            lon,
            lat,
            source_dir.display()
        );
    }

    let vrt_dir = default_scene_vrt_dir(data_root, source_name);
    let loc_token6 = full_scene::location_output_token(lon, lat);
    for (band_name, paths) in band_stacks.iter_mut() {
        if paths.len() <= 1 {
            continue;
        }

        let vrt_path = vrt_dir.join(format!("sentinel_{band_name}_{loc_token6}.vrt"));
        if vrt_path.exists() {
            eprintln!(
                "Band {band_name}: using cached mosaic VRT {} for {} source chunks",
                vrt_path.display(),
                paths.len()
            );
            *paths = vec![vrt_path];
        } else {
            bail!(
                "Band {band_name} has {} source chunks but no cached mosaic VRT at {}. \
                 Build or restore the VRT before full-scene reconstruction.",
                paths.len(),
                vrt_path.display()
            );
        }
    }

    Ok(band_stacks)
}

fn prediction_metrics_json(
    predictions: &BTreeMap<String, ndarray::Array2<f64>>,
    ground_truths: &BTreeMap<String, ndarray::Array2<f64>>,
    ordered_bands: &[String],
) -> Result<serde_json::Value> {
    let mut bands = serde_json::Map::new();
    let mut all_n = 0usize;
    let mut all_sum_err = 0.0f64;
    let mut all_sum_abs = 0.0f64;
    let mut all_sum_sq = 0.0f64;

    for band in ordered_bands {
        let pred = predictions
            .get(band)
            .with_context(|| format!("Prediction missing for band {band}"))?;
        let truth = ground_truths
            .get(band)
            .with_context(|| format!("Ground truth missing for band {band}"))?;
        if pred.dim() != truth.dim() {
            bail!(
                "Prediction/ground truth shape mismatch for band {band}: {:?} vs {:?}",
                pred.dim(),
                truth.dim()
            );
        }

        let mut n = 0usize;
        let mut sum_err = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        for (&p, &t) in pred.iter().zip(truth.iter()) {
            if p.is_finite() && t.is_finite() {
                let err = p - t;
                n += 1;
                sum_err += err;
                sum_abs += err.abs();
                sum_sq += err * err;
            }
        }

        all_n += n;
        all_sum_err += sum_err;
        all_sum_abs += sum_abs;
        all_sum_sq += sum_sq;

        let (bias, mae, rmse) = if n == 0 {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            (
                sum_err / n as f64,
                sum_abs / n as f64,
                (sum_sq / n as f64).sqrt(),
            )
        };
        bands.insert(
            band.clone(),
            serde_json::json!({
                "n": n,
                "bias": bias,
                "mae": mae,
                "rmse": rmse,
            }),
        );
    }

    let (overall_bias, overall_mae, overall_rmse) = if all_n == 0 {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        (
            all_sum_err / all_n as f64,
            all_sum_abs / all_n as f64,
            (all_sum_sq / all_n as f64).sqrt(),
        )
    };

    Ok(serde_json::json!({
        "n": all_n,
        "overall_bias": overall_bias,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "bands": bands,
    }))
}

#[allow(clippy::too_many_arguments)]
fn write_full_scene_summary(
    args: &FullSceneArgs,
    source_name: &str,
    loc_token: &str,
    ordered_bands: &[String],
    target_time_str: &str,
    target_epoch: f64,
    scene_load_seconds: f64,
    cache_dir: Option<&Path>,
    prediction_outputs: serde_json::Map<String, serde_json::Value>,
    metrics_by_method: serde_json::Map<String, serde_json::Value>,
    gt_path: &Path,
) -> Result<()> {
    let summary_dir = args.run.output_root.join("run_summaries");
    fs::create_dir_all(&summary_dir)?;
    let safe_time = target_time_str.replace(':', "-");
    let loc_token6 = full_scene::location_output_token(args.lon, args.lat);
    let summary_path = summary_dir.join(format!(
        "reconstruction_summary_{source_name}_{loc_token6}_{safe_time}.json"
    ));

    let mut merged_prediction_outputs = if let Ok(bytes) = fs::read(&summary_path) {
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| {
                v.get("prediction_outputs")
                    .and_then(|m| m.as_object())
                    .cloned()
            })
            .unwrap_or_default()
    } else {
        serde_json::Map::new()
    };
    for (method, output) in prediction_outputs {
        merged_prediction_outputs.insert(method, output);
    }

    let mut merged_metrics = if let Ok(bytes) = fs::read(&summary_path) {
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| v.get("metrics").and_then(|m| m.as_object()).cloned())
            .unwrap_or_default()
    } else {
        serde_json::Map::new()
    };
    for (method, metric) in metrics_by_method {
        merged_metrics.insert(method, metric);
    }

    let updated_methods = validate_full_scene_methods(&args.run.methods)?;
    let mut merged_methods: Vec<String> = merged_metrics.keys().cloned().collect();
    for method in &updated_methods {
        if !merged_methods.iter().any(|m| m == method) {
            merged_methods.push(method.clone());
        }
    }
    merged_methods.sort();

    let now_iso = chrono::Utc::now().to_rfc3339();
    let summary: serde_json::Value = serde_json::json!({
        "source_name": source_name,
        "lon": args.lon,
        "lat": args.lat,
        "location_token": loc_token,
        "target_time": target_time_str,
        "target_epoch": target_epoch,
        "methods_run": merged_methods,
        "methods_updated": updated_methods,
        "ordered_bands": ordered_bands,
        "min_valid_ratio": args.run.min_valid_ratio,
        "late_fraction": args.run.late_fraction,
        "window_size": args.run.window_size,
        "window_lon": args.window_lon,
        "window_lat": args.window_lat,
        "scene_cache": cache_dir.map(|p| p.display().to_string()),
        "scene_load_seconds": scene_load_seconds,
        "prediction_outputs": merged_prediction_outputs,
        "ground_truth_output": gt_path.display().to_string(),
        "metrics": merged_metrics,
        "generated_at": now_iso,
    });
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    eprintln!("Summary written to {}", summary_path.display());
    Ok(())
}

fn parse_sentinel_location_from_filename(name: &str) -> Option<(String, String, String)> {
    let rest = name.strip_prefix("COPERNICUS_S2_HARMONIZED_")?;
    let (band, rest) = rest.split_once("_lon")?;
    let (lon, lat_with_suffix) = rest.split_once("_lat")?;
    let lat_stem = lat_with_suffix.strip_suffix(".tif")?;
    let lat = strip_earth_engine_chunk_suffix(lat_stem);
    Some((band.to_string(), lon.to_string(), lat.to_string()))
}

fn strip_earth_engine_chunk_suffix(value: &str) -> &str {
    let bytes = value.as_bytes();
    for idx in 1..bytes.len() {
        if bytes[idx] != b'-' {
            continue;
        }
        let suffix = &value[idx + 1..];
        let Some((row, col)) = suffix.split_once('-') else {
            continue;
        };
        if row.len() == 10
            && col.len() == 10
            && row.bytes().all(|b| b.is_ascii_digit())
            && col.bytes().all(|b| b.is_ascii_digit())
        {
            return &value[..idx];
        }
    }
    value
}

fn discover_complete_sentinel_locations(
    data_root: &Path,
    source_name: &str,
) -> Result<Vec<(f64, f64)>> {
    let source_dir = source_data_dir(data_root, source_name);
    let entries = fs::read_dir(&source_dir)
        .with_context(|| format!("cannot read data directory: {}", source_dir.display()))?;
    let required: BTreeSet<String> = ["B2", "B3", "B4", "B8", "B11", "B12"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let mut by_location: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();

    for entry in entries {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some((band, lon, lat)) = parse_sentinel_location_from_filename(name) else {
            continue;
        };
        by_location.entry((lon, lat)).or_default().insert(band);
    }

    let mut locations = Vec::new();
    for ((lon, lat), bands) in by_location {
        if required.is_subset(&bands) {
            locations.push((
                lon.parse::<f64>()
                    .with_context(|| format!("invalid lon in source filename: {lon}"))?,
                lat.parse::<f64>()
                    .with_context(|| format!("invalid lat in source filename: {lat}"))?,
            ));
        }
    }
    locations.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(locations)
}

fn finite_median(values: &[f64]) -> f64 {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    if n % 2 == 1 {
        finite[n / 2]
    } else {
        0.5 * (finite[n / 2 - 1] + finite[n / 2])
    }
}

fn robust_standardize(values: &[f64]) -> Vec<f64> {
    let center = finite_median(values);
    let abs_dev: Vec<f64> = values
        .iter()
        .map(|&v| {
            if v.is_finite() {
                (v - center).abs()
            } else {
                f64::NAN
            }
        })
        .collect();
    let mut scale = 1.4826 * finite_median(&abs_dev);
    if !scale.is_finite() || scale <= 1e-6 {
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        let mean = finite.iter().copied().sum::<f64>() / finite.len().max(1) as f64;
        let var = finite
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>()
            / finite.len().max(1) as f64;
        scale = var.sqrt();
    }
    let center = if center.is_finite() {
        center
    } else {
        values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .sum::<f64>()
            / values.iter().filter(|v| v.is_finite()).count().max(1) as f64
    };
    let scale = scale.max(1e-6);
    values.iter().map(|&v| (v - center) / scale).collect()
}

fn run_pixel_bench(args: &PixelBenchArgs) -> Result<()> {
    if args.repeats == 0 {
        bail!("--repeats must be > 0");
    }
    if (args.row.is_some() || args.col.is_some()) && (args.row.is_none() || args.col.is_none()) {
        bail!("--row and --col must be provided together");
    }
    if (args.pixel_lon.is_some() || args.pixel_lat.is_some())
        && (args.pixel_lon.is_none() || args.pixel_lat.is_none())
    {
        bail!("--pixel-lon and --pixel-lat must be provided together");
    }

    let band_stacks =
        resolve_full_scene_band_stacks(&args.data_root, &args.source_name, args.lon, args.lat)?;
    let ordered_bands: Vec<String> = sorted_band_names(&band_stacks)
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    let first_chunk = band_stacks.values().next().unwrap().first().unwrap();
    let first_reader = RasterReader::open(first_chunk)?;
    let (rows, cols) = first_reader.shape();
    let (row_off, col_off) = if let (Some(r), Some(c)) = (args.row, args.col) {
        if r >= rows || c >= cols {
            bail!("pixel row/col out of bounds: row={r}, col={c}, shape=({rows}, {cols})");
        }
        (r, c)
    } else if let (Some(lon), Some(lat)) = (args.pixel_lon, args.pixel_lat) {
        full_scene_window_offset(&first_reader, 1, Some(lon), Some(lat))?
    } else {
        (rows / 2, cols / 2)
    };

    let mut band_cubes: BTreeMap<String, Array3<f64>> = BTreeMap::new();
    let mut band_timestamps: BTreeMap<String, Vec<String>> = BTreeMap::new();

    let load_start = Instant::now();
    for (band_name, chunk_paths) in &band_stacks {
        if chunk_paths.len() != 1 {
            bail!(
                "pixel-bench expects resolved one-path band stacks; got {} paths for {band_name}",
                chunk_paths.len()
            );
        }
        let chunk_path = &chunk_paths[0];
        eprintln!(
            "Loading {band_name} pixel row={row_off}, col={col_off} from {}",
            chunk_path.display()
        );
        let reader = RasterReader::open(chunk_path)?;
        let cube = read_all_bands_window_offset(&reader, row_off, col_off, 1, 1)?;
        let descs = extract_raw_band_descriptions(&reader)?;
        band_cubes.insert(band_name.clone(), cube);
        band_timestamps.insert(band_name.clone(), descs);
    }
    for cube in band_cubes.values_mut() {
        mask_invalid_sentinel2(cube);
    }
    let load_elapsed = load_start.elapsed();

    let target_start = Instant::now();
    let (target_time_str, _) = choose_shared_target_timestamp(
        &band_cubes,
        &band_timestamps,
        args.min_valid_ratio,
        args.late_fraction,
    )?;
    let target_elapsed = target_start.elapsed();
    let target_epoch = parse_iso8601_to_epoch_seconds(&target_time_str)
        .context("Failed to parse target timestamp")?;

    let mut masked_series: Vec<Vec<f64>> = Vec::with_capacity(ordered_bands.len());
    let mut ref_ts: Option<Vec<String>> = None;
    let mut raw_slices = 0usize;
    let mut collapsed_slices = 0usize;
    let mut target_values = Vec::with_capacity(ordered_bands.len());

    for band_name in &ordered_bands {
        let cube = band_cubes
            .get(band_name)
            .with_context(|| format!("Cube missing for band {band_name}"))?;
        let ts = band_timestamps
            .get(band_name)
            .with_context(|| format!("Timestamps missing for band {band_name}"))?;
        raw_slices = raw_slices.max(ts.len());
        let (collapsed_cube, collapsed_ts) = collapse_duplicate_timestamps(cube, ts);
        collapsed_slices = collapsed_slices.max(collapsed_ts.len());
        let (masked_cube, masked_ts, _target_idx, gt) =
            make_masked_time_series(&collapsed_cube, &collapsed_ts, &target_time_str)?;
        if let Some(existing) = &ref_ts {
            if existing != &masked_ts {
                bail!("pixel-bench requires aligned masked timestamps; band {band_name} differs");
            }
        } else {
            ref_ts = Some(masked_ts.clone());
        }
        masked_series.push(masked_cube.iter().copied().collect());
        target_values.push(gt[[0, 0]]);
    }

    let ref_ts = ref_ts.context("No masked timestamps built")?;
    let mut epochs: Vec<f64> = ref_ts
        .iter()
        .filter_map(|s| parse_iso8601_to_epoch_seconds(s))
        .collect();
    epochs.push(target_epoch);
    epochs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ts_days: Vec<f64> = epochs.iter().map(|&e| (e - epochs[0]) / 86400.0).collect();
    let target_day = (target_epoch - epochs[0]) / 86400.0;
    let masked_days: Vec<f64> = ts_days
        .iter()
        .copied()
        .filter(|&d| (d - target_day).abs() > 1e-9)
        .collect();
    if masked_days.len() != masked_series.first().map_or(0, Vec::len) {
        bail!(
            "masked timestamp count mismatch: {} days vs {} observations",
            masked_days.len(),
            masked_series.first().map_or(0, Vec::len)
        );
    }

    let config = load_nufrost_config(Some(std::path::Path::new("config/nufrost.json")))
        .unwrap_or_else(|_| default_nufrost_config());
    let valid_joint = (0..masked_days.len())
        .filter(|&i| masked_series.iter().all(|band| band[i].is_finite()))
        .count();

    let mut nufft_bins = 0usize;
    let mut nufft_power_sum = 0.0f64;
    let nufft_start = Instant::now();
    for _ in 0..args.repeats {
        let valid_idx: Vec<usize> = (0..masked_days.len())
            .filter(|&i| {
                masked_days[i].is_finite() && masked_series.iter().all(|b| b[i].is_finite())
            })
            .collect();
        if valid_idx.len() < (config.robust.min_obs as usize).max(3) {
            continue;
        }
        let t_sec: Vec<f64> = valid_idx
            .iter()
            .map(|&i| masked_days[i] * 86400.0)
            .collect();
        let t_min = t_sec.iter().copied().fold(f64::INFINITY, f64::min);
        let t_rel: Vec<f64> = t_sec.iter().map(|&t| t - t_min).collect();

        let mut spectrum_dims = Vec::with_capacity(masked_series.len());
        for band in &masked_series {
            let col: Vec<f64> = valid_idx.iter().map(|&i| band[i]).collect();
            spectrum_dims.push(robust_standardize(&col));
        }

        let (_freqs, power) = nufrost_core::nufft::type1_vector_power_kb(
            &t_rel,
            &spectrum_dims,
            config.spectral.modes as usize,
            nufrost_core::nufft::NufftOptions::default(),
        );
        nufft_bins = power.len();
        nufft_power_sum = power.iter().copied().filter(|v| v.is_finite()).sum();
    }
    let nufft_elapsed = nufft_start.elapsed();
    let nufft_per_call = nufft_elapsed.as_secs_f64() / args.repeats as f64;

    let mut predictions = Vec::new();
    let bench_start = Instant::now();
    for _ in 0..args.repeats {
        predictions = nufrost_pixel_vector(&masked_days, &masked_series, target_day, &config);
    }
    let bench_elapsed = bench_start.elapsed();
    let per_call = bench_elapsed.as_secs_f64() / args.repeats as f64;

    let rmse = {
        let mut n = 0usize;
        let mut sum_sq = 0.0;
        for (&p, &t) in predictions.iter().zip(target_values.iter()) {
            if p.is_finite() && t.is_finite() {
                let d = p - t;
                n += 1;
                sum_sq += d * d;
            }
        }
        if n == 0 {
            f64::NAN
        } else {
            (sum_sq / n as f64).sqrt()
        }
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "source_name": args.source_name,
            "scene_lon": args.lon,
            "scene_lat": args.lat,
            "row": row_off,
            "col": col_off,
            "ordered_bands": ordered_bands,
            "target_time": target_time_str,
            "raw_slices": raw_slices,
            "collapsed_slices": collapsed_slices,
            "masked_slices": masked_days.len(),
            "joint_valid_slices": valid_joint,
            "repeats": args.repeats,
            "load_seconds": load_elapsed.as_secs_f64(),
            "target_select_seconds": target_elapsed.as_secs_f64(),
            "nufft_prefix_total_seconds": nufft_elapsed.as_secs_f64(),
            "nufft_prefix_seconds_per_call": nufft_per_call,
            "nufft_bins": nufft_bins,
            "nufft_power_sum": nufft_power_sum,
            "bench_total_seconds": bench_elapsed.as_secs_f64(),
            "bench_seconds_per_call": per_call,
            "prediction": predictions,
            "target": target_values,
            "rmse": rmse,
        }))?
    );
    Ok(())
}

fn run_full_scene(args: &FullSceneArgs) -> Result<()> {
    if let Some(n) = args.run.n_jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("Failed to build rayon thread pool")?;
    }

    let methods = validate_full_scene_methods(&args.run.methods)?;

    if args.run.window_size.is_none() {
        let cache_dir = args.scene_cache.clone().unwrap_or_else(|| {
            scene_cache::default_scene_cache_dir(
                &default_scene_cache_root(&args.run.data_root, &args.run.source_name),
                &args.run.source_name,
                args.lon,
                args.lat,
            )
        });
        if !cache_dir.join("meta.json").is_file() || !cache_dir.join("cube.f32.bin").is_file() {
            scene_cache::build_scene_cache(
                &args.run.data_root,
                &default_scene_cache_root(&args.run.data_root, &args.run.source_name),
                &args.run.source_name,
                args.lon,
                args.lat,
                args.scene_cache.as_deref(),
            )?;
        }
        return run_full_scene_from_cache_streaming(args, &cache_dir);
    }

    let load_start = Instant::now();
    let scene_cache::LoadedScene {
        ordered_bands,
        mut band_cubes,
        band_timestamps,
        meta,
        cache_dir,
    } = if let Some(cache_dir) = &args.scene_cache {
        if args.run.window_size.is_some() {
            bail!("--scene-cache currently supports full-scene runs only, not --window-size");
        }
        eprintln!("Loading scene from mmap cache: {}", cache_dir.display());
        scene_cache::load_scene_cache(cache_dir)?
    } else if let Some(window_size) = args.run.window_size {
        scene_cache::load_scene_from_geotiffs_window(
            &args.run.data_root,
            &args.run.source_name,
            args.lon,
            args.lat,
            window_size,
            args.window_lon,
            args.window_lat,
        )?
    } else {
        scene_cache::load_or_build_scene_cache(
            &args.run.data_root,
            &default_scene_cache_root(&args.run.data_root, &args.run.source_name),
            &args.run.source_name,
            args.lon,
            args.lat,
        )?
    };
    let scene_load_seconds = load_start.elapsed().as_secs_f64();
    eprintln!("Scene load completed in {scene_load_seconds:.3}s");

    // 3. Mask invalid reflectance
    for cube in band_cubes.values_mut() {
        mask_invalid_sentinel2(cube);
    }

    // 4. Target timestamp selection
    let target_time_str = if let Some(cache_dir) = &cache_dir {
        if let Some(target) = scene_cache::load_cached_target_timestamp(
            cache_dir,
            args.run.min_valid_ratio,
            args.run.late_fraction,
        )? {
            eprintln!("Using cached shared target timestamp: {target}");
            target
        } else {
            eprintln!("Selecting shared target timestamp...");
            let (target, _completeness) = choose_shared_target_timestamp(
                &band_cubes,
                &band_timestamps,
                args.run.min_valid_ratio,
                args.run.late_fraction,
            )?;
            scene_cache::store_cached_target_timestamp(
                cache_dir,
                args.run.min_valid_ratio,
                args.run.late_fraction,
                &target,
            )?;
            target
        }
    } else {
        eprintln!("Selecting shared target timestamp...");
        let (target, _completeness) = choose_shared_target_timestamp(
            &band_cubes,
            &band_timestamps,
            args.run.min_valid_ratio,
            args.run.late_fraction,
        )?;
        target
    };
    eprintln!("Selected target timestamp: {target_time_str}");

    // 6. Hold-out: make masked time series per band
    let mut masked_cubes: BTreeMap<String, Array3<f64>> = BTreeMap::new();
    let mut masked_ts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut ground_truths: BTreeMap<String, ndarray::Array2<f64>> = BTreeMap::new();

    for band_name in &ordered_bands {
        let cube = band_cubes
            .get(band_name)
            .with_context(|| format!("Cube missing for band {band_name}"))?;
        let ts = band_timestamps
            .get(band_name)
            .with_context(|| format!("Timestamps missing for band {band_name}"))?;

        let ts_before = ts.len();
        let (cube, ts) = collapse_duplicate_timestamps(cube, ts);
        eprintln!(
            "Band {band_name}: collapsed {} timestamp slices to {}",
            ts_before,
            ts.len(),
        );

        let (mc, mts, _target_idx, gt) = make_masked_time_series(&cube, &ts, &target_time_str)?;
        masked_cubes.insert(band_name.clone(), mc);
        masked_ts.insert(band_name.clone(), mts);
        ground_truths.insert(band_name.clone(), gt);
    }

    // 8. Target time as epoch
    let target_epoch = parse_iso8601_to_epoch_seconds(&target_time_str)
        .context("Failed to parse target timestamp")?;

    // 10. Load configs
    let nufrost_config_path = args
        .nufrost_config
        .as_deref()
        .unwrap_or_else(|| std::path::Path::new("config/nufrost.json"));
    let mut nufrost_conf =
        load_nufrost_config(Some(nufrost_config_path)).unwrap_or_else(|_| default_nufrost_config());
    if let Some(mode) = &args.frequency_selection {
        nufrost_conf.spectral.selection = FrequencySelection::from_name(mode)
            .with_context(|| format!("unknown frequency selection mode: {mode}"))?;
    }
    let hants_conf = default_hants_config();
    let zhu2015_conf = default_zhu2015_config();

    // 11. Per-method reconstruction
    let source_name = &args.run.source_name;
    let output_root = &args.run.output_root;
    let loc_token = full_scene::location_token(args.lon, args.lat);

    let mut prediction_outputs = serde_json::Map::new();
    let mut metrics_by_method = serde_json::Map::new();

    for method in &methods {
        let method_str = method.as_str();
        eprintln!("Reconstructing with {method_str}...");

        let predictions: BTreeMap<String, ndarray::Array2<f64>> = if method_str == "nufrost" {
            reconstruct_nufrost_vector_scene(
                &ordered_bands,
                &masked_cubes,
                &masked_ts,
                target_epoch,
                &nufrost_conf,
            )?
        } else {
            // Per-band independent reconstruction for HANTS and Zhu2015 baselines.
            ordered_bands
                .par_iter()
                .map(|band_name| {
                    let cube = masked_cubes.get(band_name).unwrap();
                    let ts_strs = masked_ts.get(band_name).unwrap();

                    // Convert string timestamps to epoch seconds
                    let mut epochs: Vec<f64> = ts_strs
                        .iter()
                        .filter_map(|s| parse_iso8601_to_epoch_seconds(s))
                        .collect();
                    // Include target epoch for reference
                    epochs.push(target_epoch);
                    epochs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    // Relative days from earliest epoch of this band+target set
                    let ts_days: Vec<f64> =
                        epochs.iter().map(|&e| (e - epochs[0]) / 86400.0).collect();
                    let target_day = (target_epoch - epochs[0]) / 86400.0;

                    // The masked ts_days are all except the target
                    let masked_days: Vec<f64> = ts_days
                        .iter()
                        .copied()
                        .filter(|&d| (d - target_day).abs() > 1e-9)
                        .collect();

                    let (n_bands, rows, cols) = cube.dim();
                    let mut pred = ndarray::Array2::<f64>::from_elem((rows, cols), f64::NAN);
                    eprintln!(
                        "  {method_str}/{band_name}: {rows} rows x {cols} cols, {n_bands} time slices"
                    );
                    let progress = AtomicUsize::new(0);
                    let progress_step = (rows / 10).max(1);

                    // Parallel per-row reconstruction
                    pred.axis_iter_mut(ndarray::Axis(0))
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(r, mut row_out)| {
                            let mut ts_buf = Vec::with_capacity(n_bands);
                            let mut obs_buf = Vec::with_capacity(n_bands);
                            for c in 0..cols {
                                ts_buf.clear();
                                obs_buf.clear();
                                for b in 0..n_bands {
                                    let v = cube[[b, r, c]];
                                    if v.is_finite() {
                                        ts_buf.push(masked_days[b]);
                                        obs_buf.push(v);
                                    }
                                }
                                if !ts_buf.is_empty() {
                                    match method_str {
                                        "hants" => {
                                            let val = hants_pixel(
                                                &ts_buf,
                                                &obs_buf,
                                                target_day,
                                                hants_conf.nof,
                                                &hants_conf.sf,
                                                hants_conf.valid_min,
                                                hants_conf.valid_max,
                                                hants_conf.fet,
                                                hants_conf.dod,
                                                hants_conf.period,
                                            );
                                            row_out[c] =
                                                if val.is_finite() { val } else { f64::NAN };
                                        }
                                        "zhu2015" => {
                                            let result = fit_predict_pixel(
                                                &ts_buf,
                                                &obs_buf,
                                                target_day,
                                                zhu2015_conf.lasso_alpha,
                                            );
                                            row_out[c] = if result.prediction.is_finite() {
                                                result.prediction
                                            } else {
                                                f64::NAN
                                            };
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                            if done == rows || done % progress_step == 0 {
                                eprintln!("    {method_str}/{band_name} rows: {done}/{rows}");
                            }
                        });

                    (band_name.clone(), pred)
                })
                .collect()
        };

        // Write merged band stack
        let output_path = build_scene_stack_output_path(
            output_root,
            method_str,
            source_name,
            args.lon,
            args.lat,
            &target_time_str,
            "prediction",
        );
        write_band_stack(&output_path, &predictions, &ordered_bands, &meta)?;
        eprintln!("  Wrote {}", output_path.display());

        let metrics = prediction_metrics_json(&predictions, &ground_truths, &ordered_bands)?;
        if let Some(rmse) = metrics.get("overall_rmse").and_then(|v| v.as_f64()) {
            eprintln!("  Overall RMSE: {rmse:.6}");
        }
        prediction_outputs.insert(
            method.clone(),
            serde_json::Value::String(output_path.display().to_string()),
        );
        metrics_by_method.insert(method.clone(), metrics);
    }

    // 12. Write ground truth
    let gt_path = build_ground_truth_output_path(
        output_root,
        source_name,
        args.lon,
        args.lat,
        &target_time_str,
    );
    write_band_stack(&gt_path, &ground_truths, &ordered_bands, &meta)?;
    eprintln!("Ground truth written to {}", gt_path.display());

    // 13. Write summary JSON. When rerunning only a subset of methods, merge
    // into the existing summary so older baseline metrics remain available.
    let summary_dir = output_root.join("run_summaries");
    fs::create_dir_all(&summary_dir)?;
    let safe_time = target_time_str.replace(':', "-");
    let loc_token6 = full_scene::location_output_token(args.lon, args.lat);
    let summary_path = summary_dir.join(format!(
        "reconstruction_summary_{source_name}_{loc_token6}_{safe_time}.json"
    ));

    let mut merged_prediction_outputs = if let Ok(bytes) = fs::read(&summary_path) {
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| {
                v.get("prediction_outputs")
                    .and_then(|m| m.as_object())
                    .cloned()
            })
            .unwrap_or_default()
    } else {
        serde_json::Map::new()
    };
    for (method, output) in prediction_outputs {
        merged_prediction_outputs.insert(method, output);
    }

    let mut merged_metrics = if let Ok(bytes) = fs::read(&summary_path) {
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| v.get("metrics").and_then(|m| m.as_object()).cloned())
            .unwrap_or_default()
    } else {
        serde_json::Map::new()
    };
    for (method, metric) in metrics_by_method {
        merged_metrics.insert(method, metric);
    }

    let mut merged_methods: Vec<String> = merged_metrics.keys().cloned().collect();
    for method in &methods {
        if !merged_methods.iter().any(|m| m == method) {
            merged_methods.push(method.clone());
        }
    }
    merged_methods.sort();

    let now_iso = chrono::Utc::now().to_rfc3339();
    let summary: serde_json::Value = serde_json::json!({
        "source_name": source_name,
        "lon": args.lon,
        "lat": args.lat,
        "location_token": loc_token,
        "target_time": target_time_str,
        "target_epoch": target_epoch,
        "methods_run": merged_methods,
        "methods_updated": methods,
        "ordered_bands": ordered_bands,
        "min_valid_ratio": args.run.min_valid_ratio,
        "late_fraction": args.run.late_fraction,
        "window_size": args.run.window_size,
        "window_lon": args.window_lon,
        "window_lat": args.window_lat,
        "scene_cache": cache_dir.as_ref().map(|p| p.display().to_string()),
        "scene_load_seconds": scene_load_seconds,
        "prediction_outputs": merged_prediction_outputs,
        "ground_truth_output": gt_path.display().to_string(),
        "metrics": merged_metrics,
        "generated_at": now_iso,
    });
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    eprintln!("Summary written to {}", summary_path.display());

    eprintln!("Full-scene reconstruction complete.");
    Ok(())
}

fn run_batch_full_scene(args: &BatchFullSceneArgs) -> Result<()> {
    validate_full_scene_methods(&args.run.methods)?;
    if let Some(n) = args.run.n_jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("Failed to build rayon thread pool")?;
    }

    let mut locations = discover_complete_sentinel_locations(&args.run.data_root, &args.run.source_name)?;
    if let Some(limit) = args.limit {
        locations.truncate(limit);
    }
    if locations.is_empty() {
        bail!(
            "No complete Sentinel-2 locations found in {}",
            source_data_dir(&args.run.data_root, &args.run.source_name).display()
        );
    }
    eprintln!("Discovered {} complete locations.", locations.len());

    let mut runs = Vec::new();
    let mut failures = Vec::new();

    for (idx, (lon, lat)) in locations.iter().copied().enumerate() {
        eprintln!(
            "Batch location {}/{}: lon={lon:.4}, lat={lat:.4}",
            idx + 1,
            locations.len()
        );
        let scene_args = FullSceneArgs {
            run: SceneRun {
                source_name: args.run.source_name.clone(),
                output_root: args.run.output_root.clone(),
                data_root: args.run.data_root.clone(),
                methods: args.run.methods.clone(),
                n_jobs: None,
                window_size: args.run.window_size,
                min_valid_ratio: args.run.min_valid_ratio,
                late_fraction: args.run.late_fraction,
            },
            lon,
            lat,
            window_lon: None,
            window_lat: None,
            scene_cache: None,
            nufrost_config: None,
            frequency_selection: None,
        };

        match run_full_scene(&scene_args) {
            Ok(()) => {
                runs.push(serde_json::json!({
                    "lon": lon,
                    "lat": lat,
                    "status": "ok",
                }));
            }
            Err(err) => {
                eprintln!("Location lon={lon:.4}, lat={lat:.4} failed: {err:#}");
                failures.push(serde_json::json!({
                    "lon": lon,
                    "lat": lat,
                    "status": "failed",
                    "error": format!("{err:#}"),
                }));
                if !args.continue_on_error {
                    bail!("Batch stopped after location lon={lon:.4}, lat={lat:.4} failed");
                }
            }
        }
    }

    let summary_dir = args.run.output_root.join("run_summaries");
    fs::create_dir_all(&summary_dir)?;
    let summary_path = summary_dir.join(format!(
        "batch_summary_{}_{}.json",
        args.run.source_name,
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
    ));
    let summary = serde_json::json!({
        "source_name": args.run.source_name,
        "data_root": args.run.data_root.display().to_string(),
        "output_root": args.run.output_root.display().to_string(),
        "methods": args.run.methods,
        "window_size": args.run.window_size,
        "min_valid_ratio": args.run.min_valid_ratio,
        "late_fraction": args.run.late_fraction,
        "n_locations": locations.len(),
        "n_success": runs.len(),
        "n_failed": failures.len(),
        "runs": runs,
        "failures": failures,
    });
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    eprintln!("Batch summary written to {}", summary_path.display());

    if !failures.is_empty() {
        bail!(
            "Batch completed with {} failed locations; see {}",
            failures.len(),
            summary_path.display()
        );
    }
    Ok(())
}

struct SampleCacheView {
    dir: PathBuf,
    meta: sample_cache::SampleCacheMeta,
    samples: Mmap,
    mask: Mmap,
    index: Mmap,
    scene_times: Vec<f64>,
    sample_record_bytes: usize,
    mask_record_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct EvalTask {
    sample_idx: usize,
    holdout_seed: u64,
}

#[derive(Debug, Clone)]
struct EvalStats {
    attempted: usize,
    evaluated: usize,
    skipped: usize,
    failed: usize,
    sum_sq: Vec<f64>,
    sum_abs: Vec<f64>,
    count: Vec<usize>,
    indices: IndexEvalStats,
    gap_regular: GapCurveStats,
    gap_periodic: GapCurveStats,
}

struct EvalOne {
    truth: Vec<f64>,
    pred: Vec<f64>,
    gap_regular: f64,
    gap_periodic: f64,
}

enum EvalMethodConfig {
    Nufrost(NufrostConfig),
    Hants(HantsConfig),
    Zhu2015(Zhu2015Config),
}

impl EvalStats {
    fn new(n_bands: usize, gap_bins: usize) -> Self {
        Self {
            attempted: 0,
            evaluated: 0,
            skipped: 0,
            failed: 0,
            sum_sq: vec![0.0; n_bands],
            sum_abs: vec![0.0; n_bands],
            count: vec![0; n_bands],
            indices: IndexEvalStats::new(),
            gap_regular: GapCurveStats::new(gap_bins, n_bands),
            gap_periodic: GapCurveStats::new(gap_bins, n_bands),
        }
    }

    fn add_attempt(&mut self, result: Option<EvalOne>) {
        self.attempted += 1;
        match result {
            Some(eval) => {
                let mut any = false;
                for b in 0..self.count.len() {
                    let truth = eval.truth[b];
                    let pred = eval.pred[b];
                    if truth.is_finite() && pred.is_finite() {
                        let err = pred - truth;
                        self.sum_sq[b] += err * err;
                        self.sum_abs[b] += err.abs();
                        self.count[b] += 1;
                        any = true;
                    }
                }
                if any {
                    self.indices.add(&eval.truth, &eval.pred);
                    self.gap_regular
                        .add(eval.gap_regular, &eval.truth, &eval.pred);
                    self.gap_periodic
                        .add(eval.gap_periodic, &eval.truth, &eval.pred);
                    self.evaluated += 1;
                } else {
                    self.failed += 1;
                }
            }
            None => {
                self.skipped += 1;
            }
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.attempted += other.attempted;
        self.evaluated += other.evaluated;
        self.skipped += other.skipped;
        self.failed += other.failed;
        for b in 0..self.count.len() {
            self.sum_sq[b] += other.sum_sq[b];
            self.sum_abs[b] += other.sum_abs[b];
            self.count[b] += other.count[b];
        }
        self.indices.merge(other.indices);
        self.gap_regular.merge(other.gap_regular);
        self.gap_periodic.merge(other.gap_periodic);
        self
    }
}

#[derive(Debug, Clone)]
struct IndexEvalStats {
    ndvi: ScalarErrorStats,
    evi: ScalarErrorStats,
    ndwi: ScalarErrorStats,
    ndsi: ScalarErrorStats,
    ndmi: ScalarErrorStats,
    nbr: ScalarErrorStats,
}

impl IndexEvalStats {
    fn new() -> Self {
        Self {
            ndvi: ScalarErrorStats::new(),
            evi: ScalarErrorStats::new(),
            ndwi: ScalarErrorStats::new(),
            ndsi: ScalarErrorStats::new(),
            ndmi: ScalarErrorStats::new(),
            nbr: ScalarErrorStats::new(),
        }
    }

    fn add(&mut self, truth: &[f64], pred: &[f64]) {
        add_index_error(&mut self.ndvi, truth, pred, ndvi_from_bands);
        add_index_error(&mut self.evi, truth, pred, evi_from_bands);
        add_index_error(&mut self.ndwi, truth, pred, ndwi_from_bands);
        add_index_error(&mut self.ndsi, truth, pred, ndsi_from_bands);
        add_index_error(&mut self.ndmi, truth, pred, ndmi_from_bands);
        add_index_error(&mut self.nbr, truth, pred, nbr_from_bands);
    }

    fn merge(&mut self, other: Self) {
        self.ndvi.merge(other.ndvi);
        self.evi.merge(other.evi);
        self.ndwi.merge(other.ndwi);
        self.ndsi.merge(other.ndsi);
        self.ndmi.merge(other.ndmi);
        self.nbr.merge(other.nbr);
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "NDVI": self.ndvi.to_json(),
            "EVI": self.evi.to_json(),
            "NDWI": self.ndwi.to_json(),
            "NDSI": self.ndsi.to_json(),
            "NDMI": self.ndmi.to_json(),
            "NBR": self.nbr.to_json(),
            "definition": {
                "scale": "Input Sentinel-2 reflectance bands are divided by 10000 before computing indices.",
                "NDVI": "(B8 - B4) / (B8 + B4)",
                "EVI": "2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1)",
                "NDWI": "(B3 - B8) / (B3 + B8)",
                "NDSI": "(B3 - B11) / (B3 + B11)",
                "NDMI": "(B8 - B11) / (B8 + B11)",
                "NBR": "(B8 - B12) / (B8 + B12)",
                "validity": "Index is skipped when the denominator is not finite or has absolute value <= 1e-6."
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ScalarErrorStats {
    sum_sq: f64,
    sum_abs: f64,
    sum: f64,
    count: usize,
}

impl ScalarErrorStats {
    fn new() -> Self {
        Self {
            sum_sq: 0.0,
            sum_abs: 0.0,
            sum: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, err: f64) {
        if !err.is_finite() {
            return;
        }
        self.sum_sq += err * err;
        self.sum_abs += err.abs();
        self.sum += err;
        self.count += 1;
    }

    fn merge(&mut self, other: Self) {
        self.sum_sq += other.sum_sq;
        self.sum_abs += other.sum_abs;
        self.sum += other.sum;
        self.count += other.count;
    }

    fn rmse(&self) -> f64 {
        if self.count > 0 {
            (self.sum_sq / self.count as f64).sqrt()
        } else {
            f64::NAN
        }
    }

    fn mae(&self) -> f64 {
        if self.count > 0 {
            self.sum_abs / self.count as f64
        } else {
            f64::NAN
        }
    }

    fn bias(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            f64::NAN
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "rmse": self.rmse(),
            "mae": self.mae(),
            "bias": self.bias(),
            "count": self.count,
        })
    }
}

#[derive(Debug, Clone)]
struct GapCurveStats {
    n_bins: usize,
    n_bands: usize,
    min: f64,
    max: f64,
    sum_sq: Vec<Vec<f64>>,
    sum_abs: Vec<Vec<f64>>,
    count: Vec<Vec<usize>>,
    sample_count: Vec<usize>,
}

impl GapCurveStats {
    fn new(n_bins: usize, n_bands: usize) -> Self {
        let n_bins = n_bins.max(1);
        Self {
            n_bins,
            n_bands,
            min: 0.0,
            max: 0.2,
            sum_sq: vec![vec![0.0; n_bands]; n_bins],
            sum_abs: vec![vec![0.0; n_bands]; n_bins],
            count: vec![vec![0; n_bands]; n_bins],
            sample_count: vec![0; n_bins],
        }
    }

    fn add(&mut self, gap_value: f64, truth: &[f64], pred: &[f64]) {
        if !gap_value.is_finite() {
            return;
        }
        let bin = self.bin_index(gap_value);
        let mut any = false;
        for b in 0..self.n_bands {
            let truth = truth[b];
            let pred = pred[b];
            if truth.is_finite() && pred.is_finite() {
                let err = pred - truth;
                self.sum_sq[bin][b] += err * err;
                self.sum_abs[bin][b] += err.abs();
                self.count[bin][b] += 1;
                any = true;
            }
        }
        if any {
            self.sample_count[bin] += 1;
        }
    }

    fn merge(&mut self, other: Self) {
        debug_assert_eq!(self.n_bins, other.n_bins);
        debug_assert_eq!(self.n_bands, other.n_bands);
        for i in 0..self.n_bins {
            self.sample_count[i] += other.sample_count[i];
            for b in 0..self.n_bands {
                self.sum_sq[i][b] += other.sum_sq[i][b];
                self.sum_abs[i][b] += other.sum_abs[i][b];
                self.count[i][b] += other.count[i][b];
            }
        }
    }

    fn bin_index(&self, value: f64) -> usize {
        let clamped = value.max(self.min);
        if clamped >= self.max {
            return self.n_bins - 1;
        }
        let width = (self.max - self.min) / self.n_bins as f64;
        ((clamped - self.min) / width).floor() as usize
    }

    fn bin_bounds(&self, bin: usize) -> (f64, f64) {
        let width = (self.max - self.min) / self.n_bins as f64;
        let start = self.min + width * bin as f64;
        let end = if bin + 1 == self.n_bins {
            self.max
        } else {
            start + width
        };
        (start, end)
    }

    fn rows_json(&self, name: &str, bands: &[String]) -> serde_json::Value {
        let rows: Vec<_> = (0..self.n_bins)
            .map(|bin| {
                let (start, end) = self.bin_bounds(bin);
                let mut band_metrics = serde_json::Map::new();
                let mut total_sq = 0.0;
                let mut total_abs = 0.0;
                let mut total_count = 0usize;
                for (b, band) in bands.iter().enumerate() {
                    let count = self.count[bin][b];
                    let rmse = if count > 0 {
                        (self.sum_sq[bin][b] / count as f64).sqrt()
                    } else {
                        f64::NAN
                    };
                    let mae = if count > 0 {
                        self.sum_abs[bin][b] / count as f64
                    } else {
                        f64::NAN
                    };
                    total_sq += self.sum_sq[bin][b];
                    total_abs += self.sum_abs[bin][b];
                    total_count += count;
                    band_metrics.insert(
                        band.clone(),
                        serde_json::json!({
                            "rmse": rmse,
                            "mae": mae,
                            "count": count,
                        }),
                    );
                }
                let overall_rmse = if total_count > 0 {
                    (total_sq / total_count as f64).sqrt()
                } else {
                    f64::NAN
                };
                let overall_mae = if total_count > 0 {
                    total_abs / total_count as f64
                } else {
                    f64::NAN
                };
                serde_json::json!({
                    "gap_index": name,
                    "bin": bin,
                    "range_start": start,
                    "range_end": end,
                    "range_label": if bin + 1 == self.n_bins {
                        format!("[{start:.3},+inf)")
                    } else {
                        format!("[{start:.3},{end:.3})")
                    },
                    "sample_count": self.sample_count[bin],
                    "overall": {
                        "rmse": overall_rmse,
                        "mae": overall_mae,
                        "count": total_count,
                    },
                    "bands": band_metrics,
                })
            })
            .collect();
        serde_json::Value::Array(rows)
    }
}

fn run_eval_sample_cache(args: &EvalSampleCacheArgs) -> Result<()> {
    if args.n_eval == 0 {
        bail!("--n-eval must be > 0");
    }
    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("Failed to build rayon thread pool")?;
    }

    let eval_config = match args.method.as_str() {
        "nufrost" => EvalMethodConfig::Nufrost(load_nufrost_config(args.config.as_deref())?),
        "hants" => EvalMethodConfig::Hants(load_hants_config(args.config.as_deref())?),
        "zhu2015" => EvalMethodConfig::Zhu2015(load_zhu2015_config(args.config.as_deref())?),
        other => bail!("Unsupported eval method: {other}"),
    };
    let cache = open_sample_cache(&args.cache_dir)?;
    let n_eval = args.n_eval.min(cache.meta.n_samples);
    let mut rng = EvalRng::new(args.seed);
    let tasks: Vec<EvalTask> = (0..n_eval)
        .map(|_| EvalTask {
            sample_idx: rng.gen_range(cache.meta.n_samples),
            holdout_seed: rng.next_u64(),
        })
        .collect();

    eprintln!(
        "sample cache eval: method={}, cache={}, n_eval={}, cache_samples={}, max_times={}, bands={}, threads={}",
        args.method,
        cache.dir.display(),
        n_eval,
        cache.meta.n_samples,
        cache.meta.max_times,
        cache.meta.n_bands,
        rayon::current_num_threads()
    );

    let prediction_writer = if let Some(path) = &args.output_prediction_csv {
        Some(Arc::new(Mutex::new(create_prediction_csv_writer(
            path,
            &cache.meta.bands,
        )?)))
    } else {
        None
    };

    let started = Instant::now();
    let progress = AtomicUsize::new(0);
    let stats = tasks
        .par_iter()
        .enumerate()
        .map(|(idx, task)| {
            let result = evaluate_sample_holdout(&cache, *task, &eval_config, args.min_joint_valid)
                .with_context(|| {
                    format!("failed to evaluate task {idx} sample {}", task.sample_idx)
                })
                .ok()
                .flatten();
            if let (Some(writer), Some(eval)) = (&prediction_writer, &result) {
                if let Ok(mut writer) = writer.lock() {
                    let _ = write_prediction_csv_row(&mut writer, task, eval);
                }
            }
            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % SAMPLE_CACHE_EVAL_PROGRESS_INTERVAL == 0 || done == n_eval {
                let elapsed = started.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed.max(1.0e-9);
                let eta = (n_eval - done) as f64 / rate.max(1.0e-9);
                eprintln!(
                    "sample cache eval progress: {done}/{n_eval} ({:.2}%), rate={rate:.1} samples/s, elapsed={}, eta={}",
                    done as f64 * 100.0 / n_eval as f64,
                    format_duration(elapsed),
                    format_duration(eta)
                );
            }
            let mut stats = EvalStats::new(cache.meta.n_bands, args.gap_bins);
            stats.add_attempt(result);
            stats
        })
        .reduce(
            || EvalStats::new(cache.meta.n_bands, args.gap_bins),
            |left, right| left.merge(right),
        );

    let elapsed = started.elapsed().as_secs_f64();
    let mut band_metrics = serde_json::Map::new();
    let mut total_sq = 0.0;
    let mut total_abs = 0.0;
    let mut total_count = 0usize;
    for (b, band) in cache.meta.bands.iter().enumerate() {
        let count = stats.count[b];
        let rmse = if count > 0 {
            (stats.sum_sq[b] / count as f64).sqrt()
        } else {
            f64::NAN
        };
        let mae = if count > 0 {
            stats.sum_abs[b] / count as f64
        } else {
            f64::NAN
        };
        total_sq += stats.sum_sq[b];
        total_abs += stats.sum_abs[b];
        total_count += count;
        band_metrics.insert(
            band.clone(),
            serde_json::json!({
                "rmse": rmse,
                "mae": mae,
                "count": count,
            }),
        );
    }
    let overall_rmse = if total_count > 0 {
        (total_sq / total_count as f64).sqrt()
    } else {
        f64::NAN
    };
    let overall_mae = if total_count > 0 {
        total_abs / total_count as f64
    } else {
        f64::NAN
    };

    let summary = serde_json::json!({
        "method": args.method,
        "cache_dir": cache.dir.display().to_string(),
        "n_eval_requested": args.n_eval,
        "n_eval_used": n_eval,
        "seed": args.seed,
        "min_joint_valid_after_holdout": args.min_joint_valid,
        "attempted": stats.attempted,
        "evaluated": stats.evaluated,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "overall": {
            "rmse": overall_rmse,
            "mae": overall_mae,
            "count": total_count,
        },
        "bands": band_metrics,
        "indices": stats.indices.to_json(),
        "gap_indices": {
            "definition": {
                "gap_regular": "I_gap = sum(l_i^2) / N^2 over adjacent valid-observation gaps after holdout",
                "gap_periodic": "I_periodic = |sum(l_i * exp(j*2*pi*tau_i))|^2 / N^2 over adjacent valid-observation gaps after holdout",
                "l_i": "time interval between adjacent remaining valid observations",
                "tau_i": "gap midpoint normalized to [0,1] over the sample time span"
            },
            "n_bins": args.gap_bins.max(1),
            "gap_regular": stats.gap_regular.rows_json("gap_regular", &cache.meta.bands),
            "gap_periodic": stats.gap_periodic.rows_json("gap_periodic", &cache.meta.bands),
        },
        "elapsed_seconds": elapsed,
    });

    println!("{}", serde_json::to_string_pretty(&summary)?);
    if let Some(path) = &args.output_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, serde_json::to_string_pretty(&summary)?)?;
        eprintln!("Sample cache evaluation written to {}", path.display());
    }
    if let Some(path) = &args.output_gap_csv {
        write_gap_curve_csv(path, &args.method, &cache.meta.bands, &stats)?;
        eprintln!("Sample cache gap curves written to {}", path.display());
    }
    if let Some(path) = &args.output_prediction_csv {
        if let Some(writer) = prediction_writer {
            if let Ok(mut writer) = writer.lock() {
                writer.flush()?;
            }
        }
        eprintln!("Sample cache prediction CSV written to {}", path.display());
    }
    Ok(())
}

fn run_sample_pixel_curve(args: &SamplePixelCurveArgs) -> Result<()> {
    let cache = open_sample_cache(&args.cache_dir)?;
    if args.sample_idx >= cache.meta.n_samples {
        bail!(
            "--sample-idx {} is out of bounds for cache with {} samples",
            args.sample_idx,
            cache.meta.n_samples
        );
    }
    let (scene_id, row, col, n_times) = read_sample_index_record(&cache, args.sample_idx)?;
    if scene_id >= cache.meta.scenes.len() || n_times == 0 || n_times > cache.meta.max_times {
        bail!(
            "invalid sample index record: scene_id={scene_id}, n_times={n_times}, max_times={}",
            cache.meta.max_times
        );
    }
    let scene = &cache.meta.scenes[scene_id];
    if scene.time_offset + n_times > cache.scene_times.len() {
        bail!("sample scene time axis exceeds scene_times file");
    }

    let mask_start = args
        .sample_idx
        .checked_mul(cache.mask_record_bytes)
        .context("mask offset overflow")?;
    let mask = &cache.mask[mask_start..mask_start + cache.mask_record_bytes];
    let valid_positions: Vec<usize> = (0..n_times).filter(|&i| mask[i] != 0).collect();
    if valid_positions.len() <= args.min_joint_valid {
        bail!(
            "sample has only {} joint-valid dates, which is not enough for min_joint_valid={}",
            valid_positions.len(),
            args.min_joint_valid
        );
    }

    let times = &cache.scene_times[scene.time_offset..scene.time_offset + n_times];
    let sample_start = args
        .sample_idx
        .checked_mul(cache.sample_record_bytes)
        .context("sample offset overflow")?;
    let sample = &cache.samples[sample_start..sample_start + cache.sample_record_bytes];
    let mut full_obs = vec![vec![f64::NAN; n_times]; cache.meta.n_bands];
    for t in 0..n_times {
        if mask[t] == 0 {
            continue;
        }
        for b in 0..cache.meta.n_bands {
            let offset = t
                .checked_mul(cache.meta.n_bands)
                .and_then(|v| v.checked_add(b))
                .and_then(|v| v.checked_mul(4))
                .context("sample value offset overflow")?;
            full_obs[b][t] =
                f32::from_le_bytes(sample[offset..offset + 4].try_into().unwrap()) as f64;
        }
    }

    let nufrost = load_nufrost_config(args.nufrost_config.as_deref())?;
    let hants = load_hants_config(args.hants_config.as_deref())?;
    let zhu2015 = load_zhu2015_config(args.zhu2015_config.as_deref())?;

    if let Some(parent) = args.output_csv.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut writer = BufWriter::new(
        File::create(&args.output_csv)
            .with_context(|| format!("Cannot create {}", args.output_csv.display()))?,
    );
    writeln!(
        writer,
        "sample_idx,scene_id,scene_lon,scene_lat,row,col,n_times,valid_count,target_index,kind,time_index,time_epoch,time_day,time_year,band,observed,nufrost,hants,zhu2015"
    )?;

    let time0 = times.iter().copied().fold(f64::INFINITY, f64::min);
    if !time0.is_finite() {
        bail!("sample has no finite time axis");
    }
    let rel_days: Vec<f64> = times.iter().map(|&t| (t - time0) / 86400.0).collect();
    let valid_count = valid_positions.len();
    let target = if let Some(seed) = args.holdout_seed {
        let mut rng = EvalRng::new(seed);
        valid_positions[rng.gen_range(valid_positions.len())]
    } else {
        valid_positions[valid_positions.len() / 2]
    };
    let fit_valid_count = valid_count.saturating_sub(1);
    if fit_valid_count < args.min_joint_valid {
        bail!(
            "sample has only {} training dates after holding out target {}, min_joint_valid={}",
            fit_valid_count,
            target,
            args.min_joint_valid
        );
    }

    let mut obs = full_obs.clone();
    for band_obs in &mut obs {
        band_obs[target] = f64::NAN;
    }

    let dense_n = 240usize;
    let min_day = valid_positions
        .iter()
        .map(|&i| rel_days[i])
        .fold(f64::INFINITY, f64::min);
    let max_day = valid_positions
        .iter()
        .map(|&i| rel_days[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let dense_days: Vec<f64> = if max_day > min_day {
        (0..dense_n)
            .map(|i| min_day + (max_day - min_day) * (i as f64) / ((dense_n - 1) as f64))
            .collect()
    } else {
        vec![rel_days[target]]
    };

    let nufrost_fit = nufrost_fit_pixel_vector(&rel_days, &obs, &nufrost);
    let nufrost_curves = nufrost_predict_vector_curve(&nufrost_fit, &dense_days);
    let nufrost_target = nufrost_predict_vector(&nufrost_fit, rel_days[target]);

    let mut hants_curves = vec![vec![f64::NAN; dense_days.len()]; cache.meta.n_bands];
    let mut hants_target = vec![f64::NAN; cache.meta.n_bands];
    let mut zhu_curves = vec![vec![f64::NAN; dense_days.len()]; cache.meta.n_bands];
    let mut zhu_target = vec![f64::NAN; cache.meta.n_bands];
    for b in 0..cache.meta.n_bands {
        let h_fit = hants_fit(
            &rel_days,
            &obs[b],
            hants.nof,
            &hants.sf,
            hants.valid_min,
            hants.valid_max,
            hants.fet,
            hants.dod,
            hants.period,
        );
        hants_curves[b] = hants_predict_curve(&h_fit, &dense_days);
        hants_target[b] = hants_predict(&h_fit, rel_days[target]);

        if let Some(z_fit) = fit_segment(&rel_days, &obs[b], 0, rel_days.len(), zhu2015.lasso_alpha)
        {
            for (i, &day) in dense_days.iter().enumerate() {
                zhu_curves[b][i] = segment_predict(&z_fit, day);
            }
            zhu_target[b] = segment_predict(&z_fit, rel_days[target]);
        }
    }

    for &i in &valid_positions {
        let kind = if i == target { "target" } else { "observed" };
        for (band_idx, band) in cache.meta.bands.iter().enumerate() {
            let observed = full_obs[band_idx][i];
            if !observed.is_finite() {
                continue;
            }
            writeln!(
                writer,
                "{},{},{:.8},{:.8},{},{},{},{},{},{},{},{:.8},{:.8},{:.10},{},{:.8},{:.8},{:.8},{:.8}",
                args.sample_idx,
                scene_id,
                scene.lon,
                scene.lat,
                row,
                col,
                n_times,
                valid_count,
                target,
                kind,
                i,
                times[i],
                rel_days[i],
                rel_days[i] / 365.25,
                band,
                observed,
                if i == target {
                    nufrost_target.get(band_idx).copied().unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                },
                if i == target {
                    hants_target.get(band_idx).copied().unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                },
                if i == target {
                    zhu_target.get(band_idx).copied().unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                },
            )?;
        }
    }

    for (curve_idx, &day) in dense_days.iter().enumerate() {
        let epoch = time0 + day * 86400.0;
        for (band_idx, band) in cache.meta.bands.iter().enumerate() {
            writeln!(
                writer,
                "{},{},{:.8},{:.8},{},{},{},{},{},curve,{},{:.8},{:.8},{:.10},{},,{:.8},{:.8},{:.8}",
                args.sample_idx,
                scene_id,
                scene.lon,
                scene.lat,
                row,
                col,
                n_times,
                valid_count,
                target,
                curve_idx,
                epoch,
                day,
                day / 365.25,
                band,
                nufrost_curves
                    .get(band_idx)
                    .and_then(|v| v.get(curve_idx))
                    .copied()
                    .unwrap_or(f64::NAN),
                hants_curves[band_idx][curve_idx],
                zhu_curves[band_idx][curve_idx],
            )?;
        }
    }
    writer.flush()?;
    eprintln!(
        "single-fit sample pixel curve written: {}, sample_idx={}, scene_id={}, row={}, col={}, target_index={}, dense_points={}",
        args.output_csv.display(),
        args.sample_idx,
        scene_id,
        row,
        col,
        target,
        dense_days.len()
    );
    Ok(())
}

fn create_prediction_csv_writer(path: &Path, bands: &[String]) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut writer = BufWriter::new(
        File::create(path).with_context(|| format!("Cannot create {}", path.display()))?,
    );
    write!(writer, "sample_idx,holdout_seed,gap_regular,gap_periodic")?;
    for band in bands {
        write!(writer, ",truth_{band}")?;
    }
    for band in bands {
        write!(writer, ",pred_{band}")?;
    }
    writeln!(writer)?;
    Ok(writer)
}

fn write_prediction_csv_row(
    writer: &mut BufWriter<File>,
    task: &EvalTask,
    eval: &EvalOne,
) -> std::io::Result<()> {
    write!(
        writer,
        "{},{},{:.12},{:.12}",
        task.sample_idx, task.holdout_seed, eval.gap_regular, eval.gap_periodic
    )?;
    for value in &eval.truth {
        write!(writer, ",{value:.8}")?;
    }
    for value in &eval.pred {
        write!(writer, ",{value:.8}")?;
    }
    writeln!(writer)
}

fn write_gap_curve_csv(
    path: &Path,
    method: &str,
    bands: &[String],
    stats: &EvalStats,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut out = String::from(
        "method,gap_index,bin,range_start,range_end,range_label,band,sample_count,count,rmse,mae\n",
    );
    append_gap_curve_csv_rows(&mut out, method, "gap_regular", bands, &stats.gap_regular);
    append_gap_curve_csv_rows(&mut out, method, "gap_periodic", bands, &stats.gap_periodic);
    fs::write(path, out)?;
    Ok(())
}

fn append_gap_curve_csv_rows(
    out: &mut String,
    method: &str,
    gap_name: &str,
    bands: &[String],
    curve: &GapCurveStats,
) {
    for bin in 0..curve.n_bins {
        let (start, end) = curve.bin_bounds(bin);
        let label = if bin + 1 == curve.n_bins {
            format!(">= {start:.3}")
        } else {
            format!("{start:.3}-{end:.3}")
        };
        let mut total_sq = 0.0;
        let mut total_abs = 0.0;
        let mut total_count = 0usize;
        for b in 0..curve.n_bands {
            total_sq += curve.sum_sq[bin][b];
            total_abs += curve.sum_abs[bin][b];
            total_count += curve.count[bin][b];
        }
        let overall_rmse = if total_count > 0 {
            (total_sq / total_count as f64).sqrt()
        } else {
            f64::NAN
        };
        let overall_mae = if total_count > 0 {
            total_abs / total_count as f64
        } else {
            f64::NAN
        };
        out.push_str(&format!(
            "{method},{gap_name},{bin},{start:.8},{end:.8},{label},overall,{},{},{overall_rmse:.8},{overall_mae:.8}\n",
            curve.sample_count[bin], total_count
        ));
        for (b, band) in bands.iter().enumerate() {
            let count = curve.count[bin][b];
            let rmse = if count > 0 {
                (curve.sum_sq[bin][b] / count as f64).sqrt()
            } else {
                f64::NAN
            };
            let mae = if count > 0 {
                curve.sum_abs[bin][b] / count as f64
            } else {
                f64::NAN
            };
            out.push_str(&format!(
                "{method},{gap_name},{bin},{start:.8},{end:.8},{label},{band},{},{},{rmse:.8},{mae:.8}\n",
                curve.sample_count[bin], count
            ));
        }
    }
}

fn open_sample_cache(dir: &Path) -> Result<SampleCacheView> {
    let meta_path = dir.join("meta.json");
    let meta: sample_cache::SampleCacheMeta = serde_json::from_slice(
        &fs::read(&meta_path).with_context(|| format!("Cannot read {}", meta_path.display()))?,
    )
    .with_context(|| format!("Invalid sample cache meta {}", meta_path.display()))?;
    let sample_record_bytes = meta
        .max_times
        .checked_mul(meta.n_bands)
        .and_then(|v| v.checked_mul(4))
        .context("sample record byte size overflow")?;
    let mask_record_bytes = meta.max_times;
    let samples = mmap_file(&dir.join(&meta.sample_file))?;
    let mask = mmap_file(&dir.join(&meta.mask_file))?;
    let index = mmap_file(&dir.join(&meta.index_file))?;
    let scene_time_mmap = mmap_file(&dir.join(&meta.scene_time_file))?;
    if scene_time_mmap.len() % 8 != 0 {
        bail!("scene time file length is not divisible by 8");
    }
    let scene_times = scene_time_mmap
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("chunk length checked")))
        .collect();
    let expected_sample_bytes = meta
        .n_samples
        .checked_mul(sample_record_bytes)
        .context("sample file byte size overflow")?;
    let expected_mask_bytes = meta
        .n_samples
        .checked_mul(mask_record_bytes)
        .context("mask file byte size overflow")?;
    let expected_index_bytes = meta
        .n_samples
        .checked_mul(4 * 8)
        .context("index file byte size overflow")?;
    if samples.len() != expected_sample_bytes {
        bail!(
            "sample file has wrong size: expected {}, got {}",
            expected_sample_bytes,
            samples.len()
        );
    }
    if mask.len() != expected_mask_bytes {
        bail!(
            "mask file has wrong size: expected {}, got {}",
            expected_mask_bytes,
            mask.len()
        );
    }
    if index.len() != expected_index_bytes {
        bail!(
            "index file has wrong size: expected {}, got {}",
            expected_index_bytes,
            index.len()
        );
    }
    Ok(SampleCacheView {
        dir: dir.to_path_buf(),
        meta,
        samples,
        mask,
        index,
        scene_times,
        sample_record_bytes,
        mask_record_bytes,
    })
}

fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("Cannot open {}", path.display()))?;
    unsafe { Mmap::map(&file).with_context(|| format!("Cannot mmap {}", path.display())) }
}

fn evaluate_sample_holdout(
    cache: &SampleCacheView,
    task: EvalTask,
    config: &EvalMethodConfig,
    min_joint_valid: usize,
) -> Result<Option<EvalOne>> {
    let (scene_id, n_times) = read_sample_index(cache, task.sample_idx)?;
    if scene_id >= cache.meta.scenes.len() || n_times == 0 || n_times > cache.meta.max_times {
        return Ok(None);
    }
    let scene = &cache.meta.scenes[scene_id];
    if scene.time_offset + n_times > cache.scene_times.len() {
        bail!("sample scene time axis exceeds scene_times file");
    }
    let mask_start = task
        .sample_idx
        .checked_mul(cache.mask_record_bytes)
        .context("mask offset overflow")?;
    let mask = &cache.mask[mask_start..mask_start + cache.mask_record_bytes];
    let valid_positions: Vec<usize> = (0..n_times).filter(|&i| mask[i] != 0).collect();
    if valid_positions.len() <= min_joint_valid {
        return Ok(None);
    }
    let mut rng = EvalRng::new(task.holdout_seed);
    let holdout = valid_positions[rng.gen_range(valid_positions.len())];
    let times = &cache.scene_times[scene.time_offset..scene.time_offset + n_times];
    let mut fit_valid_positions = valid_positions;
    fit_valid_positions.retain(|&i| i != holdout);
    let (gap_regular, gap_periodic) = compute_gap_indices(times, &fit_valid_positions);
    let sample_start = task
        .sample_idx
        .checked_mul(cache.sample_record_bytes)
        .context("sample offset overflow")?;
    let sample = &cache.samples[sample_start..sample_start + cache.sample_record_bytes];
    let mut obs = vec![vec![f64::NAN; n_times]; cache.meta.n_bands];
    let mut truth = vec![f64::NAN; cache.meta.n_bands];
    for t in 0..n_times {
        if mask[t] == 0 {
            continue;
        }
        for b in 0..cache.meta.n_bands {
            let offset = t
                .checked_mul(cache.meta.n_bands)
                .and_then(|v| v.checked_add(b))
                .and_then(|v| v.checked_mul(4))
                .context("sample value offset overflow")?;
            let value = f32::from_le_bytes(sample[offset..offset + 4].try_into().unwrap()) as f64;
            if t == holdout {
                truth[b] = value;
            } else {
                obs[b][t] = value;
            }
        }
    }
    if truth.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    let time0 = times.iter().copied().fold(f64::INFINITY, f64::min);
    if !time0.is_finite() {
        return Ok(None);
    }
    let rel_days: Vec<f64> = times.iter().map(|&t| (t - time0) / 86400.0).collect();
    let pred = predict_eval_holdout(&rel_days, &obs, rel_days[holdout], config);
    Ok(Some(EvalOne {
        truth,
        pred,
        gap_regular,
        gap_periodic,
    }))
}

fn compute_gap_indices(times: &[f64], valid_positions: &[usize]) -> (f64, f64) {
    if valid_positions.len() < 2 || times.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let first = times[0];
    let last = times[times.len() - 1];
    let span = last - first;
    if !span.is_finite() || span <= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let mut sum_l2 = 0.0;
    let mut real = 0.0;
    let mut imag = 0.0;
    for pair in valid_positions.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if a >= times.len() || b >= times.len() || b <= a {
            continue;
        }
        let ta = times[a];
        let tb = times[b];
        let l = tb - ta;
        if !l.is_finite() || l <= 0.0 {
            continue;
        }
        let tau = (((ta + tb) * 0.5) - first) / span;
        let phase = std::f64::consts::TAU * tau;
        sum_l2 += l * l;
        real += l * phase.cos();
        imag += l * phase.sin();
    }

    let denom = span * span;
    let gap_regular = sum_l2 / denom;
    let gap_periodic = (real * real + imag * imag) / denom;
    (gap_regular.clamp(0.0, 1.0), gap_periodic.clamp(0.0, 1.0))
}

fn predict_eval_holdout(
    times: &[f64],
    obs: &[Vec<f64>],
    target_time: f64,
    config: &EvalMethodConfig,
) -> Vec<f64> {
    match config {
        EvalMethodConfig::Nufrost(cfg) => nufrost_pixel_vector(times, obs, target_time, cfg),
        EvalMethodConfig::Hants(cfg) => obs
            .iter()
            .map(|band_obs| {
                hants_pixel(
                    times,
                    band_obs,
                    target_time,
                    cfg.nof,
                    &cfg.sf,
                    cfg.valid_min,
                    cfg.valid_max,
                    cfg.fet,
                    cfg.dod,
                    cfg.period,
                )
            })
            .collect(),
        EvalMethodConfig::Zhu2015(cfg) => obs
            .iter()
            .map(|band_obs| {
                fit_predict_pixel(times, band_obs, target_time, cfg.lasso_alpha).prediction
            })
            .collect(),
    }
}

fn add_index_error(
    stats: &mut ScalarErrorStats,
    truth: &[f64],
    pred: &[f64],
    index_fn: fn(&[f64]) -> Option<f64>,
) {
    if let (Some(truth_index), Some(pred_index)) = (index_fn(truth), index_fn(pred)) {
        stats.add(pred_index - truth_index);
    }
}

fn ndvi_from_bands(bands: &[f64]) -> Option<f64> {
    let s2 = sentinel2_reflectance(bands)?;
    normalized_difference(s2.b8, s2.b4)
}

fn evi_from_bands(bands: &[f64]) -> Option<f64> {
    let s2 = sentinel2_reflectance(bands)?;
    valid_ratio_denominator(s2.b8 + 6.0 * s2.b4 - 7.5 * s2.b2 + 1.0)
        .map(|denom| 2.5 * (s2.b8 - s2.b4) / denom)
}

fn ndwi_from_bands(bands: &[f64]) -> Option<f64> {
    let s2 = sentinel2_reflectance(bands)?;
    normalized_difference(s2.b3, s2.b8)
}

fn ndsi_from_bands(bands: &[f64]) -> Option<f64> {
    let s2 = sentinel2_reflectance(bands)?;
    normalized_difference(s2.b3, s2.b11)
}

fn ndmi_from_bands(bands: &[f64]) -> Option<f64> {
    let s2 = sentinel2_reflectance(bands)?;
    normalized_difference(s2.b8, s2.b11)
}

fn nbr_from_bands(bands: &[f64]) -> Option<f64> {
    let s2 = sentinel2_reflectance(bands)?;
    normalized_difference(s2.b8, s2.b12)
}

#[derive(Debug, Clone, Copy)]
struct Sentinel2Reflectance {
    b2: f64,
    b3: f64,
    b4: f64,
    b8: f64,
    b11: f64,
    b12: f64,
}

fn sentinel2_reflectance(bands: &[f64]) -> Option<Sentinel2Reflectance> {
    if bands.len() < 6 {
        return None;
    }
    let reflectance = Sentinel2Reflectance {
        b2: bands[0] / 10000.0,
        b3: bands[1] / 10000.0,
        b4: bands[2] / 10000.0,
        b8: bands[3] / 10000.0,
        b11: bands[4] / 10000.0,
        b12: bands[5] / 10000.0,
    };
    if [
        reflectance.b2,
        reflectance.b3,
        reflectance.b4,
        reflectance.b8,
        reflectance.b11,
        reflectance.b12,
    ]
    .iter()
    .all(|v| v.is_finite())
    {
        Some(reflectance)
    } else {
        None
    }
}

fn normalized_difference(a: f64, b: f64) -> Option<f64> {
    valid_ratio_denominator(a + b).map(|denom| (a - b) / denom)
}

fn valid_ratio_denominator(value: f64) -> Option<f64> {
    if value.is_finite() && value.abs() > 1.0e-6 {
        Some(value)
    } else {
        None
    }
}

fn read_sample_index(cache: &SampleCacheView, sample_idx: usize) -> Result<(usize, usize)> {
    let (scene_id, _row, _col, n_times) = read_sample_index_record(cache, sample_idx)?;
    Ok((scene_id, n_times))
}

fn read_sample_index_record(
    cache: &SampleCacheView,
    sample_idx: usize,
) -> Result<(usize, usize, usize, usize)> {
    let start = sample_idx
        .checked_mul(4 * 8)
        .context("sample index offset overflow")?;
    let index = &cache.index[start..start + 4 * 8];
    let scene_id = read_u64_le(&index[0..8]) as usize;
    let row = read_u64_le(&index[8..16]) as usize;
    let col = read_u64_le(&index[16..24]) as usize;
    let n_times = read_u64_le(&index[24..32]) as usize;
    Ok((scene_id, row, col, n_times))
}

fn read_u64_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("u64 slice length"))
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "unknown".to_string();
    }
    if seconds >= 3600.0 {
        format!("{:.2} h", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1} min", seconds / 60.0)
    } else {
        format!("{seconds:.1} s")
    }
}

#[derive(Debug, Clone, Copy)]
struct EvalRng {
    state: u64,
}

impl EvalRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            (self.next_u64() % upper as u64) as usize
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.algorithm {
        Algorithm::Nufrost(args) => match detect_input_mode(&args.shared)? {
            InputMode::NpzFixture(fixture) => run_nufrost_npz(args, &fixture),
            InputMode::GeoTiff { reader, output } => run_nufrost_geotiff(args, &reader, &output),
        },
        Algorithm::Hants(args) => match detect_input_mode(&args.shared)? {
            InputMode::NpzFixture(fixture) => run_hants_npz(args, &fixture),
            InputMode::GeoTiff { reader, output } => run_hants_geotiff(args, &reader, &output),
        },
        Algorithm::Zhu2015(args) => match detect_input_mode(&args.shared)? {
            InputMode::NpzFixture(fixture) => run_zhu2015_npz(args, &fixture),
            InputMode::GeoTiff { reader, output } => run_zhu2015_geotiff(args, &reader, &output),
        },
        Algorithm::FullScene(args) => run_full_scene(args),
        Algorithm::BuildSceneCache(args) => run_build_scene_cache(args),
        Algorithm::BuildSampleCache(args) => run_build_sample_cache(args),
        Algorithm::EvalSampleCache(args) => run_eval_sample_cache(args),
        Algorithm::SamplePixelCurve(args) => run_sample_pixel_curve(args),
        Algorithm::BatchFullScene(args) => run_batch_full_scene(args),
        Algorithm::PixelBench(args) => run_pixel_bench(args),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Parse args and return the CLI struct.
    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("nufrost-cli").chain(args.iter().copied()))
            .expect("valid args should parse")
    }

    /// Parse args and expect failure.
    fn parse_fails(args: &[&str]) -> String {
        Cli::try_parse_from(std::iter::once("nufrost-cli").chain(args.iter().copied()))
            .unwrap_err()
            .to_string()
    }

    // ── Subcommand routing ──────────────────────────────────────────────

    #[test]
    fn nufrost_subcommand() {
        let cli = parse(&["nufrost"]);
        assert!(matches!(cli.algorithm, Algorithm::Nufrost(_)));
    }

    #[test]
    fn hants_subcommand() {
        let cli = parse(&["hants"]);
        assert!(matches!(cli.algorithm, Algorithm::Hants(_)));
    }

    #[test]
    fn zhu2015_subcommand() {
        let cli = parse(&["zhu2015"]);
        assert!(matches!(cli.algorithm, Algorithm::Zhu2015(_)));
    }

    #[test]
    fn build_sample_cache_options() {
        let cli = parse(&[
            "build-sample-cache",
            "--source-name",
            "sentinel-2",
            "--scene-cache-root",
            "data/cache/sentinel-2/16-sites",
            "--output",
            "data/tests/sample-cache-test",
            "--n-samples",
            "128",
            "--min-joint-valid",
            "8",
            "--seed",
            "123",
            "--max-attempts-multiplier",
            "5",
            "--limit-scenes",
            "2",
        ]);
        match cli.algorithm {
            Algorithm::BuildSampleCache(args) => {
                assert_eq!(args.source_name, "sentinel-2");
                assert_eq!(
                    args.scene_cache_root,
                    PathBuf::from("data/cache/sentinel-2/16-sites")
                );
                assert_eq!(args.output, PathBuf::from("data/tests/sample-cache-test"));
                assert_eq!(args.n_samples, 128);
                assert_eq!(args.min_joint_valid, 8);
                assert_eq!(args.seed, 123);
                assert_eq!(args.max_attempts_multiplier, 5);
                assert_eq!(args.limit_scenes, Some(2));
            }
            other => panic!("expected BuildSampleCache, got {other:?}"),
        }
    }

    #[test]
    fn batch_full_scene_options() {
        let cli = parse(&[
            "batch-full-scene",
            "--source-name",
            "sentinel-2",
            "--data-root",
            "data",
            "--output-root",
            "/tmp/out",
            "--methods",
            "nufrost,hants",
            "--window-size",
            "3",
            "--n-jobs",
            "4",
            "--limit",
            "2",
            "--continue-on-error",
        ]);
        match &cli.algorithm {
            Algorithm::BatchFullScene(args) => {
                assert_eq!(args.run.source_name, "sentinel-2");
                assert_eq!(args.run.data_root, std::path::PathBuf::from("data"));
                assert_eq!(args.run.output_root, std::path::PathBuf::from("/tmp/out"));
                assert_eq!(args.run.methods, "nufrost,hants");
                assert_eq!(args.run.window_size, Some(3));
                assert_eq!(args.run.n_jobs, Some(4));
                assert_eq!(args.limit, Some(2));
                assert!(args.continue_on_error);
            }
            _ => panic!("expected BatchFullScene"),
        }
    }

    #[test]
    fn parse_sentinel_location_handles_negative_lat_and_chunks() {
        let parsed = parse_sentinel_location_from_filename(
            "COPERNICUS_S2_HARMONIZED_B2_lon-62.2159_lat-3.4653-0000000000-0000001024.tif",
        )
        .expect("chunked negative-lat filename should parse");
        assert_eq!(parsed.0, "B2");
        assert_eq!(parsed.1, "-62.2159");
        assert_eq!(parsed.2, "-3.4653");

        let parsed = parse_sentinel_location_from_filename(
            "COPERNICUS_S2_HARMONIZED_B11_lon124.7115_lat52.3352-0000000000-0000001280.tif",
        )
        .expect("chunked positive-lat filename should parse");
        assert_eq!(parsed.0, "B11");
        assert_eq!(parsed.1, "124.7115");
        assert_eq!(parsed.2, "52.3352");
    }

    #[test]
    fn build_scene_cache_options() {
        let cli = parse(&[
            "build-scene-cache",
            "--source-name",
            "sentinel-2",
            "--lon",
            "94.2605",
            "--lat",
            "29.7733",
            "--data-root",
            "data",
            "--cache-root",
            "data/cache/sentinel-2/16-sites",
            "--output",
            "/tmp/scene-cache",
        ]);
        match &cli.algorithm {
            Algorithm::BuildSceneCache(args) => {
                assert_eq!(args.source_name, "sentinel-2");
                assert_eq!(args.lon, 94.2605);
                assert_eq!(args.lat, 29.7733);
                assert_eq!(args.data_root, std::path::PathBuf::from("data"));
                assert_eq!(
                    args.cache_root,
                    std::path::PathBuf::from("data/cache/sentinel-2/16-sites")
                );
                assert_eq!(
                    args.output,
                    Some(std::path::PathBuf::from("/tmp/scene-cache"))
                );
            }
            _ => panic!("expected BuildSceneCache"),
        }
    }

    #[test]
    fn full_scene_scene_cache_option() {
        let cli = parse(&[
            "full-scene",
            "--source-name",
            "sentinel-2",
            "--lon",
            "94.2605",
            "--lat",
            "29.7733",
            "--scene-cache",
            "data/cache/sentinel-2/16-sites/lon94_260500_lat29_773300",
        ]);
        match &cli.algorithm {
            Algorithm::FullScene(args) => {
                assert_eq!(
                    args.scene_cache,
                    Some(std::path::PathBuf::from(
                        "data/cache/sentinel-2/16-sites/lon94_260500_lat29_773300"
                    ))
                );
            }
            _ => panic!("expected FullScene"),
        }
    }

    #[test]
    fn pixel_bench_options() {
        let cli = parse(&[
            "pixel-bench",
            "--source-name",
            "sentinel-2",
            "--lon",
            "94.2605",
            "--lat",
            "29.7733",
            "--row",
            "10",
            "--col",
            "20",
            "--repeats",
            "3",
        ]);
        match &cli.algorithm {
            Algorithm::PixelBench(args) => {
                assert_eq!(args.source_name, "sentinel-2");
                assert_eq!(args.lon, 94.2605);
                assert_eq!(args.lat, 29.7733);
                assert_eq!(args.row, Some(10));
                assert_eq!(args.col, Some(20));
                assert_eq!(args.repeats, 3);
            }
            _ => panic!("expected PixelBench"),
        }
    }

    // ── NPZ mode option parsing ─────────────────────────────────────────

    #[test]
    fn nufrost_all_options_npz() {
        let cli = parse(&[
            "nufrost",
            "--config",
            "/tmp/cfg.json",
            "--data",
            "/tmp/data.npz",
            "--target-time",
            "372.7",
            "--output",
            "/tmp/out.txt",
            "--threads",
            "4",
        ]);
        match &cli.algorithm {
            Algorithm::Nufrost(args) => {
                assert_eq!(
                    args.config.as_deref(),
                    Some(std::path::Path::new("/tmp/cfg.json"))
                );
                assert_eq!(
                    args.shared.data.as_deref(),
                    Some(std::path::Path::new("/tmp/data.npz"))
                );
                assert_eq!(args.shared.target_time, Some(372.7));
                assert_eq!(
                    args.shared.output.as_deref(),
                    Some(std::path::Path::new("/tmp/out.txt"))
                );
                assert_eq!(args.shared.threads, 4);
                assert!(args.shared.input_geotiff.is_none());
            }
            _ => panic!("expected Nufrost"),
        }
    }

    #[test]
    fn nufrost_geotiff_options() {
        let cli = parse(&[
            "nufrost",
            "--input-geotiff",
            "/tmp/input.tif",
            "--output",
            "/tmp/out.tif",
        ]);
        match &cli.algorithm {
            Algorithm::Nufrost(args) => {
                assert!(args.config.is_none());
                assert!(args.shared.data.is_none());
                assert_eq!(
                    args.shared.input_geotiff.as_deref(),
                    Some(std::path::Path::new("/tmp/input.tif"))
                );
                assert_eq!(
                    args.shared.output.as_deref(),
                    Some(std::path::Path::new("/tmp/out.tif"))
                );
            }
            _ => panic!("expected Nufrost"),
        }
    }

    #[test]
    fn hants_all_options() {
        let cli = parse(&[
            "hants",
            "--config",
            "/tmp/ch.json",
            "--data",
            "/tmp/d.npz",
            "-t",
            "200.0",
            "-o",
            "/tmp/o.txt",
        ]);
        match &cli.algorithm {
            Algorithm::Hants(args) => {
                assert_eq!(
                    args.config.as_deref(),
                    Some(std::path::Path::new("/tmp/ch.json"))
                );
                assert_eq!(args.shared.target_time, Some(200.0));
                assert_eq!(
                    args.shared.output.as_deref(),
                    Some(std::path::Path::new("/tmp/o.txt"))
                );
            }
            _ => panic!("expected Hants"),
        }
    }

    #[test]
    fn zhu2015_all_options() {
        let cli = parse(&[
            "zhu2015",
            "--config",
            "/tmp/cz.json",
            "--data",
            "/tmp/dz.npz",
            "-t",
            "500.0",
        ]);
        match &cli.algorithm {
            Algorithm::Zhu2015(args) => {
                assert_eq!(
                    args.config.as_deref(),
                    Some(std::path::Path::new("/tmp/cz.json"))
                );
                assert_eq!(args.shared.target_time, Some(500.0));
            }
            _ => panic!("expected Zhu2015"),
        }
    }

    // ── Defaults ────────────────────────────────────────────────────────

    #[test]
    fn nufrost_minimal_args() {
        let cli = parse(&["nufrost"]);
        match &cli.algorithm {
            Algorithm::Nufrost(args) => {
                assert!(args.config.is_none());
                assert!(args.shared.data.is_none());
                assert!(args.shared.target_time.is_none());
                assert!(args.shared.output.is_none());
                assert_eq!(args.shared.threads, 1);
                assert!(args.shared.input_geotiff.is_none());
            }
            _ => panic!("expected Nufrost"),
        }
    }

    // ── Error handling ──────────────────────────────────────────────────

    #[test]
    fn unknown_algorithm() {
        let err = parse_fails(&["invalid_alg"]);
        assert!(
            err.contains("invalid_alg") || err.contains("unrecognized") || err.contains("doesn't"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_required_data() {
        // No --data or --input-geotiff — but this is a runtime check, not clap parse error.
        // clap should accept this as valid since all args are optional.
        let cli = parse(&["nufrost", "--output", "/tmp/out.txt"]);
        match &cli.algorithm {
            Algorithm::Nufrost(args) => {
                assert!(args.shared.data.is_none());
                assert!(args.shared.input_geotiff.is_none());
            }
            _ => panic!("expected Nufrost"),
        }
    }

    // ── Contract: shared pool flags must be removed ──────────────────────

    /// Contract test: `--spectral-top-k` must NOT be accepted by `full-scene`.
    ///
    /// FAILS now (flag exists) → will PASS after T6 removes it from
    /// `FullSceneArgs`.
    #[test]
    fn full_scene_rejects_spectral_top_k() {
        let err = parse_fails(&[
            "full-scene",
            "--source-name",
            "sentinel-2",
            "--lon",
            "100.0",
            "--lat",
            "25.0",
            "--spectral-top-k",
            "4",
        ]);
        assert!(
            err.contains("spectral-top-k")
                || err.contains("unrecognized")
                || err.contains("unexpected")
                || err.contains("error")
                || err.contains("invalid"),
            "expected '--spectral-top-k' to be rejected, got: {err}"
        );
    }

    /// Contract test: `--preferred-top-k` must NOT be accepted by `full-scene`.
    ///
    /// FAILS now (flag exists) → will PASS after T6 removes it from
    /// `FullSceneArgs`.
    #[test]
    fn full_scene_rejects_preferred_top_k() {
        let err = parse_fails(&[
            "full-scene",
            "--source-name",
            "sentinel-2",
            "--lon",
            "100.0",
            "--lat",
            "25.0",
            "--preferred-top-k",
            "4",
        ]);
        assert!(
            err.contains("preferred-top-k")
                || err.contains("unrecognized")
                || err.contains("unexpected")
                || err.contains("error")
                || err.contains("invalid"),
            "expected '--preferred-top-k' to be rejected, got: {err}"
        );
    }

    /// Contract test: `full-scene --help` must NOT mention shared-pool flags.
    ///
    /// FAILS now (help text includes them) → will PASS after T6 removes them.
    #[test]
    fn full_scene_help_omits_shared_pool_flags() {
        let mut cmd = Cli::command();
        let full_scene_cmd = cmd
            .find_subcommand_mut("full-scene")
            .expect("full-scene subcommand must exist");
        let mut buf: Vec<u8> = Vec::new();
        full_scene_cmd.write_help(&mut buf).unwrap();
        let help = String::from_utf8(buf).unwrap();

        assert!(
            !help.contains("spectral-top-k"),
            "help must NOT mention --spectral-top-k (shared pool flag to be removed)\nHelp:\n{help}"
        );
        assert!(
            !help.contains("preferred-top-k"),
            "help must NOT mention --preferred-top-k (shared pool flag to be removed)\nHelp:\n{help}"
        );
    }
}
