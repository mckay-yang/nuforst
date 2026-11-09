# NUFROST paper reproduction

This standalone Cargo workspace preserves the Supported V1 implementation and configuration used in the 2026-10-10 paper experiments. The solver, time and frequency calculations, temporal support modules, and three configuration files are byte-for-byte copies of the experiment's source snapshot. The historical hashes are in [source_hashes.json](source_hashes.json). They also list the original runner paths and hashes recorded during the experiment.

The examples reuse the repository's HANTS and Zhu2015 crates. Their pixel fitting code matches the original experiment. Zhu2015 supplies the LHR baseline. The examples also emit the historical `original` NUFROST comparison where the archived runners did so. The `nufrost` or `supported_v1` rows identify the main paper method.

The sample runner and source-selection module come from the paper snapshot. The full-scene runner comes from the archived 2026-10-11 resumed run. The centered-gap runner comes from `tests/gap16_fig10_bundle_20261010`, and the whole-interval runner comes from `tests/gap_source_consistent_20261010` in the data archive. The examples use `NUFROST_DATA_ROOT` in place of the original machine-specific paths and embed the frozen configuration.

## Build and run

Use the project's GDAL installation. With the project-local setup, run the following commands from the repository root.

```sh
paper_gdal="$PWD/.conda/gdal"
export PATH="$paper_gdal/bin:$PATH"
export PKG_CONFIG_PATH="$paper_gdal/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export NUFROST_DATA_ROOT="/absolute/path/to/nufrost-data"
cargo build --locked --release --manifest-path tools/paper-reproduction/Cargo.toml --examples
```

The GDAL library directory must be on the runtime linker path. For macOS, one option is to set `RUSTFLAGS="-C link-arg=-Wl,-rpath,$paper_gdal/lib"` before the build. For Linux, set `LD_LIBRARY_PATH` to include the GDAL library directory.

Use fresh output directories. The optional final argument to the first two commands limits the number of attempted targets. The scene runner accepts an optional maximum row count after the output path and supports resuming compatible output.

```sh
cargo run --locked --release --manifest-path tools/paper-reproduction/Cargo.toml \
  --example paper_sample_rerun -- /path/to/new/sample-results 100000
cargo run --locked --release --manifest-path tools/paper-reproduction/Cargo.toml \
  --example paper_gap16_rerun -- /path/to/new/target-gap-results 16000
cargo run --locked --release --manifest-path tools/paper-reproduction/Cargo.toml \
  --example paper_scene_rerun -- 0 /path/to/new/aoi-0-results
cargo run --locked --release --manifest-path tools/paper-reproduction/Cargo.toml \
  --example gap_source_consistent -- /path/to/new/whole-interval-results 0 8
```

The data root is the original experiment data tree. It must contain `cache/sentinel-2/100k/` and `tests/paper_supported_rerun_20261010/sources.json`. Scene reconstruction additionally reads `cache/sentinel-2/16-sites/`. Centered gaps use `products/reconstruction_evaluation/controlled_gap_v01/manifest.csv`. Whole-interval reconstruction uses `tests/gap_untouched_quality_20261010/manifest.csv` and `tests/gap_source_consistent_20261010/cohort/retained_indices.json`.

The whole-interval runner reproduces predictions for the 39 records in the archived evaluation run. The manuscript's Table II uses the 29 eligible records in `tests/gap_source_consistent_20261010/cohort/selection_audit.csv`. For each gap width, select eligible sample IDs and rows with `hidden=true` and a real observation index. Compute mean squared error within each sample, average those means across samples, then take the square root.

## Archived results for checking

- Single-date predictions: `tests/paper_supported_rerun_20261010/paired_holdout_v2/predictions.csv`.
- Full scenes: `tests/paper_supported_rerun_20261010/scenes/site_<id>/`.
- Centered gaps: `tests/gap16_fig10_20261010/chunk_<id>/`.
- Whole-interval predictions: `tests/gap_source_consistent_20261010/evaluation/predictions.csv`.

Compare rows by task or sample identity, method and, where applicable, gap width. Parallel execution can change the order of saved rows.
