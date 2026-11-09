# HAPO paper evaluations

The solver in `crates/hapo-core` is the one archived with the paper's HAPO
holdout and full-scene runs. Its source SHA-256 is
`be4a788375127231bde2582eb72c42ba1ad3b3a7ab87196e467cdc5cc42ef4f1`.
The holdout runner separates the archived input handling and metrics into
modules. The full-scene runner uses the archived source-selection policy from
this directory. The archived dependency lockfile is retained.

The benchmark expects the project data root as its first argument. Use a new
output directory for each run. The input data and earlier predictions are read
in place.

## Single-date holdout and gap targets

```sh
cargo build --release --locked --manifest-path tools/hapo-benchmark/Cargo.toml --bins
tools/hapo-benchmark/target/release/hapo-benchmark data NEW_HOLDOUT_OUTPUT 99969
tools/hapo-benchmark/target/release/gap_supplement data NEW_GAP_OUTPUT
```

The holdout runner reads the cached 100,000 pixel samples, frozen
`tests/paper_supported_rerun_20261010/sources.json`, and saved paired predictions.
It verifies the site, tile, date, retained acquisition count, and target values
before fitting HAPO on the same 99,969 tasks. It reuses the predictions of the
other methods. Predictions and convergence diagnostics are written to JSONL.

The gap runner reads the saved target records and masks in
`tests/gap16_fig10_20261010/chunk_*/`. It predicts the 15,431 common targets at
all seven exclusion widths from 0 to 60 days, using the exact retained indices.

The `--fit` mode reads a JSON object with `days`, `values`, `query`, and an
optional `initial` vector from stdin. It writes the prediction, fitted
coefficients, scaling and trend parameters, and solver diagnostics. The
paper's representative curves and whole-interval driver use this mode.

## Full-scene comparison

Build with GDAL available. When using the local GDAL prefix on macOS, set
`GDAL_HOME` and `GDAL_VERSION` for the build and `DYLD_LIBRARY_PATH` for the run.

```sh
GDAL_HOME="$PWD/.conda/gdal" GDAL_VERSION=3.10.3 cargo build --release --locked --manifest-path tools/hapo-benchmark/Cargo.toml --features scene --bin full_scene
DYLD_LIBRARY_PATH="$PWD/.conda/gdal/lib" tools/hapo-benchmark/target/release/full_scene data NEW_SCENE_OUTPUT 0,5,15
```

The scene runner reads the frozen scene manifests and shared source records to
recover the paper targets and uses the saved rasters for the comparisons. It
writes HAPO GeoTIFFs in the original band order and reports scores on the
common finite support. An optional final argument limits processed rows for a
smoke run.

## Run records

The corresponding run records are under these paths in the data root:

- `tests/hapo_20261104/v1-diagnostics/` for the paired holdout.
- `tests/hapo_full_scene_20261104/v1/` for the full-scene comparison.
- `tests/hapo_paper_supplement_20261104/gap-exact-masks/` for the gap targets.
