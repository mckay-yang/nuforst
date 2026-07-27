# NUFROST Agent Notes

## Project-Specific Notes

- This is now a Rust-first research repo. The active implementation lives under `crates/`.
- Treat `NUFROST`, `Zhu2015`, and `HANTS` as three separate reconstruction algorithms. `Zhu2015` and `HANTS` are comparison baselines, not sub-parts of NUFROST.
- In prose and docs, write the algorithm name as `NUFROST`; keep Rust crate identifiers lowercase/kebab-case (`nufrost-core`, `hants-core`, `zhu2015-core`).

## Branch Policy

- Always retain the long-lived branches `main`, `develop`, and `feature/nufrost-cd`.
- Name feature branches with the `feature/` prefix.
- Name experimental branches with the `exp/` prefix.
- Name temporary Codex branches with the `codex/` prefix.
- Name backup or archived branches with the `archive/` prefix.
- Do not delete or rename existing branches without explicit user approval.

## Rust Layout

- `crates/gdal`: GeoTIFF/VRT I/O, timestamp parsing, full-scene helpers, generic raster traversal.
- `crates/nufrost-core`: NUFROST algorithm and NUFFT/fitting logic.
- `crates/hants-core`: HANTS baseline algorithm.
- `crates/zhu2015-core`: Zhu2015 baseline algorithm.
- `crates/nufrost-cli`: command-line entrypoint.

Dependency direction:

- `gdal` is independent and must not depend on algorithm crates.
- `nufrost-core`, `hants-core`, and `zhu2015-core` may depend on `gdal`.
- `nufrost-cli` depends on `gdal` and all three algorithm crates.

## Config And Data

- Runtime algorithm defaults are JSON files under `config/`.
- Rust parity fixtures live under `tests/fixtures/rust_parity/`.
- Real imagery and generated outputs live under `data/`.

## Testing Layout

- Keep source-level unit tests next to the source they exercise, inside `#[cfg(test)] mod tests`.
- Put integration tests under the repository-level `tests/` directory.
- Put integration-test input and output data under `tests/data/`.
- Keep `tests/data/` structurally aligned with the runtime `data/` root. For example, Sentinel-2 raw fixtures should live under `tests/data/raw/sentinel-2/16-sites/`, cache outputs under `tests/data/cache/...`, and product-like outputs under `tests/data/products/...` or `tests/data/tests/...` depending on whether they are accepted fixtures or candidate test outputs.

## Output And Verification

- Use `cargo check --workspace` for compile verification.
- Use `cargo test -p gdal@0.1.0 --lib` for the local `gdal` crate because the workspace also depends on upstream `gdal`.
- GDAL-linked tests may need `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib` on this machine.
- Python scripts, notebooks, and pytest tests have been removed from the active workflow. Future plotting scripts should be rebuilt around Rust outputs.
