#!/usr/bin/env bash
# Run algorithm tests without GDAL, then raster and CLI tests when GDAL is installed.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

gdal_prefix="$root/.conda/gdal"
if [[ -d "$gdal_prefix/bin" ]]; then
    export PATH="$gdal_prefix/bin${PATH:+:$PATH}"
fi
for pkg_dir in "$gdal_prefix/lib/pkgconfig" "$gdal_prefix/lib64/pkgconfig"; do
    if [[ -d "$pkg_dir" ]]; then
        export PKG_CONFIG_PATH="$pkg_dir${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    fi
done

echo "== algorithm tests =="
cargo test -p nufrost-core -p hants-core -p zhu2015-core

if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists gdal; then
    echo "GDAL not found via pkg-config; skipped gdal and nufrost-cli tests."
    echo "Install the project-local library with ./scripts/setup.sh"
    exit 0
fi

libdir="$(pkg-config --variable=libdir gdal)"
if [[ -n "$libdir" ]]; then
    case "$(uname -s)" in
        Darwin)
            export DYLD_LIBRARY_PATH="${libdir}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
            ;;
        Linux)
            export LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
    esac
fi

echo "== raster and CLI tests =="
cargo test -p gdal@0.1.0 --lib
cargo test -p gdal@0.1.0 --test time_series_fixtures
cargo test -p nufrost-cli --lib
cargo test -p nufrost-cli --test full_scene_fixtures
