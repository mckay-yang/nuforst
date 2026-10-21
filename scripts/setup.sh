#!/usr/bin/env bash
# Install the project-local GDAL library into .conda/gdal.
# Requires conda, mamba, or micromamba. The prefix is gitignored.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
prefix="$root/.conda/gdal"

if [[ -f "$prefix/lib/pkgconfig/gdal.pc" || -f "$prefix/lib64/pkgconfig/gdal.pc" ]]; then
    echo "GDAL already installed at $prefix"
    exit 0
fi

installer=""
for candidate in micromamba mamba conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
        installer="$candidate"
        break
    fi
done
if [[ -z "$installer" ]]; then
    echo "Install conda, mamba, or micromamba, then rerun ./scripts/setup.sh" >&2
    exit 1
fi

"$installer" create -y -p "$prefix" -c conda-forge "libgdal-core=3.10.3" pkg-config
echo "GDAL installed at $prefix"
echo "Run tests with ./scripts/test.sh"
