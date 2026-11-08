#!/usr/bin/env bash
# Fetches the ROCm build of CTranslate2 (published only inside a zip on GitHub releases,
# so pip/poetry can't install it from a URL) and installs everything with poetry.
set -euo pipefail
cd "$(dirname "$0")"

CT2_VERSION=4.8.3
WHEEL="ctranslate2-${CT2_VERSION}-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

if [[ ! -f "vendor/$WHEEL" ]]; then
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    curl -fL -o "$tmp/wheels.zip" \
        "https://github.com/OpenNMT/CTranslate2/releases/download/v${CT2_VERSION}/rocm-python-wheels-Linux.zip"
    unzip -q -j "$tmp/wheels.zip" "*/$WHEEL" -d vendor/
fi

poetry sync
