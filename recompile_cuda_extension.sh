#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_DIR="$ROOT_DIR/hierslam-diff-gaussian-rasterization-w-depth"
PYTHON="${OHSLAM_PYTHON:-python}"
export PATH="$(dirname "$PYTHON"):/usr/local/cuda/bin:$PATH"
cd "$EXT_DIR"
grep -Eq '^#define NUM_SEMANTIC 21 ' cuda_rasterizer/config.h
"$PYTHON" setup.py build_ext --inplace --force
echo "Built the in-project 21-channel extension; no global environment reinstall performed."
