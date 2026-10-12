#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  else
    printf '%s\n' 'Set CUDA_HOME to the CUDA 11.8 toolkit directory before building.' >&2
    exit 1
  fi
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-2}"
"${CUDA_HOME}/bin/nvcc" --version
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)'

# Cube-Splat adds rel_R and maps per-face pose gradients to the shared pose.
# Always build these bundled sources instead of installing MonoGS's rasterizer.
python -m pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall \
  ./submodules/simple-knn ./submodules/diff-gaussian-rasterization
