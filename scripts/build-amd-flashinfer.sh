#!/usr/bin/env bash
# Build the upstream JIT/source wheel in the active Python environment.
set -euo pipefail

python - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"
assert torch.__version__.split(".")[:2] == ["2", "9"], torch.__version__
assert torch.version.hip and torch.version.hip.startswith("6.4."), torch.version.hip
PY

python -m pip install 'scikit-build-core>=0.4.3' 'setuptools-scm>=9.2' \
    'cmake>=3.26.1' ninja numpy
python -m pip wheel --no-deps --no-build-isolation --wheel-dir "${1:-dist}" \
    'git+https://github.com/ROCm/flashinfer.git@d981804f78bfcde37984edc2ea4592eaab03b81c'
