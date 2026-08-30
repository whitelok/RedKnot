#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${REDKNOT_BOOTSTRAP_PYTHON:-python3.11}"
VENV="${REDKNOT_VENV:-${REPO_ROOT}/.venv_tf5}"
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo "missing environment: ${VENV}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"RedKnot requires CPython 3.11.x, got {sys.version}")
PY
  "${PYTHON_BIN}" -m venv "${VENV}"
fi

if [[ "${CHECK_ONLY}" == "0" ]]; then
  "${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV}/bin/python" -m pip install \
    -r "${SCRIPT_DIR}/requirements-deepseek-v4-flash.txt"
  "${VENV}/bin/python" -m pip install --no-deps -e "${REPO_ROOT}/python"
fi

PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  "${VENV}/bin/python" - <<'PY'
import importlib.metadata as metadata
import json
import platform
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"expected CPython 3.11.x, got {sys.version}")

import deep_gemm
import flash_mla
import sgl_kernel
import torch

required_flash_mla = (
    "get_mla_metadata",
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
)
missing = [name for name in required_flash_mla if not hasattr(flash_mla, name)]
if missing:
    raise SystemExit(f"FlashMLA API mismatch: missing {missing}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected PyTorch CUDA 12.8, got {torch.version.cuda}")

expected = {
    "torch": "2.9.1",
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "safetensors": "0.8.0",
    "flashinfer-python": "0.5.3",
    "sgl-kernel": "0.3.20",
    "triton": "3.5.1",
}
observed = {name: metadata.version(name) for name in expected}
mismatch = {
    name: {"expected": expected[name], "observed": observed[name]}
    for name in expected
    if observed[name] != expected[name]
}
if mismatch:
    raise SystemExit(f"dependency version mismatch: {mismatch}")

print(json.dumps({
    "status": "ready",
    "python": platform.python_version(),
    "torch_cuda": torch.version.cuda,
    "packages": observed,
    "flash_mla": metadata.version("flash-mla"),
    "deep_gemm": deep_gemm.__file__,
    "sgl_kernel": sgl_kernel.__file__,
}, sort_keys=True))
PY
