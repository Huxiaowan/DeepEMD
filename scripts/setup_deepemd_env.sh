#!/usr/bin/env bash
# Linux/Conda bootstrap. Run with bash; activate the parent shell separately.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_DIR=${ENV_DIR:-$REPO/.venv_py310}
[[ "$ENV_DIR" = /* ]] || ENV_DIR="$REPO/$ENV_DIR"
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
# Official matching wheel pair; existing working installations are kept in auto mode.
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.25.0}
INSTALL_TORCH=${INSTALL_TORCH:-auto}
REQUIRE_CUDA=${REQUIRE_CUDA:-1}

fail() { printf '[error] %s\n' "$*" >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || fail 'This bootstrap supports Linux. Use an activated Conda environment on Windows.'
[[ "$PYTHON_VERSION" =~ ^3\.([0-9]+)$ ]] && (( BASH_REMATCH[1] >= 10 )) || fail 'PYTHON_VERSION must be 3.10 or later, e.g. 3.10.'
[[ "$INSTALL_TORCH" =~ ^(auto|0|1)$ ]] || fail 'INSTALL_TORCH must be auto, 0, or 1.'
[[ "$REQUIRE_CUDA" =~ ^[01]$ ]] || fail 'REQUIRE_CUDA must be 0 or 1.'
command -v conda >/dev/null 2>&1 || fail 'Conda is not available. Load Miniconda/Anaconda first.'

printf '[info] project: %s\n[info] environment: %s\n' "$REPO" "$ENV_DIR"
conda_hook="$(conda shell.bash hook)"
eval "$conda_hook"

if [[ -e "$ENV_DIR" ]]; then
  [[ -f "$ENV_DIR/conda-meta/history" && -x "$ENV_DIR/bin/python" ]] || fail "Existing directory is not a Conda environment: $ENV_DIR"
  printf '[info] reusing Conda environment\n'
else
  conda create --prefix "$ENV_DIR" "python=$PYTHON_VERSION" pip -y
fi
conda activate "$ENV_DIR"
PYTHON="$CONDA_PREFIX/bin/python"
[[ -x "$PYTHON" ]] || fail 'The activated environment has no Python executable.'
"$PYTHON" - "$PYTHON_VERSION" <<'PY'
import sys
expected = tuple(map(int, sys.argv[1].split('.')))
if sys.version_info[:2] != expected:
    raise SystemExit(f"Python version mismatch: expected {expected}, got {sys.version_info[:2]}. Use a different ENV_DIR or set PYTHON_VERSION explicitly.")
print('[info] Python:', sys.version.split()[0], sys.executable)
PY

# Install missing C/C++ runtime libraries while keeping existing installations.
if [[ ! -e "$CONDA_PREFIX/lib/libstdc++.so.6" || ! -e "$CONDA_PREFIX/lib/libgcc_s.so.1" ]]; then
  conda install --prefix "$CONDA_PREFIX" -c conda-forge -y libstdcxx-ng libgcc-ng
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

torch_works() {
  "$PYTHON" - <<'PY'
import torch
import torchvision
if int(torch.__version__.split('.')[0]) < 2:
    raise SystemExit('PyTorch >= 2 is required.')
print('[info] existing torch:', torch.__version__, '| torchvision:', torchvision.__version__)
PY
}

install_torch=0
case "$INSTALL_TORCH" in
  1) install_torch=1 ;;
  auto) if ! torch_works; then install_torch=1; fi ;;
  0) printf '[info] PyTorch installation disabled (INSTALL_TORCH=0)\n' ;;
esac
if [[ "$install_torch" == 1 ]]; then
  printf '[info] installing torch=%s torchvision=%s from %s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" "$TORCH_INDEX_URL"
  "$PYTHON" -m pip install --upgrade --force-reinstall \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"
fi

# Runtime dependencies are separate so INSTALL_TORCH=0 really skips PyTorch.
"$PYTHON" -m pip install -r "$REPO/requirements-runtime.txt"
check_args=(--project-imports)
[[ "$REQUIRE_CUDA" == 0 ]] || check_args+=(--require-cuda)
"$PYTHON" "$REPO/scripts/check_environment.py" "${check_args[@]}"
"$PYTHON" -m pip freeze > "$CONDA_PREFIX/deepemd-pip-freeze.txt"

printf '[done] DeepEMD environment is ready. Installed versions: %s\n' "$CONDA_PREFIX/deepemd-pip-freeze.txt"
printf '[next] Activate it in your current terminal:\n  ENV_DIR=%q source %q\n' "$CONDA_PREFIX" "$REPO/scripts/activate_deepemd_env.sh"
