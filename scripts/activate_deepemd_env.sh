#!/usr/bin/env bash
# Must be sourced. Do not change the caller's errexit/nounset/pipefail options.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '[error] Use: source scripts/activate_deepemd_env.sh\n' >&2
  exit 1
fi

_deepemd_activate_env() {
  local script_dir repo env_path conda_hook
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
  repo="$(cd -- "$script_dir/.." && pwd)" || return 1
  env_path=${ENV_DIR:-$repo/.venv_py310}
  [[ "$env_path" = /* ]] || env_path="$repo/$env_path"
  if ! command -v conda >/dev/null 2>&1; then
    printf '[error] Conda is not available. Load Miniconda/Anaconda first.\n' >&2
    return 1
  fi
  if [[ ! -f "$env_path/conda-meta/history" || ! -x "$env_path/bin/python" ]]; then
    printf '[error] Conda environment not found: %s\n' "$env_path" >&2
    printf '[next] Run: bash %q\n' "$repo/scripts/setup_deepemd_env.sh" >&2
    return 1
  fi
  conda_hook="$(conda shell.bash hook)" || return 1
  eval "$conda_hook" || return 1
  conda activate "$env_path" || return 1
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$CONDA_PREFIX/lib:"*) ;;
    *) export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
  printf '[info] activated DeepEMD env: %s\n' "$CONDA_PREFIX"
}

if _deepemd_activate_env; then
  unset -f _deepemd_activate_env
  return 0
else
  unset -f _deepemd_activate_env
  return 1
fi
