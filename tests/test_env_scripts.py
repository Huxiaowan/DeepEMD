"""Exercise Bash bootstrap control flow with fake Conda/pip; never install packages."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

FAKE_CONDA = r'''#!/usr/bin/env bash
printf 'conda' >> "$CALL_LOG"
printf ' <%s>' "$@" >> "$CALL_LOG"
printf '\n' >> "$CALL_LOG"
if [[ "$1 $2" == 'shell.bash hook' ]]; then
  [[ "${FAIL_HOOK:-0}" == 0 ]] || exit 9
  cat <<'HOOK'
conda() {
  if [[ "$1" == activate ]]; then
    [[ "${FAIL_ACTIVATE:-0}" == 0 ]] || return 8
    export CONDA_PREFIX="$2"
    export PATH="$2/bin:$PATH"
  else
    command conda "$@"
  fi
}
HOOK
elif [[ "$1" == create ]]; then
  target="$3"
  mkdir -p "$target/bin" "$target/conda-meta" "$target/lib"
  touch "$target/conda-meta/history" "$target/lib/libstdc++.so.6" "$target/lib/libgcc_s.so.1"
  cp "$FAKE_PYTHON" "$target/bin/python"
  chmod +x "$target/bin/python"
fi
'''

FAKE_PYTHON = r'''#!/usr/bin/env bash
printf 'python' >> "$CALL_LOG"
printf ' <%s>' "$@" >> "$CALL_LOG"
printf '\n' >> "$CALL_LOG"
if [[ "$1" == - ]]; then
  script="$(cat)"
  if [[ "$script" == *'import torch'* && "$script" != *'import sys'* ]]; then
    [[ "${TORCH_PRESENT:-0}" == 1 ]]
    exit $?
  fi
elif [[ "$1 $2 $3" == '-m pip freeze' ]]; then
  printf 'torch==2.10.0\n'
elif [[ "$1" == *check_environment.py ]]; then
  exit "${CHECK_EXIT:-0}"
fi
'''


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("bash"), "Linux Bash test; run under WSL or on the server")
class EnvironmentScriptsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="deepemd env test ")
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.repo = base / "project with spaces"
        (self.repo / "scripts").mkdir(parents=True)
        for name in ("setup_deepemd_env.sh", "activate_deepemd_env.sh"):
            shutil.copy2(ROOT / "scripts" / name, self.repo / "scripts" / name)
        (self.repo / "requirements-runtime.txt").write_text("numpy\n")
        self.fake_bin = base / "fakebin"
        self.fake_bin.mkdir()
        self.log = base / "calls.log"
        fake_python = self.fake_bin / "fake-python"
        fake_python.write_text(FAKE_PYTHON)
        (self.fake_bin / "conda").write_text(FAKE_CONDA)
        for path in self.fake_bin.iterdir():
            path.chmod(0o755)
        self.env = os.environ.copy()
        for key in ("ENV_DIR", "PYTHON_VERSION", "INSTALL_TORCH", "REQUIRE_CUDA", "CONDA_PREFIX",
                    "TORCH_VERSION", "TORCHVISION_VERSION", "TORCH_INDEX_URL", "BASH_ENV"):
            self.env.pop(key, None)
        self.env.update(PATH=str(self.fake_bin) + ":" + self.env["PATH"], CALL_LOG=str(self.log), FAKE_PYTHON=str(fake_python))

    def setup(self, **overrides):
        return subprocess.run(["bash", str(self.repo / "scripts/setup_deepemd_env.sh")],
                              cwd=self.fake_bin, env=dict(self.env, **overrides), text=True, capture_output=True)

    def test_syntax(self):
        for name in ("setup_deepemd_env.sh", "activate_deepemd_env.sh"):
            subprocess.run(["bash", "-n", str(self.repo / "scripts" / name)], check=True)

    def test_fresh_setup_creates_installs_and_checks(self):
        proc = self.setup()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        log = self.log.read_text()
        self.assertIn("conda <create> <--prefix>", log)
        self.assertIn("<torch==2.10.0> <torchvision==0.25.0>", log)
        self.assertIn("<--require-cuda>", log)
        self.assertIn("requirements-runtime.txt>", log)
        self.assertTrue((self.repo / ".venv_py310/deepemd-pip-freeze.txt").is_file())

    def test_existing_working_torch_is_kept(self):
        self.assertEqual(self.setup().returncode, 0)
        self.log.write_text("")
        proc = self.setup(TORCH_PRESENT="1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        log = self.log.read_text()
        self.assertNotIn("conda <create>", log)
        self.assertNotIn("<torch==", log)

    def test_custom_prefix_skip_torch_and_cpu_check(self):
        custom = str(self.repo.parent / "custom environment")
        proc = self.setup(ENV_DIR=custom, INSTALL_TORCH="0", REQUIRE_CUDA="0")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        log = self.log.read_text()
        self.assertIn("<" + custom + ">", log)
        self.assertNotIn("<torch==", log)
        self.assertNotIn("<--require-cuda>", log)

    def test_failed_verification_has_no_success_receipt(self):
        proc = self.setup(CHECK_EXIT="1")
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("[done]", proc.stdout)
        self.assertFalse((self.repo / ".venv_py310/deepemd-pip-freeze.txt").exists())

    def test_activation_persists_without_changing_shell_options(self):
        self.assertEqual(self.setup().returncode, 0)
        script = self.repo / "scripts/activate_deepemd_env.sh"
        command = r'''before="$(set +o)"; source "$1" || exit 1; [[ "$(set +o)" == "$before" ]] || exit 2
[[ "$CONDA_PREFIX" == "$2" ]] || exit 3
[[ "$(command -v python)" == "$2/bin/python" ]] || exit 4
first="$LD_LIBRARY_PATH"; source "$1" || exit 5; [[ "$LD_LIBRARY_PATH" == "$first" ]] || exit 6'''
        proc = subprocess.run(["bash", "-c", command, "bash", str(script), str(self.repo / ".venv_py310")],
                              env=self.env, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_running_activation_without_source_is_rejected(self):
        proc = subprocess.run(["bash", str(self.repo / "scripts/activate_deepemd_env.sh")],
                              env=self.env, text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("source", proc.stderr)

    def test_invalid_existing_directory_is_not_modified(self):
        target = self.repo / ".venv_py310"
        target.mkdir()
        (target / "keep.txt").write_text("user data")
        proc = self.setup()
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(list(target.iterdir()), [target / "keep.txt"])

    def test_conda_hook_failure_stops_before_creating_environment(self):
        proc = self.setup(FAIL_HOOK="1")
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse((self.repo / ".venv_py310").exists())


if __name__ == "__main__":
    unittest.main()
