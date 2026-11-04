#!/usr/bin/env python3
"""Build a source-only DeepEMD ZIP from an explicit allowlist."""
import hashlib
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE_FILES = (
    "README.md",
    "requirements.txt",
    "requirements-runtime.txt",
    "run_deepemd.py",
    "visualize_deepemd.py",
    "config/bern40_ts768_ov050_tilek64.yml",
    "config/r9a_inr_baseline_decoder_pe6_noanneal_tilek64.yml",
    "deepemd/__init__.py",
    "deepemd/runner.py",
    "deepemd/scan.py",
    "deepemd/engine.py",
    "deepemd/inr_engine.py",
    "deepemd/completed.py",
    "deepemd/visualization.py",
    "deepemd/models/__init__.py",
    "deepemd/models/Fill_network_arch.py",
    "deepemd/models/Fill_network_inr.py",
    "deepemd/models/Hybrid2D1D.py",
    "deepemd/models/Skip3D.py",
    "deepemd/models/SkipModel.py",
    "deepemd/models/SkipModel_INR.py",
    "deepemd/models/ZAxisAttention.py",
    "deepemd/models/ZAxisPyramid.py",
    "deepemd/utils/__init__.py",
    "deepemd/utils/loss.py",
    "deepemd/utils/metrics.py",
    "deepemd/utils/smooth_util.py",
    "deepemd/utils/util.py",
    "deepemd/utils/util_inr.py",
    "scripts/setup_deepemd_env.sh",
    "scripts/activate_deepemd_env.sh",
    "scripts/check_environment.py",
    "scripts/package_release.py",
    "tests/test_workflow.py",
    "tests/test_env_scripts.py",
    "assets/deepemd_logo.png",
)


def release_inputs(root: Path) -> dict[str, bytes]:
    """Reject missing files, unsafe names, and symlinked source ancestors."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError(f"Release root must not be a symlink: {root}")
    root = root.resolve()
    if len(set(RELEASE_FILES)) != len(RELEASE_FILES):
        raise ValueError("Duplicate release entries.")
    inputs = {}
    for name in sorted(RELEASE_FILES):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe release entry: {name}")
        path = root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"Release input must not use symlinks: {path}")
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Release input must be a regular file inside root: {path}")
        inputs[name] = path.read_bytes()
    return inputs


def build_release(root: Path = ROOT, destination: Path | None = None) -> tuple[Path, str]:
    inputs = release_inputs(root)
    destination = Path(destination) if destination is not None else Path(root) / "dist" / "DeepEMD.zip"
    if destination.is_symlink() or destination.with_suffix(".zip.sha256").is_symlink():
        raise ValueError("Release destination or checksum must not be a symlink.")
    if any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("Release destination parents must not be symlinks.")
    sources = {Path(root).resolve() / name for name in inputs}
    if destination.resolve() in sources or destination.with_suffix(".zip.sha256").resolve() in sources:
        raise ValueError("Release destination must not overwrite source files.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".DeepEMD-", suffix=".zip.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in inputs.items():
                entry = zipfile.ZipInfo("DeepEMD/" + name)
                entry.create_system = 3
                entry.external_attr = (0o100755 if Path(name).suffix == ".sh" else 0o100644) << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, content)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity check failed.")
            if set(archive.namelist()) != {"DeepEMD/" + name for name in inputs}:
                raise ValueError("ZIP contents do not match the release file list.")
            for name, content in inputs.items():
                if archive.read("DeepEMD/" + name) != content:
                    raise ValueError(f"ZIP content differs from source: {name}")
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(destination)
        destination.with_suffix(".zip.sha256").write_text(digest + "  " + destination.name + "\n", encoding="ascii")
        return destination, digest
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    destination, digest = build_release()
    print(f"{destination}\nfiles={len(RELEASE_FILES)} bytes={destination.stat().st_size}\nsha256={digest}")


if __name__ == "__main__":
    main()

