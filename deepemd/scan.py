"""Execute a single denoising subprocess and preserve its log/status."""
from __future__ import annotations

import csv
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def command_for(spec: dict, config: Path, output: Path) -> list[str]:
    module = "deepemd.inr_engine" if "inr_variant" in spec["config"] else "deepemd.engine"
    cmd = [sys.executable, "-u", "-m", module, "--noisy_path", spec["input_path"],
           "--gt_path", spec["gt_path"] or spec["input_path"], "--config", str(config),
           "--out_dir", str(output), "--model_name", "DeepEMD_" + spec["config_name"],
           "--seed", str(spec["seed"]), "--deterministic", "1",
           "--tile_size", str(spec["config"]["scan_tile_size"]),
           "--tile_overlap", str(spec["config"]["scan_tile_overlap"]),
           "--frame_start", str(spec["frame_start"])]
    if spec["frame_end"] is not None:
        cmd += ["--frame_end", str(spec["frame_end"])]
    return cmd


def write_status(output: Path, row: dict) -> None:
    tmp = output / "status.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    tmp.replace(output / "status.csv")


def execute(spec: dict, command: list[str], output: Path, gpu: str) -> int:
    from deepemd.completed import validate_result
    log = output / "logs" / (spec["config_name"] + ".log")
    final = output / "denoised_tiff" / "denoised.tif"
    row = {"config_name": spec["config_name"], "status": "running", "gpu": gpu,
           "input_path": spec["input_path"], "denoised_path": str(final),
           "frame_start": spec["frame_start"], "frame_end": spec["frame_end"],
           "log_path": str(log), "exit_code": "", "elapsed_sec": ""}
    if final.exists():
        validate_result(output, spec, "first,mid,last")
        row.update(status="skipped_existing", exit_code=0)
        write_status(output, row)
        print(f"[skip] Validated existing final TIFF: {final}", flush=True)
        return 0
    log.parent.mkdir(exist_ok=True)
    if log.exists():
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log.rename(log.with_name(log.stem + ".previous_" + stamp + ".log"))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="" if gpu == "cpu" else gpu, PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8", DEEPEMD_REQUIRE_CUDA="0" if gpu == "cpu" else "1")
    print(f"[run] GPU={gpu}; training/denoising progress: {log}", flush=True)
    start = time.monotonic()
    write_status(output, row)
    try:
        with log.open("w", encoding="utf-8") as handle:
            handle.write(subprocess.list2cmdline(command) + "\n")
            handle.flush()
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            try:
                code = proc.wait()
            except BaseException:
                proc.terminate()
                proc.wait()
                raise
        if code:
            raise RuntimeError(f"Model process exited with code {code}; see {log}")
        validate_result(output, spec, "first,mid,last")
        row.update(status="ok", exit_code=0)
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
        row.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", exit_code=1)
        print(f"[error] {exc}", file=sys.stderr, flush=True)
    finally:
        row["elapsed_sec"] = round(time.monotonic() - start, 3)
        write_status(output, row)
    return int(row["exit_code"])
