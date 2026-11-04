"""Public single-volume, single-configuration workflow. No GPU imports here."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "r9a_inr_baseline_decoder_pe6_noanneal_tilek64.yml"
MASKS = ("bernoulli", "block", "stripe_h", "stripe_v", "stripe_cross",
         "checkerboard", "temporal_comp", "stripe_d", "stripe_mix_hv",
         "block_stripe_cross", "dual_random_stripe_cross", "grid_stride3")


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


@contextmanager
def result_lock(folder: Path):
    """The OS releases this advisory lock even if the launcher is killed."""
    path = folder / ".run.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(f"Another launcher is using this output directory: {folder}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_config(args) -> tuple[Path, dict, bytes]:
    path = args.config.expanduser().resolve()
    original = path.read_bytes()
    cfg = yaml.safe_load(original)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    overrides = {"scan_tile_size": args.tile_size, "scan_tile_overlap": args.tile_overlap,
                 "mask_type": args.mask_type, "input_depth": args.window_size,
                 "mask_replace": args.mask_replace, "mask_block_size": args.mask_block_size,
                 "mask_center_only": args.mask_center_only}
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    if args.mask_ratio is not None:
        for key in ("mask_ratio", "train_mask_ratio", "infer_mask_ratio"):
            cfg[key] = [args.mask_ratio, args.mask_ratio]
    if cfg.get("multchannel"):
        raise ValueError("The released presets support single-channel TIFF volumes; select one channel first.")
    tile = cfg.get("scan_tile_size")
    if not isinstance(tile, int) or tile < 0 or tile % 64:
        raise ValueError("Tile size must be 0 (full frame) or a positive multiple of 64.")
    if not 0 <= float(cfg.get("scan_tile_overlap", 0.5)) < 1:
        raise ValueError("Tile overlap must be in [0, 1).")
    window = cfg.get("input_depth")
    if not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise ValueError("Window size must be a positive odd number.")
    if cfg.get("mask_type") not in MASKS:
        raise ValueError(f"Unsupported mask type: {cfg.get('mask_type')}")
    if cfg.get("mask_replace", "zero") not in ("zero", "random", "mean", "neighbor"):
        raise ValueError("mask_replace must be zero, random, mean or neighbor.")
    block_size = cfg.get("mask_block_size", 16)
    if type(block_size) is not int or block_size < 1:
        raise ValueError("Mask block size must be a positive integer.")
    if type(cfg.get("mask_center_only", False)) is not bool:
        raise ValueError("mask_center_only must be a boolean.")
    for key in ("mask_ratio", "train_mask_ratio", "infer_mask_ratio"):
        if key in cfg:
            ratio = cfg[key]
            if not isinstance(ratio, list) or len(ratio) != 2 or not 0 < ratio[0] <= ratio[1] < 100:
                raise ValueError(f"{key} must be [minimum, maximum] percentages between 1 and 99.")
    if int(cfg.get("num_iter", 0)) < 1 or int(cfg.get("tile_infer_k", 0)) < 1:
        raise ValueError("num_iter and tile_infer_k must be positive.")
    if cfg.get("output_tag"):
        raise ValueError("Use --output for the result directory; output_tag must be empty.")
    changed = any(value is not None for value in overrides.values()) or args.mask_ratio is not None
    content = yaml.safe_dump(cfg, sort_keys=False).encode() if changed else original
    return path, cfg, content


def input_identity(path: Path) -> dict:
    import tifffile
    stat = path.stat()
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape, axes, dtype = tuple(series.shape), series.axes, str(series.dtype)
        if len(tif.series) != 1:
            raise ValueError(f"Select a single series before running: {path}")
        if len(shape) not in (2, 3) or axes[-2:] != "YX" or (len(shape) == 3 and axes[0] not in "TZQI"):
            raise ValueError(f"Expected a single-channel [Z,Y,X] or [Y,X] TIFF, got {axes} {shape}.")
        if dtype not in ("uint8", "uint16"):
            raise ValueError(f"Supported input dtypes are uint8 and uint16; got {dtype}. Convert explicitly before running.")
        if min(shape[-2:]) < 32:
            raise ValueError("The released workflow requires each image dimension to be at least 32 pixels.")
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "shape": list(shape), "axes": axes, "dtype": dtype}


def prepare(args) -> tuple[dict, bytes, Path]:
    source, cfg, content = read_config(args)
    input_path = args.input.expanduser().resolve()
    gt_path = args.gt.expanduser().resolve() if args.gt else None
    if gt_path == input_path:
        raise ValueError("--gt must be an independent clean reference; omit it for no-GT data.")
    if args.frame_start < 0 or (args.frame_end is not None and args.frame_end <= args.frame_start):
        raise ValueError("Frame range must satisfy 0 <= start < end (end exclusive).")
    output = (args.output or ROOT / "results" / input_path.stem / source.stem).expanduser().resolve()
    spec = {"schema_version": 1, "framework": "DeepEMD", "config_name": source.stem,
            "config_sha256": hashlib.sha256(content).hexdigest(), "config": cfg,
            "input_path": str(input_path), "gt_path": str(gt_path) if gt_path else None,
            "seed": 2024, "deterministic": True, "frame_start": args.frame_start,
            "frame_end": args.frame_end}
    if not args.dry_run:
        spec["input"] = input_identity(input_path)
        shape = spec["input"]["shape"]
        count = shape[0] if len(shape) == 3 else 1
        if args.frame_start >= count or (args.frame_end is not None and args.frame_end > count):
            raise ValueError(f"Frame range exceeds input length ({count}).")
        if gt_path:
            spec["gt"] = input_identity(gt_path)
            if spec["gt"]["shape"] != shape or spec["gt"]["dtype"] != spec["input"]["dtype"]:
                raise ValueError("GT and input must have the same shape and dtype and be spatially aligned.")
    return spec, content, output


def run(args) -> int:
    from deepemd.scan import command_for, execute
    spec, content, output = prepare(args)
    work = output / "_dry_run" if args.dry_run else output
    work.mkdir(parents=True, exist_ok=True)
    with result_lock(work):
        spec_path = work / "run_spec.json"
        if not args.dry_run:
            if spec_path.exists():
                old = json.loads(spec_path.read_text(encoding="utf-8"))
                if old != spec:
                    raise ValueError("This output belongs to a different input/config/range. Choose a new --output.")
            elif any((work / name).exists() for name in ("denoised_tiff", "status.csv", "generated_configs")):
                raise ValueError("Existing output has no matching run_spec.json. Choose an empty --output.")
        generated = work / "generated_configs"
        generated.mkdir(exist_ok=True)
        config_path = generated / (spec["config_name"] + ".yml")
        config_path.write_bytes(content)
        write_json(spec_path, spec)
        command = command_for(spec, config_path, output if args.dry_run else work)
        print(f"[DeepEMD] config: {spec['config_name']} (one configuration)", flush=True)
        print(f"[DeepEMD] input:  {spec['input_path']}", flush=True)
        print(f"[DeepEMD] output: {output}", flush=True)
        if args.dry_run:
            write_json(work / "plan.json", {"gpu": args.gpu, "command": command, "jobs": 1,
                                           "dry_run": True, "output": str(output)})
            print(f"[dry-run] One job planned; no TIFF decoding or model execution. Plan: {work / 'plan.json'}")
            return 0
        result = execute(spec, command, work, args.gpu)
        if result:
            return result
        from deepemd.completed import create_pdf
        create_pdf(work, frames=args.pdf_frames, filename="comparison.pdf")
        return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeepEMD: run ONE selected config on ONE TIFF volume, then create a comparison PDF.")
    p.add_argument("--input", type=Path, required=True, help="Single-channel uint8/uint16 TIFF; shape is detected automatically.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="One YAML file; default: config/r9a_inr_baseline_decoder_pe6_noanneal_tilek64.yml.")
    p.add_argument("--output", type=Path, help="Result folder (default: results/<input>/<config>).")
    p.add_argument("--gpu", default="0", help="One physical GPU ID, e.g. 4. Use cpu for small tests.")
    p.add_argument("--tile-size", type=int, help="Tile width/height in pixels; multiple of 64, or 0 for full frame.")
    p.add_argument("--tile-overlap", type=float, help="Overlap fraction; defaults to the selected config.")
    p.add_argument("--window-size", type=int, help="Positive odd number of neighboring slices.")
    masks = p.add_argument_group("Volumetric masking")
    masks.add_argument("--mask-type", choices=MASKS, help="Pattern applied across neighboring slices; default: stripe_cross in the INR preset.")
    masks.add_argument("--mask-ratio", type=int, choices=range(1, 100), metavar="1..99", help="Mask-generation percentage for train/infer (not necessarily the realized masked fraction).")
    masks.add_argument("--mask-replace", choices=("zero", "random", "mean", "neighbor"), help="Replacement for masked pixels; default: random in the INR preset.")
    masks.add_argument("--mask-block-size", type=int, help="Positive block width/height for the block pattern; model default: 16.")
    masks.add_argument("--mask-center-only", action=argparse.BooleanOptionalAction, default=None, help="Mask only the center slice, or use --no-mask-center-only to mask the whole window.")
    p.add_argument("--gt", type=Path, help="Optional independent, aligned clean reference.")
    p.add_argument("--frame-start", type=int, default=0, help="First source slice, zero based.")
    p.add_argument("--frame-end", type=int, help="Exclusive end slice; default: full volume.")
    p.add_argument("--pdf-frames", default="first,mid,last", help='PDF slices, e.g. "mid" or "even:5".')
    p.add_argument("--dry-run", action="store_true", help="Write one-job plan under <output>/_dry_run without running the model.")
    return p


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.gpu != "cpu" and not args.gpu.isdigit():
        p.error("--gpu accepts one numeric GPU ID or cpu.")
    try:
        return run(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 1
