"""Read-only TIFF inspection and PDF rendering for one DeepEMD run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from deepemd.visualization import PdfPages, TiffStack, page_compare, page_table, parse_frames, read_csv


def signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def validate_result(folder: Path, spec: dict, frame_spec: str):
    """Check TIFF structure and selected pixels, not every pixel of a large volume."""
    import tifffile
    source = Path(spec["input_path"])
    result = folder / "denoised_tiff" / "denoised.tif"
    if not result.is_file():
        raise ValueError(f"No completed denoised.tif at {folder}; partial TIFFs are excluded.")
    signatures = {path: signature(path) for path in (source, result)}
    identity = spec.get("input")
    if identity and signatures[source] != (identity["bytes"], identity["mtime_ns"]):
        raise ValueError("Input TIFF has changed since this run; refusing to compare different data.")
    with TiffStack(source) as raw, TiffStack(result) as den:
        start = int(spec.get("frame_start") or 0)
        end = spec.get("frame_end") or raw.n_frames
        if not 0 <= start < end <= raw.n_frames:
            raise ValueError("Invalid source frame range.")
        if den.n_frames != end - start:
            raise ValueError(f"Incomplete TIFF: {den.n_frames} frames, expected {end - start}.")
        frames = parse_frames(frame_spec, raw.n_frames, start, end, 20260612)
        for index in sorted({start, end - 1, *frames}):
            if raw.read(index).shape != den.read(index - start).shape:
                raise ValueError(f"Input/output shape mismatch at frame {index}.")
    with tifffile.TiffFile(result) as tif:
        dtype = str(tif.pages[0].dtype)
        if identity and dtype != identity["dtype"]:
            raise ValueError(f"Output dtype {dtype} differs from input {identity['dtype']}.")
    for path, before in signatures.items():
        if signature(path) != before:
            raise ValueError(f"TIFF changed while being checked: {path}")
    return source, result, start, end, frames, signatures


def create_pdf(folder: Path, frames="first,mid,last", crop_size=160, num_crops=3,
               crop_boxes=None, filename="completed_comparison.pdf") -> Path:
    folder = folder.resolve()
    spec = json.loads((folder / "run_spec.json").read_text(encoding="utf-8"))
    source, result, start, end, indices, signatures = validate_result(folder, spec, frames)
    status = read_csv(folder / "status.csv")
    row = {"config_name": spec["config_name"], "status": "file_ok",
           "gpu": status[0].get("gpu", "") if status else "",
           "input_path": str(source), "denoised_path": str(result),
           "frame_start": str(start), "frame_end": str(end)}
    cfg = spec["config"]
    manifest = dict(cfg, config_name=spec["config_name"],
                    train_mask_ratio=cfg.get("train_mask_ratio", cfg.get("mask_ratio")),
                    note="Visual comparison; no clean-reference quality score.")
    title = "DeepEMD | " + source.stem
    if len(title) > 90:
        title = title[:87] + "..."
    pdf_path = folder / filename
    with tempfile.NamedTemporaryFile(prefix=".comparison_", suffix=".pdf", dir=folder, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with PdfPages(temporary) as pdf:
            for index in indices:
                page_compare(pdf, source, [row], index, 1, 1, crop_size, num_crops,
                             20260612, crop_boxes, title)
            page_table(pdf, [row], [manifest], "DeepEMD")
        for path, before in signatures.items():
            if signature(path) != before:
                raise ValueError(f"TIFF changed during rendering; previous PDF preserved: {path}")
        temporary.replace(pdf_path)
    finally:
        temporary.unlink(missing_ok=True)
    from deepemd.runner import write_json
    write_json(pdf_path.with_suffix(".json"), {
        "pdf": str(pdf_path), "input": str(source), "denoised": str(result),
        "config": spec["config_name"], "frames": indices, "source_frame_range": [start, end],
        "validation": "Matching frame count/dtype; readable endpoints and displayed frames; matching spatial shapes; stable size/mtime.",
        "display": "Shared 0.5/99.5 percentile grayscale limits per page; identical crop coordinates.",
        "quality_metrics": "PDF is a visual comparison, not a clean-reference quality evaluation."
    })
    print(f"[done] PDF: {pdf_path} | frames={indices} | pages={len(indices) + 1}", flush=True)
    return pdf_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DeepEMD: visualize one completed run without training or changing its status/logs.")
    p.add_argument("--results-root", required=True, type=Path, help="One run folder containing run_spec.json.")
    p.add_argument("--frames", default="first,mid,last", help='Examples: "mid", "first,mid,last", "even:5".')
    p.add_argument("--crop-size", type=int, default=160)
    p.add_argument("--num-crops", type=int, default=3)
    p.add_argument("--crop-boxes", help="Optional: x0,y0,x1,y1;x0,y0,x1,y1")
    args = p.parse_args(argv)
    if args.crop_size <= 0 or not 0 <= args.num_crops <= 5:
        p.error("crop-size must be positive and num-crops must be between 0 and 5")
    try:
        create_pdf(args.results_root, args.frames, args.crop_size, args.num_crops, args.crop_boxes)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
