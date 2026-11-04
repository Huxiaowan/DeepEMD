#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

try:
    import tifffile as tiff
except ImportError:  # pragma: no cover
    tiff = None


CROP_COLORS = ["red", "yellow", "cyan", "lime", "magenta"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def int_or_none(value: str | int | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _infer_time_axis(shape: tuple[int, ...], axes: str | None = None) -> int | None:
    if len(shape) <= 2:
        return None
    if axes:
        axes = axes.upper()
        for label in ("T", "Z", "Q"):
            pos = axes.find(label)
            if 0 <= pos < len(shape) and shape[pos] > 1:
                return pos
    if len(shape) == 3:
        if shape[-1] <= 4 and shape[0] > 16 and shape[1] > 16:
            return None
        return 0 if shape[0] > 1 else None
    if len(shape) == 4:
        if shape[0] > 1 and shape[1] <= 4:
            return 0
        if shape[0] <= 4 and shape[1] > 1:
            return 1
        if shape[-1] <= 4 and shape[0] > 1:
            return 0
        return 0 if shape[0] > 1 else None
    return 0 if shape[0] > 1 else None


def _to_2d(arr: np.ndarray, axes: str | None = None) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        axes = axes.upper() if axes else ""
        if axes and "C" in axes:
            c_axis = axes.find("C")
            if 0 <= c_axis < arr.ndim:
                return np.take(arr, 0, axis=c_axis)
        if arr.shape[-1] <= 4:
            return arr[..., 0]
        if arr.shape[0] <= 4:
            return arr[0]
        return arr[0]
    while arr.ndim > 2:
        arr = np.take(arr, 0, axis=0)
        arr = np.squeeze(arr)
    return arr


class TiffStack:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.pil = None
        self.tf = None
        self.axes = None
        self.shape: tuple[int, ...] | None = None
        self.time_axis: int | None = None
        self.n_frames = 1
        self.mapped = None
        self.tried_memmap = False
        if tiff is not None:
            try:
                self.tf = tiff.TiffFile(str(self.path))
                series = self.tf.series[0]
                self.shape = tuple(series.shape)
                self.axes = getattr(series, "axes", None)
                self.time_axis = _infer_time_axis(self.shape, self.axes)
                if self.time_axis is not None:
                    self.n_frames = int(self.shape[self.time_axis])
                elif len(self.tf.pages) > 1:
                    self.n_frames = len(self.tf.pages)
                return
            except Exception:
                if self.tf is not None:
                    self.tf.close()
                self.tf = None
        self.pil = Image.open(self.path)
        self.n_frames = int(getattr(self.pil, "n_frames", 1))

    def read(self, index: int) -> np.ndarray:
        index = max(0, min(int(index), self.n_frames - 1))
        if self.tf is not None:
            try:
                if len(self.tf.pages) == self.n_frames and self.n_frames > 1:
                    page_axes = None
                    if self.axes and self.time_axis == 0:
                        page_axes = self.axes[1:]
                    return _to_2d(self.tf.pages[index].asarray(), page_axes)
                # The glioma inputs can store the entire Z stack in one TIFF
                # page. Map that stack and copy only the selected frame, rather
                # than materializing tens of GB for each PDF panel.
                if self.time_axis is not None:
                    if not self.tried_memmap:
                        self.tried_memmap = True
                        try:
                            self.mapped = tiff.memmap(str(self.path), mode="r", series=0)
                        except (ValueError, OSError):
                            self.mapped = None
                    if self.mapped is not None:
                        axes = self.axes
                        if axes and len(axes) == self.mapped.ndim:
                            axes = axes[: self.time_axis] + axes[self.time_axis + 1 :]
                        return _to_2d(np.take(self.mapped, index, axis=self.time_axis), axes)
                arr = self.tf.asarray(key=index)
                return _to_2d(arr, self.axes)
            except Exception:
                arr = self.tf.asarray()
                axes = self.axes
                if self.time_axis is not None:
                    arr = np.take(arr, index, axis=self.time_axis)
                    if axes and len(axes) == len(self.shape or ()):
                        axes = axes[: self.time_axis] + axes[self.time_axis + 1 :]
                return _to_2d(arr, axes)
        assert self.pil is not None
        self.pil.seek(index)
        return _to_2d(np.asarray(self.pil.copy()))

    def close(self) -> None:
        self.mapped = None
        if self.tf is not None:
            self.tf.close()
        if self.pil is not None:
            self.pil.close()

    def __enter__(self) -> "TiffStack":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def frame_count(path: Path) -> int:
    with TiffStack(path) as stack:
        return stack.n_frames


def to_gray_float(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        else:
            arr = arr[..., 0]
    return arr.astype(np.float32, copy=False)


def align_images(images: list[np.ndarray]) -> list[np.ndarray]:
    h = min(img.shape[-2] for img in images)
    w = min(img.shape[-1] for img in images)
    return [img[:h, :w] for img in images]


def display_limits(images: list[np.ndarray], low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    samples = []
    for img in images:
        flat = img.astype(np.float32, copy=False).ravel()
        step = max(1, flat.size // 200000)
        samples.append(flat[::step])
    stack = np.concatenate(samples) if samples else np.array([0.0, 1.0], dtype=np.float32)
    lo = float(np.percentile(stack, low))
    hi = float(np.percentile(stack, high))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return lo, hi


def scale_to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = (arr.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def source_frame_range(rows: list[dict[str, str]], input_n: int) -> tuple[int, int]:
    starts = [int_or_none(row.get("frame_start")) for row in rows]
    ends = [int_or_none(row.get("frame_end")) for row in rows]
    start = max([0, *[x for x in starts if x is not None]])
    end_candidates = [input_n, *[x for x in ends if x is not None]]
    end = min(end_candidates)
    if end <= start:
        return 0, input_n
    return start, end


def parse_frames(spec: str, input_n: int, start: int, end: int, seed: int) -> list[int]:
    start = max(0, min(start, input_n))
    end = max(start + 1, min(end, input_n))
    if not spec:
        spec = "mid"
    spec = spec.strip()
    if spec.lower() in {"all", "*"}:
        return list(range(start, end))

    frames: list[int] = []
    for part in spec.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"first", "start"}:
            frames.append(start)
        elif token == "mid":
            frames.append((start + end - 1) // 2)
        elif token == "last":
            frames.append(end - 1)
        elif token.startswith("random:"):
            count = max(1, int(token.split(":", 1)[1]))
            rng = np.random.default_rng(seed)
            count = min(count, end - start)
            frames.extend(int(x) for x in rng.choice(np.arange(start, end), size=count, replace=False))
        elif token.startswith("even:"):
            count = max(1, int(token.split(":", 1)[1]))
            frames.extend(int(round(x)) for x in np.linspace(start, end - 1, min(count, end - start)))
        elif "-" in token:
            if ":" in token:
                span, step_text = token.split(":", 1)
                step = max(1, int(step_text))
            else:
                span, step = token, 1
            a_text, b_text = span.split("-", 1)
            a = start if a_text in {"", "start"} else int(a_text)
            b = end - 1 if b_text in {"", "last"} else int(b_text)
            frames.extend(range(a, b + 1, step))
        else:
            frames.append(int(token))

    frames = sorted(set(max(0, min(int(x), input_n - 1)) for x in frames))
    return [x for x in frames if start <= x < end] or [(start + end - 1) // 2]


def choose_crops(
    h: int,
    w: int,
    crop_size: int,
    num_crops: int,
    seed: int,
    frame_idx: int,
    explicit: str | None,
) -> list[tuple[int, int, int, int]]:
    if explicit:
        boxes = []
        for part in explicit.split(";"):
            vals = [int(v.strip()) for v in part.split(",") if v.strip()]
            if len(vals) != 4:
                raise SystemExit(f"Bad crop box: {part}. Use x0,y0,x1,y1;...")
            x0, y0, x1, y1 = vals
            boxes.append((max(0, x0), max(0, y0), min(w, x1), min(h, y1)))
        return [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    if crop_size <= 0 or num_crops <= 0:
        return []
    cw = min(crop_size, w)
    ch = min(crop_size, h)
    if cw <= 0 or ch <= 0:
        return []
    if num_crops == 1:
        x0 = max(0, (w - cw) // 2)
        y0 = max(0, (h - ch) // 2)
        return [(x0, y0, x0 + cw, y0 + ch)]

    rng = np.random.default_rng(seed + frame_idx * 1009)
    boxes: list[tuple[int, int, int, int]] = []
    margin = max(8, min(cw, ch) // 12)
    attempts = 0
    while len(boxes) < num_crops and attempts < 3000:
        attempts += 1
        x0 = int(rng.integers(0, max(1, w - cw + 1)))
        y0 = int(rng.integers(0, max(1, h - ch + 1)))
        box = (x0, y0, x0 + cw, y0 + ch)
        overlaps = any(
            not (
                box[2] + margin < old[0]
                or box[0] - margin > old[2]
                or box[3] + margin < old[1]
                or box[1] - margin > old[3]
            )
            for old in boxes
        )
        if not overlaps:
            boxes.append(box)

    while len(boxes) < num_crops:
        frac = (len(boxes) + 1) / (num_crops + 1)
        x0 = int(max(0, min(w - cw, round(frac * (w - cw)))))
        y0 = int(max(0, min(h - ch, round(frac * (h - ch)))))
        boxes.append((x0, y0, x0 + cw, y0 + ch))
    return boxes


def add_border(img: np.ndarray, color: str, width: int = 5) -> np.ndarray:
    if img.ndim == 2:
        rgb = np.repeat(img[..., None], 3, axis=2)
    else:
        rgb = img
    color_rgb = {
        "red": (255, 0, 0),
        "yellow": (255, 220, 0),
        "cyan": (0, 220, 255),
        "lime": (0, 255, 70),
        "magenta": (255, 0, 220),
    }.get(color, (255, 255, 255))
    out = np.zeros((rgb.shape[0] + 2 * width, rgb.shape[1] + 2 * width, 3), dtype=np.uint8)
    out[:, :] = color_rgb
    out[width : width + rgb.shape[0], width : width + rgb.shape[1]] = rgb
    return out


def title_text(name: str) -> str:
    if name == "Input":
        return name
    name = name.replace("_tilek64", "")
    name = name.replace("r9a_inr_baseline_decoder_pe6_noanneal", "DeepEMD INR")
    return textwrap.fill(name.replace("_", " "), width=18)


def order_rows(status_rows: list[dict[str, str]], manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    good = [
        row
        for row in status_rows
        if row.get("status") in {"ok", "skipped_existing"}
        and row.get("denoised_path")
        and Path(row["denoised_path"]).exists()
    ]
    manifest_order = {row.get("config_name", ""): idx for idx, row in enumerate(manifest_rows)}
    return sorted(good, key=lambda row: manifest_order.get(row.get("config_name", ""), 10_000))


def load_panels(input_path: Path, rows: list[dict[str, str]], source_frame: int) -> tuple[list[str], list[np.ndarray]]:
    names = ["Input"]
    images: list[np.ndarray] = []
    with TiffStack(input_path) as input_stack:
        images.append(to_gray_float(input_stack.read(source_frame)))

    for row in rows:
        den_path = Path(row["denoised_path"])
        frame_start = int_or_none(row.get("frame_start")) or 0
        den_frame = max(0, source_frame - frame_start)
        with TiffStack(den_path) as den_stack:
            den_frame = min(den_frame, den_stack.n_frames - 1)
            images.append(to_gray_float(den_stack.read(den_frame)))
        names.append(row["config_name"])
    return names, align_images(images)


def page_compare(
    pdf: PdfPages,
    input_path: Path,
    rows: list[dict[str, str]],
    source_frame: int,
    page_index: int,
    page_count: int,
    crop_size: int,
    num_crops: int,
    crop_seed: int,
    crop_boxes: str | None,
    title: str,
) -> None:
    names, images = load_panels(input_path, rows, source_frame)
    h, w = images[0].shape[-2:]
    boxes = choose_crops(h, w, crop_size, num_crops, crop_seed, source_frame, crop_boxes)
    lo, hi = display_limits(images)
    disp = [scale_to_uint8(img, lo, hi) for img in images]

    ncols = len(disp)
    nrows = 1 + len(boxes)
    fig_w = max(6.0, 2.75 * ncols)
    fig_h = max(4.0, 3.0 + 1.95 * len(boxes))
    fig = plt.figure(figsize=(fig_w, fig_h))
    suffix = f" | part {page_index}/{page_count}" if page_count > 1 else ""
    fig.suptitle(f"{title} | frame {source_frame}{suffix}", fontsize=14, y=0.99)
    gs = fig.add_gridspec(nrows, ncols, wspace=0.025, hspace=0.06)

    for col, (name, img) in enumerate(zip(names, disp)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        ax.set_title(title_text(name), fontsize=8, pad=6)
        if col == 0:
            for box, color in zip(boxes, CROP_COLORS):
                x0, y0, x1, y1 = box
                rect = patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    linewidth=1.7,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

        for row_idx, (box, color) in enumerate(zip(boxes, CROP_COLORS), start=1):
            x0, y0, x1, y1 = box
            crop = add_border(img[y0:y1, x0:x1], color)
            ax_crop = fig.add_subplot(gs[row_idx, col])
            ax_crop.imshow(crop, cmap="gray", vmin=0, vmax=255)
            ax_crop.axis("off")

    pdf.savefig(fig, bbox_inches="tight", dpi=220)
    plt.close(fig)


def page_table(pdf: PdfPages, rows: list[dict[str, str]], manifest_rows: list[dict[str, str]], title: str) -> None:
    manifest_by_name = {row.get("config_name", ""): row for row in manifest_rows}
    table_rows = []
    for row in rows:
        m = manifest_by_name.get(row.get("config_name", ""), {})
        table_rows.append(
            [
                textwrap.fill(row.get("config_name", ""), width=60),
                row.get("status", ""),
                row.get("gpu", ""),
                str(m.get("input_depth", "")),
                str(m.get("num_iter", "")),
                str(m.get("mask_type", "")),
                str(m.get("train_mask_ratio", "")),
                str(m.get("scan_tile_size", row.get("tile_size", ""))),
                str(m.get("scan_tile_overlap", row.get("tile_overlap", ""))),
                str(m.get("tile_infer_k", "")),
                textwrap.fill(str(m.get("note", "")), width=32),
            ]
        )

    if not table_rows:
        return

    columns = ["Config", "Status", "GPU", "Depth", "Iter", "Mask", "Train ratio", "Tile", "Overlap", "K", "Note"]
    row_h = 0.38
    fig, ax = plt.subplots(figsize=(16, max(5.5, 1.6 + row_h * (len(table_rows) + 1))))
    ax.axis("off")
    ax.set_title(f"{title} config summary", fontsize=14, pad=12)
    widths = [0.30, 0.075, 0.035, 0.04, 0.04, 0.08, 0.07, 0.05, 0.055, 0.035, 0.22]
    table = ax.table(cellText=table_rows, colLabels=columns, cellLoc="center", loc="center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for r, values in enumerate(table_rows, start=1):
        lines = max(len(str(value).splitlines()) for value in values)
        for c in range(len(columns)):
            table[r, c].set_height(table[r, c].get_height() * max(1, lines))
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        if c in {0, 10} and r > 0:
            cell.set_text_props(ha="left")
    pdf.savefig(fig, bbox_inches="tight", dpi=180)
    plt.close(fig)
