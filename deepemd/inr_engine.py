import argparse

import numpy as np

from deepemd import engine as vt
from deepemd.utils.util import get_args
from deepemd.utils.util_inr import get_model_inr


def main():
    parser = argparse.ArgumentParser(description="DeepEMD INR engine")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/r9a_inr_baseline_decoder_pe6_noanneal_tilek64.yml",
        help="Path to INR config file.",
    )
    parser.add_argument("--noisy_path", type=str, required=True, help="Path to noisy TIFF file")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth TIFF file")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory to save denoised TIFFs.")
    parser.add_argument("--model_name", type=str, default="model", help="Model name for saving.")
    parser.add_argument("--output_tag", type=str, default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", type=int, default=None)
    parser.add_argument(
        "--tile_size",
        type=int,
        default=None,
        help="Spatial tile size for large frames (0/None=full frame).",
    )
    parser.add_argument(
        "--tile_overlap",
        type=float,
        default=None,
        help="Tile overlap ratio for stitched inference.",
    )

    cli_args = parser.parse_args()
    config_args = get_args(cli_args.config)
    for key, value in vars(cli_args).items():
        if key != "config":
            setattr(config_args, key, value)
    config_args.config = cli_args.config
    config_args.resume = False

    seed = getattr(config_args, "seed", None)
    if seed is None:
        seed = 2024
        setattr(config_args, "seed", seed)
    deterministic = getattr(config_args, "deterministic", None)
    if deterministic is None:
        deterministic = True
        setattr(config_args, "deterministic", deterministic)

    deterministic_bool = (
        bool(int(deterministic))
        if isinstance(deterministic, (int, np.integer, str))
        else bool(deterministic)
    )
    vt.set_deterministic(int(seed), deterministic_bool)

    # Reuse the DeepEMD TIFF pipeline, including its 512-tile blind-spot
    # inference path, while swapping only the model factory to the INR variant.
    vt.get_model = get_model_inr
    vt.denoise_tiff(config_args)


if __name__ == "__main__":
    main()
