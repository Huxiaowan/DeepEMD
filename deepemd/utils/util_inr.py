import numpy as np
import torch

from deepemd.models.Fill_network_inr import FillNetworkINR
from deepemd.utils.util import comp_params


def get_model_inr(args, resume=True, device="cuda"):
    multchannel = getattr(args, "multchannel", False) if not resume else True

    use_self_attn = getattr(args, "use_self_attn", False)
    num_heads = getattr(args, "num_heads", 8)
    depth = getattr(args, "depth", 4)
    dim = getattr(args, "dim", 64)
    attn_window_size = getattr(args, "attn_window_size", 16)
    use_cross_attn = getattr(args, "use_cross_attn", True)

    use_z_attn = getattr(args, "use_z_attn", False)
    use_z_pyramid = getattr(args, "use_z_pyramid", False)
    z_window = getattr(args, "z_window", 7)
    use_hybrid_2d1d = getattr(args, "use_hybrid_2d1d", False)

    use_3d = getattr(args, "use_3d", False)
    use_pure_3d = getattr(args, "use_pure_3d", False)

    mask_type = getattr(args, "mask_type", "bernoulli")
    mask_replace = getattr(args, "mask_replace", "zero")
    mask_fixed = getattr(args, "mask_fixed", False)
    mask_block_size = getattr(args, "mask_block_size", 16)
    mask_share_across_channels = getattr(args, "mask_share_across_channels", False)
    mask_patchsz = getattr(args, "mask_patchsz", 1)
    mask_center_only = getattr(args, "mask_center_only", False)
    train_mask_ratio = getattr(args, "train_mask_ratio", None)
    infer_mask_ratio = getattr(args, "infer_mask_ratio", None)
    mask_ratio = getattr(args, "mask_ratio", None)
    if mask_ratio is None:
        mask_ratio = train_mask_ratio if train_mask_ratio is not None else [15, 15]
        setattr(args, "mask_ratio", mask_ratio)

    inr_variant = getattr(args, "inr_variant", "decoder")
    inr_pe_levels = int(getattr(args, "inr_pe_levels", 4) or 4)
    inr_head_hidden = int(getattr(args, "inr_head_hidden", 32) or 32)
    inr_t_levels = int(getattr(args, "inr_t_levels", 4) or 4)
    inr_t_hidden = int(getattr(args, "inr_t_hidden", 32) or 32)
    inr_low_pe_levels = int(getattr(args, "inr_low_pe_levels", 2) or 2)
    inr_low_hidden = int(getattr(args, "inr_low_hidden", 32) or 32)
    inr_siren_w0 = float(getattr(args, "inr_siren_w0", 30.0) or 30.0)
    inr_res_scale = float(getattr(args, "inr_res_scale", 0.1) or 0.1)
    inr_res_hidden = int(getattr(args, "inr_res_hidden", 32) or 32)
    inr_use_multiscale_pe = bool(getattr(args, "inr_use_multiscale_pe", False))

    net_fill = FillNetworkINR(
        in_chans=args.input_depth,
        mask_ratio=mask_ratio,
        multchannel=multchannel,
        patchsz=mask_patchsz,
        mask_type=mask_type,
        mask_replace=mask_replace,
        mask_fixed=mask_fixed,
        mask_block_size=mask_block_size,
        mask_share_across_channels=mask_share_across_channels,
        mask_center_only=mask_center_only,
        train_mask_ratio=train_mask_ratio,
        infer_mask_ratio=infer_mask_ratio,
        network=str(getattr(args, "modelpath", "skip_inr.pth")).split("_")[-1].split(".")[0],
        use_self_attn=use_self_attn,
        num_heads=num_heads,
        depth=depth,
        dim=dim,
        attn_window_size=attn_window_size,
        use_z_attn=use_z_attn,
        use_z_pyramid=use_z_pyramid,
        z_window=z_window,
        use_hybrid_2d1d=use_hybrid_2d1d,
        use_3d=use_3d,
        use_pure_3d=use_pure_3d,
        use_cross_attn=use_cross_attn,
        inr_variant=inr_variant,
        inr_pe_levels=inr_pe_levels,
        inr_head_hidden=inr_head_hidden,
        inr_t_levels=inr_t_levels,
        inr_t_hidden=inr_t_hidden,
        inr_low_pe_levels=inr_low_pe_levels,
        inr_low_hidden=inr_low_hidden,
        inr_siren_w0=inr_siren_w0,
        inr_res_scale=inr_res_scale,
        inr_res_hidden=inr_res_hidden,
        inr_use_multiscale_pe=inr_use_multiscale_pe,
    ).to(device)

    comp_params(net_fill)

    lr_groups = []
    if resume:
        saved_state_dict = torch.load(args.modelpath, map_location=device)["params"]
        net_fill.load_state_dict(saved_state_dict, strict=True)
    lr_groups.append({"params": net_fill.parameters(), "lr": args.lr})
    return net_fill, lr_groups
