import torch.nn as nn

from deepemd.models.Fill_network_arch import Fill_network
from deepemd.models.SkipModel_INR import (
    SkipINR3DDecoder,
    SkipINRDecoder,
    SkipINRRegLoss,
    SkipINRResidual,
    SkipINRSirenDecoder,
    SkipINRTemporalFiLM,
)


class FillNetworkINR(Fill_network):
    def __init__(
        self,
        in_chans=3,
        mask_ratio=(30, 30),
        multchannel=True,
        patchsz=1,
        network="skip",
        mask_type="bernoulli",
        mask_replace="zero",
        mask_fixed=False,
        mask_block_size=16,
        mask_share_across_channels=False,
        mask_center_only=False,
        train_mask_ratio=None,
        infer_mask_ratio=None,
        use_self_attn=False,
        num_heads=8,
        depth=4,
        dim=64,
        attn_window_size=16,
        use_z_attn=False,
        use_z_pyramid=False,
        z_window=7,
        use_hybrid_2d1d=False,
        use_3d=False,
        use_pure_3d=False,
        use_cross_attn=True,
        inr_variant="decoder",
        inr_pe_levels=4,
        inr_head_hidden=32,
        inr_t_levels=4,
        inr_t_hidden=32,
        inr_low_pe_levels=2,
        inr_low_hidden=32,
        inr_siren_w0=30.0,
        inr_res_scale=0.1,
        inr_res_hidden=32,
        inr_use_multiscale_pe=False,
    ):
        super().__init__(
            in_chans=in_chans,
            mask_ratio=mask_ratio,
            multchannel=multchannel,
            patchsz=patchsz,
            network=network,
            mask_type=mask_type,
            mask_replace=mask_replace,
            mask_fixed=mask_fixed,
            mask_block_size=mask_block_size,
            mask_share_across_channels=mask_share_across_channels,
            mask_center_only=mask_center_only,
            train_mask_ratio=train_mask_ratio,
            infer_mask_ratio=infer_mask_ratio,
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
        )

        v = str(inr_variant or "").strip().lower()
        if v in ("3d_decoder", "inr3d", "inr3d_decoder", "inr_3d", "inr_3d_decoder"):
            self.net = SkipINR3DDecoder(
                input_channels=self.in_chans,
                output_channels=1,
                dim=dim,
                depth=depth,
                use_z_attn=use_z_attn,
                z_window=z_window,
                pe_levels=inr_pe_levels,
                head_hidden=inr_head_hidden,
            )
        elif v in ("decoder", "inr_decoder"):
            self.net = SkipINRDecoder(
                input_channels=self.in_chans,
                output_channels=self.in_chans,
                dim=dim,
                depth=depth,
                use_self_attn=use_self_attn,
                num_heads=num_heads,
                attn_window_size=attn_window_size,
                use_z_attn=use_z_attn,
                use_z_pyramid=use_z_pyramid,
                z_window=z_window,
                use_cross_attn=use_cross_attn,
                pe_levels=inr_pe_levels,
                head_hidden=inr_head_hidden,
                use_multiscale_pe=inr_use_multiscale_pe,
            )
        elif v in ("siren", "siren_decoder", "inr_siren", "inr_siren_decoder"):
            self.net = SkipINRSirenDecoder(
                input_channels=self.in_chans,
                output_channels=self.in_chans,
                dim=dim,
                depth=depth,
                use_self_attn=use_self_attn,
                num_heads=num_heads,
                attn_window_size=attn_window_size,
                use_z_attn=use_z_attn,
                use_z_pyramid=use_z_pyramid,
                z_window=z_window,
                use_cross_attn=use_cross_attn,
                pe_levels=inr_pe_levels,
                head_hidden=inr_head_hidden,
                siren_w0=inr_siren_w0,
            )
        elif v in ("tfilm", "temporal", "inr_tfilm", "inr_temporal"):
            self.net = SkipINRTemporalFiLM(
                input_channels=self.in_chans,
                output_channels=self.in_chans,
                dim=dim,
                depth=depth,
                use_self_attn=use_self_attn,
                num_heads=num_heads,
                attn_window_size=attn_window_size,
                use_z_attn=use_z_attn,
                use_z_pyramid=use_z_pyramid,
                z_window=z_window,
                use_cross_attn=use_cross_attn,
                t_levels=inr_t_levels,
                t_hidden=inr_t_hidden,
            )
        elif v in ("reg", "regloss", "inr_reg", "inr_regloss"):
            self.net = SkipINRRegLoss(
                input_channels=self.in_chans,
                output_channels=self.in_chans,
                dim=dim,
                depth=depth,
                use_self_attn=use_self_attn,
                num_heads=num_heads,
                attn_window_size=attn_window_size,
                use_z_attn=use_z_attn,
                use_z_pyramid=use_z_pyramid,
                z_window=z_window,
                use_cross_attn=use_cross_attn,
                low_pe_levels=inr_low_pe_levels,
                low_hidden=inr_low_hidden,
            )
        elif v in ("res", "residual", "inr_res", "inr_residual"):
            self.net = SkipINRResidual(
                input_channels=self.in_chans,
                output_channels=self.in_chans,
                dim=dim,
                depth=depth,
                use_self_attn=use_self_attn,
                num_heads=num_heads,
                attn_window_size=attn_window_size,
                use_z_attn=use_z_attn,
                use_z_pyramid=use_z_pyramid,
                z_window=z_window,
                use_cross_attn=use_cross_attn,
                pe_levels=inr_pe_levels,
                res_hidden=inr_res_hidden,
                res_scale=inr_res_scale,
            )
        else:
            self.net = nn.Identity()

    def get_inr_low(self):
        if hasattr(self.net, "get_inr_low"):
            return self.net.get_inr_low()
        return None

    def set_freq_gate(self, gate):
        if hasattr(self.net, "set_freq_gate"):
            self.net.set_freq_gate(gate)
