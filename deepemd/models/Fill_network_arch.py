import torch
import torch.nn as nn
import random

from deepemd.models.SkipModel import Skip


class Fill_network(nn.Module):
    """Volumetric masking: mask [B,C,H,W] slice windows before prediction.

    With multchannel=False, one spatial mask is broadcast across slices.
    This is not a separate volumetric/3D mask-generation algorithm.
    """
    def __init__(self, in_chans=3, mask_ratio=[30, 30],
                 multchannel=True, patchsz=1, network='skip',
                 mask_type='bernoulli',
                 mask_replace='zero',
                 mask_fixed=False,
                 mask_block_size=16,
                 mask_share_across_channels=False,
                 mask_center_only=False,
                 train_mask_ratio=None,
                 infer_mask_ratio=None,
                 use_self_attn=False, num_heads=8, depth=4, dim=64,
                 attn_window_size=16,
                 use_z_attn=False, use_z_pyramid=False, z_window=7,
                 use_hybrid_2d1d=False,
                 use_3d=False, use_pure_3d=False,
                 use_cross_attn=True):
        super(Fill_network, self).__init__()
        self.in_chans = in_chans
        self.multchannel = multchannel
        self.mask_ratio = mask_ratio
        self.patchsz = patchsz
        self.mask_type = mask_type
        self.mask_replace = mask_replace
        self.mask_fixed = mask_fixed
        self.mask_block_size = mask_block_size
        self.mask_share_across_channels = mask_share_across_channels
        self.mask_center_only = mask_center_only
        self.train_mask_ratio = train_mask_ratio if train_mask_ratio is not None else mask_ratio
        self.infer_mask_ratio = infer_mask_ratio if infer_mask_ratio is not None else mask_ratio
        self._mask_context = None
        self._mask_enabled_override = None
        self._fixed_mask = None

        # 选择网络架构
        if use_pure_3d:
            from deepemd.models.Skip3D import Pure3DUNet
            self.net = Pure3DUNet(in_channels=1, out_channels=1, base_dim=dim, depth=depth)
        elif use_3d:
            from deepemd.models.Skip3D import Skip3D
            # Skip3D应该输出1通道（中间切片），而不是self.in_chans
            self.net = Skip3D(self.in_chans, 1,  # 输入多通道，输出1通道
                            depth=depth,
                            dim=dim,
                            use_z_attn=use_z_attn,
                            z_window=z_window)
        else:
            self.net = Skip(self.in_chans, self.in_chans,
                           use_self_attn=use_self_attn,
                           num_heads=num_heads,
                           depth=depth,
                           dim=dim,
                           attn_window_size=attn_window_size,
                           use_z_attn=use_z_attn,
                           use_z_pyramid=use_z_pyramid,
                           z_window=z_window,
                           use_hybrid_2d1d=use_hybrid_2d1d,
                           use_cross_attn=use_cross_attn)
        # self.net = Skip(self.in_chans, 3)
        # self.net = Skip(self.in_chans, self.in_chans+45)

    def set_mask_context(self, context):
        self._mask_context = context

    def set_mask_enabled(self, enabled):
        self._mask_enabled_override = enabled

    def _get_mask_ratio(self):
        ctx = self._mask_context
        if ctx is None:
            ctx = 'train' if self.net.training else 'infer'
        ratio = self.train_mask_ratio if ctx == 'train' else self.infer_mask_ratio
        if isinstance(ratio, (int, float)):
            return float(ratio) / 100.0 if float(ratio) > 1.0 else float(ratio)
        if isinstance(ratio, (list, tuple)) and len(ratio) == 2:
            return random.randint(int(ratio[0]), int(ratio[1])) / 100.0
        return random.randint(self.mask_ratio[0], self.mask_ratio[1]) / 100.0

    def add_input_mask(self, img):
        B, C, H, W = img.shape
        prob = self._get_mask_ratio()
        channels = 1 if (self.mask_share_across_channels or (not self.multchannel)) else self.in_chans

        if self.mask_fixed and self._fixed_mask is not None:
            mask = self._fixed_mask
        else:
            h0 = max(1, H // self.patchsz)
            w0 = max(1, W // self.patchsz)
            if self.mask_type == 'block':
                bs = max(1, int(self.mask_block_size))
                hh = max(1, (h0 + bs - 1) // bs)
                ww = max(1, (w0 + bs - 1) // bs)
                coarse = (torch.rand((B, 1, hh, ww), device=img.device) > prob).float()
                coarse = coarse.repeat(1, channels, 1, 1)
                mask_small = torch.repeat_interleave(coarse, bs, dim=-2)
                mask_small = torch.repeat_interleave(mask_small, bs, dim=-1)
                mask_small = mask_small[..., :h0, :w0]
            elif self.mask_type == 'stripe_h':
                row_keep = (torch.rand((B, 1, h0, 1), device=img.device) > prob).float()
                mask_small = row_keep.repeat(1, channels, 1, w0)
            elif self.mask_type == 'stripe_v':
                col_keep = (torch.rand((B, 1, 1, w0), device=img.device) > prob).float()
                mask_small = col_keep.repeat(1, channels, h0, 1)
            elif self.mask_type == 'stripe_cross':
                # 同时生成 H 和 V，取逻辑或 (mask=1 表示保留)
                row_keep = (torch.rand((B, 1, h0, 1), device=img.device) > prob).float()
                col_keep = (torch.rand((B, 1, 1, w0), device=img.device) > prob).float()
                # 只有当行和列都被遮挡时才遮挡？不，只要有一个遮挡就遮挡。
                # mask=1 是保留。如果 row_keep=0 (drop)，则 mask=0。
                # 所以应该是 row_keep * col_keep。
                mask_small = row_keep * col_keep
                mask_small = mask_small.repeat(1, channels, 1, 1)
            elif self.mask_type == 'checkerboard':
                # 动态步长棋盘格 (模拟最优盲点)
                # 随机步长 [2, 8]
                stride = random.randint(2, 8)
                off_x = random.randint(0, stride-1)
                off_y = random.randint(0, stride-1)
                # 生成网格坐标
                y = torch.arange(h0, device=img.device).view(-1, 1)
                x = torch.arange(w0, device=img.device).view(1, -1)
                # 遮挡特定相位点: mask=0 where (x%s==ox) & (y%s==oy)
                # 默认 mask=1 (保留)
                mask_small = torch.ones((1, 1, h0, w0), device=img.device)
                # 注意：mask_small 是 float，所以不能直接用 mask_small[...] = 0
                mask_idx = ((y % stride) == off_y) & ((x % stride) == off_x)
                mask_small[0, 0][mask_idx] = 0.0
                mask_small = mask_small.repeat(B, channels, 1, 1)
            elif self.mask_type == 'temporal_comp':
                # 时空互补: 仅适用于多通道输入 (C>=2)
                # 基础随机 Mask
                base_mask = (torch.rand((B, 1, h0, w0), device=img.device) > prob).float()
                if channels >= 2:
                    mask_c = []
                    for c in range(channels):
                        if c % 2 == 0:
                            mask_c.append(base_mask)
                        else:
                            mask_c.append(1.0 - base_mask)
                    mask_small = torch.cat(mask_c, dim=1)
                else:
                    mask_small = base_mask
            elif self.mask_type == 'stripe_d':
                # 对角线/斜向条纹 (Diagonal)
                # 45度斜线: x + y = const
                stride = random.randint(2, 8) # 条纹间隔
                width = max(1, int(stride * prob)) # 条纹宽度 (prob 是保留率，这里用作保留宽度)
                # 实际上 prob 是丢弃率还是保留率?
                # prob = _get_mask_ratio() -> 通常是 mask_ratio (如 0.4 表示遮挡 40%)
                # 代码逻辑: (rand > prob).float() -> mask=1 if rand > prob. 
                # 所以 prob 是遮挡概率。
                # 如果 prob=0.4 (遮挡 40%)，我们希望保留 60%。
                # 斜线掩码: (x + y) % stride < threshold -> Mask (0)
                # threshold = stride * prob
                y = torch.arange(h0, device=img.device).view(-1, 1)
                x = torch.arange(w0, device=img.device).view(1, -1)
                mask_idx = ((x + y) % stride) < (stride * prob)
                mask_small = torch.ones((1, 1, h0, w0), device=img.device)
                mask_small[0, 0][mask_idx] = 0.0
                mask_small = mask_small.repeat(B, channels, 1, 1)
            elif self.mask_type == 'stripe_mix_hv':
                if random.random() < 0.5:
                    row_keep = (torch.rand((B, 1, h0, 1), device=img.device) > prob).float()
                    mask_small = row_keep.repeat(1, channels, 1, w0)
                else:
                    col_keep = (torch.rand((B, 1, 1, w0), device=img.device) > prob).float()
                    mask_small = col_keep.repeat(1, channels, h0, 1)
            elif self.mask_type == 'block_stripe_cross':
                h_s, w_s = max(1, h0 // 2), max(1, w0 // 2)
                row_keep = (torch.rand((B, 1, h_s, 1), device=img.device) > prob).float()
                col_keep = (torch.rand((B, 1, 1, w_s), device=img.device) > prob).float()
                mask_tiny = row_keep * col_keep
                mask_small = torch.nn.functional.interpolate(mask_tiny, size=(h0, w0), mode='nearest')
                mask_small = mask_small.repeat(1, channels, 1, 1)
            elif self.mask_type == 'dual_random_stripe_cross':
                prob_row = self._get_mask_ratio()
                prob_col = self._get_mask_ratio()
                row_keep = (torch.rand((B, 1, h0, 1), device=img.device) > prob_row).float()
                col_keep = (torch.rand((B, 1, 1, w0), device=img.device) > prob_col).float()
                mask_small = row_keep * col_keep
                mask_small = mask_small.repeat(1, channels, 1, 1)
            elif self.mask_type == 'grid_stride3':
                stride = 3
                off_x = random.randint(0, stride-1)
                off_y = random.randint(0, stride-1)
                y = torch.arange(h0, device=img.device).view(-1, 1)
                x = torch.arange(w0, device=img.device).view(1, -1)
                mask_idx = ((y % stride) == off_y) & ((x % stride) == off_x)
                mask_small = torch.ones((1, 1, h0, w0), device=img.device)
                mask_small[0, 0][mask_idx] = 0.0
                mask_small = mask_small.repeat(B, channels, 1, 1)
            else:
                mask_small = (torch.rand((B, channels, h0, w0), device=img.device) > prob).float()

            if self.patchsz > 1:
                mask = torch.repeat_interleave(mask_small, self.patchsz, dim=-2)
                mask = torch.repeat_interleave(mask, self.patchsz, dim=-1)
                mask = mask[..., :H, :W]
            else:
                mask = mask_small

            if self.mask_fixed:
                self._fixed_mask = mask

        if self.mask_center_only and C > 1:
            mask_full = torch.ones((B, C, H, W), device=img.device, dtype=mask.dtype)
            if C == 3:
                center = slice(1, 2)
            elif C == 9:
                center = slice(3, 6)
            else:
                mid = C // 2
                center = slice(mid, mid + 1)
            if mask.size(1) == 1:
                mask_full[:, center] = mask[:, :1]
            else:
                mask_full[:, center] = mask[:, center]
            mask = mask_full

        self.mask = mask

        if self.mask_replace == 'random':
            rep = torch.rand_like(img)
            img_masked = img * mask + rep * (1 - mask)
        elif self.mask_replace == 'mean':
            rep = torch.nn.functional.avg_pool2d(img, kernel_size=3, stride=1, padding=1)
            img_masked = img * mask + rep * (1 - mask)
        elif self.mask_replace == 'neighbor':
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            dy, dx = offsets[random.randint(0, len(offsets) - 1)]
            rep = torch.roll(img, shifts=(dy, dx), dims=(-2, -1))
            img_masked = img * mask + rep * (1 - mask)
        else:
            img_masked = img * mask

        return img_masked, mask

    def get_mask(self):
        return self.mask

    def save_model(self, path):
        """保存模型"""
        import torch
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'config': {
                'in_chans': self.in_chans,
                'multchannel': self.multchannel,
                'mask_ratio': self.mask_ratio,
            }
        }, path)
        print(f"模型已保存: {path}")

    def forward(self, x):
        x_ = x.detach().clone()
        apply_mask = self.net.training
        if self._mask_enabled_override is not None:
            apply_mask = bool(self._mask_enabled_override)
        if apply_mask:
            x, _ = self.add_input_mask(x)
        else:
            B, C, H, W = x.shape
            self.mask = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)
        # out = self.net(torch.cat([x, mask], dim=1))  # +x_mean

        out = self.net(x)
        return out
