import math
import torch
import torch.nn as nn

from deepemd.models.Skip3D import Skip3D
from deepemd.models.SkipModel import ConcatBlock, conv


def _make_coord_grid(h: int, w: int, device, dtype):
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def _fourier_encode_2d(xx, yy, levels: int):
    freqs = (2.0 ** torch.arange(levels, device=xx.device, dtype=xx.dtype)) * math.pi
    xx = xx[..., None] * freqs
    yy = yy[..., None] * freqs
    enc = torch.cat([torch.sin(xx), torch.cos(xx), torch.sin(yy), torch.cos(yy)], dim=-1)
    return enc


def _fourier_encode_1d(tt, levels: int):
    freqs = (2.0 ** torch.arange(levels, device=tt.device, dtype=tt.dtype)) * math.pi
    tt = tt[..., None] * freqs
    enc = torch.cat([torch.sin(tt), torch.cos(tt)], dim=-1)
    return enc


class CoordEncoding1D(nn.Module):
    def __init__(self, levels: int = 4):
        super().__init__()
        self.levels = int(levels)

    def forward(self, t: torch.Tensor, levels: int | None = None):
        lv = int(levels) if levels is not None else self.levels
        return _fourier_encode_1d(t, lv)


class CoordEncoding2D(nn.Module):
    def __init__(self, levels: int = 4):
        super().__init__()
        self.levels = int(levels)
        self._cache = {}

    def forward(self, h: int, w: int, device, dtype, gate: torch.Tensor | None = None):
        key = (h, w, str(device), str(dtype), self.levels)
        v = self._cache.get(key, None)
        if v is not None:
            enc = v
        else:
            xx, yy = _make_coord_grid(h, w, device=device, dtype=dtype)
            enc = _fourier_encode_2d(xx, yy, self.levels)
            enc = enc.permute(2, 0, 1).contiguous().unsqueeze(0)
            self._cache[key] = enc
        if gate is None:
            return enc
        gate = gate.to(device=device, dtype=enc.dtype).view(1, self.levels, 1, 1, 1)
        enc5 = enc.view(1, self.levels, 4, h, w) * gate
        return enc5.view(1, self.levels * 4, h, w)


class CoordHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden, hidden, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden, out_ch, 1, padding_mode="reflect"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _siren_init_(layer: nn.Conv2d, is_first: bool, w0: float) -> None:
    in_dim = layer.weight.size(1)
    if is_first:
        bound = 1.0 / max(1, in_dim)
    else:
        bound = (math.sqrt(6.0 / max(1, in_dim))) / max(1e-6, float(w0))
    nn.init.uniform_(layer.weight, -bound, bound)
    if layer.bias is not None:
        nn.init.uniform_(layer.bias, -bound, bound)


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenCoordHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int, out_ch: int, w0: float = 30.0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_ch, hidden, 1, padding_mode="reflect")
        self.act1 = Sine(w0=w0)
        self.fc2 = nn.Conv2d(hidden, hidden, 1, padding_mode="reflect")
        self.act2 = Sine(w0=1.0)
        self.fc3 = nn.Conv2d(hidden, out_ch, 1, padding_mode="reflect")
        self.out = nn.Sigmoid()
        _siren_init_(self.fc1, is_first=True, w0=w0)
        _siren_init_(self.fc2, is_first=False, w0=1.0)
        _siren_init_(self.fc3, is_first=False, w0=1.0)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return self.out(x)


class TemporalFiLM(nn.Module):
    def __init__(self, channels: int, levels: int = 4, hidden: int = 32):
        super().__init__()
        self.channels = int(channels)
        self.levels = int(levels)
        self.mlp = nn.Sequential(
            nn.Linear(self.levels * 2, hidden),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _encode_t(self, t: torch.Tensor):
        freqs = (2.0 ** torch.arange(self.levels, device=t.device, dtype=t.dtype)) * math.pi
        tt = t[..., None] * freqs
        return torch.cat([torch.sin(tt), torch.cos(tt)], dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels:
            return x
        if self.channels == 1:
            t = torch.zeros((1,), device=x.device, dtype=x.dtype)
        else:
            t = torch.linspace(-1.0, 1.0, self.channels, device=x.device, dtype=x.dtype)
        te = self._encode_t(t)
        sb = self.mlp(te)
        scale = 1.0 + sb[:, 0].view(1, c, 1, 1)
        bias = sb[:, 1].view(1, c, 1, 1)
        return x * scale + bias


class SkipINRDecoder(nn.Module):
    """Fourier voxel neural field (Figure 1): existing feature-conditioned head.

    The released decoder encodes XY coordinates in 2D; neighboring Z slices
    are input/output channels, not a third coordinate in the Fourier grid.
    """
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        dim: int = 64,
        depth: int = 4,
        use_self_attn: bool = False,
        num_heads: int = 8,
        attn_window_size: int = 16,
        use_z_attn: bool = False,
        use_z_pyramid: bool = False,
        z_window: int = 7,
        use_cross_attn: bool = True,
        pe_levels: int = 4,
        head_hidden: int = 32,
        use_multiscale_pe: bool = False,
    ):
        super().__init__()
        self.layers = ConcatBlock(
            input_channels,
            dim,
            depth=depth,
            is_first=True,
            use_self_attn=use_self_attn,
            num_heads=num_heads,
            attn_window_size=attn_window_size,
            use_z_attn=use_z_attn,
            use_z_pyramid=use_z_pyramid,
            z_window=z_window,
            use_cross_attn=use_cross_attn,
        )
        self.pe = CoordEncoding2D(levels=pe_levels)
        pe_ch = int(pe_levels) * 4
        self.head = CoordHead(dim + pe_ch, head_hidden, output_channels)
        self.freq_gate = None
        self.use_multiscale_pe = use_multiscale_pe
        if use_multiscale_pe:
            self.ms_pe_mlp = nn.Sequential(
                nn.Conv2d(pe_ch, head_hidden, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(head_hidden, dim * 2, 1),
            )
            nn.init.zeros_(self.ms_pe_mlp[-1].weight)
            nn.init.zeros_(self.ms_pe_mlp[-1].bias)

    def set_freq_gate(self, gate: torch.Tensor | None):
        self.freq_gate = gate

    def forward(self, x):
        feat = self.layers(x)
        b, c, h, w = feat.shape
        pe = self.pe(h, w, device=feat.device, dtype=feat.dtype, gate=self.freq_gate).expand(b, -1, -1, -1)
        if self.use_multiscale_pe:
            sb = self.ms_pe_mlp(pe)
            scale = 1.0 + sb[:, :c]
            bias = sb[:, c:]
            feat = feat * scale + bias
        out = self.head(torch.cat([feat, pe], dim=1))
        return out


class SkipINR3DDecoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 7,
        output_channels: int = 1,
        dim: int = 64,
        depth: int = 4,
        use_z_attn: bool = False,
        z_window: int = 7,
        pe_levels: int = 4,
        head_hidden: int = 32,
    ):
        super().__init__()
        self.backbone = Skip3D(
            input_channels=input_channels,
            output_channels=dim,
            dim=dim,
            depth=depth,
            use_z_attn=use_z_attn,
            z_window=z_window,
        )
        self.pe = CoordEncoding2D(levels=pe_levels)
        pe_ch = int(pe_levels) * 4
        self.head = CoordHead(dim + pe_ch, head_hidden, output_channels)
        self.freq_gate = None

    def set_freq_gate(self, gate: torch.Tensor | None):
        self.freq_gate = gate

    def forward(self, x):
        feat = self.backbone(x)
        b, _, h, w = feat.shape
        pe = self.pe(h, w, device=feat.device, dtype=feat.dtype, gate=self.freq_gate).expand(b, -1, -1, -1)
        return self.head(torch.cat([feat, pe], dim=1))


class SkipINRSirenDecoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        dim: int = 64,
        depth: int = 4,
        use_self_attn: bool = False,
        num_heads: int = 8,
        attn_window_size: int = 16,
        use_z_attn: bool = False,
        use_z_pyramid: bool = False,
        z_window: int = 7,
        use_cross_attn: bool = True,
        pe_levels: int = 4,
        head_hidden: int = 32,
        siren_w0: float = 30.0,
    ):
        super().__init__()
        self.layers = ConcatBlock(
            input_channels,
            dim,
            depth=depth,
            is_first=True,
            use_self_attn=use_self_attn,
            num_heads=num_heads,
            attn_window_size=attn_window_size,
            use_z_attn=use_z_attn,
            use_z_pyramid=use_z_pyramid,
            z_window=z_window,
            use_cross_attn=use_cross_attn,
        )
        self.pe = CoordEncoding2D(levels=pe_levels)
        pe_ch = int(pe_levels) * 4
        self.head = SirenCoordHead(dim + pe_ch, head_hidden, output_channels, w0=siren_w0)
        self.freq_gate = None

    def set_freq_gate(self, gate: torch.Tensor | None):
        self.freq_gate = gate

    def forward(self, x):
        feat = self.layers(x)
        b, _, h, w = feat.shape
        pe = self.pe(h, w, device=feat.device, dtype=feat.dtype, gate=self.freq_gate).expand(b, -1, -1, -1)
        out = self.head(torch.cat([feat, pe], dim=1))
        return out


class SkipINRResidual(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        dim: int = 64,
        depth: int = 4,
        use_self_attn: bool = False,
        num_heads: int = 8,
        attn_window_size: int = 16,
        use_z_attn: bool = False,
        use_z_pyramid: bool = False,
        z_window: int = 7,
        use_cross_attn: bool = True,
        pe_levels: int = 4,
        res_hidden: int = 32,
        res_scale: float = 0.1,
    ):
        super().__init__()
        self.res_scale = float(res_scale)
        self.layers = ConcatBlock(
            input_channels,
            dim,
            depth=depth,
            is_first=True,
            use_self_attn=use_self_attn,
            num_heads=num_heads,
            attn_window_size=attn_window_size,
            use_z_attn=use_z_attn,
            use_z_pyramid=use_z_pyramid,
            z_window=z_window,
            use_cross_attn=use_cross_attn,
        )
        self.base = nn.Sequential(
            nn.Conv2d(dim, output_channels, 1, padding_mode="reflect"),
            nn.Sigmoid(),
        )
        self.pe = CoordEncoding2D(levels=pe_levels)
        pe_ch = int(pe_levels) * 4
        self.res_head = nn.Sequential(
            nn.Conv2d(dim + pe_ch, res_hidden, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(res_hidden, output_channels, 1, padding_mode="reflect"),
            nn.Tanh(),
        )
        self.freq_gate = None

    def set_freq_gate(self, gate: torch.Tensor | None):
        self.freq_gate = gate

    def forward(self, x):
        feat = self.layers(x)
        base = self.base(feat)
        b, _, h, w = feat.shape
        pe = self.pe(h, w, device=feat.device, dtype=feat.dtype, gate=self.freq_gate).expand(b, -1, -1, -1)
        res = self.res_head(torch.cat([feat, pe], dim=1)) * self.res_scale
        return torch.clamp(base + res, 0.0, 1.0)


class SkipINRTemporalFiLM(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        dim: int = 64,
        depth: int = 4,
        use_self_attn: bool = False,
        num_heads: int = 8,
        attn_window_size: int = 16,
        use_z_attn: bool = False,
        use_z_pyramid: bool = False,
        z_window: int = 7,
        use_cross_attn: bool = True,
        t_levels: int = 4,
        t_hidden: int = 32,
    ):
        super().__init__()
        self.tfilm = TemporalFiLM(input_channels, levels=t_levels, hidden=t_hidden)
        self.layers = ConcatBlock(
            input_channels,
            dim,
            depth=depth,
            is_first=True,
            use_self_attn=use_self_attn,
            num_heads=num_heads,
            attn_window_size=attn_window_size,
            use_z_attn=use_z_attn,
            use_z_pyramid=use_z_pyramid,
            z_window=z_window,
            use_cross_attn=use_cross_attn,
        )
        self.last = nn.Sequential(
            nn.Conv2d(dim, output_channels, 1, padding_mode="reflect"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.tfilm(x)
        x = self.layers(x)
        x = self.last(x)
        return x


class SkipINRRegLoss(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        dim: int = 64,
        depth: int = 4,
        use_self_attn: bool = False,
        num_heads: int = 8,
        attn_window_size: int = 16,
        use_z_attn: bool = False,
        use_z_pyramid: bool = False,
        z_window: int = 7,
        use_cross_attn: bool = True,
        low_pe_levels: int = 2,
        low_hidden: int = 32,
    ):
        super().__init__()
        self.layers = ConcatBlock(
            input_channels,
            dim,
            depth=depth,
            is_first=True,
            use_self_attn=use_self_attn,
            num_heads=num_heads,
            attn_window_size=attn_window_size,
            use_z_attn=use_z_attn,
            use_z_pyramid=use_z_pyramid,
            z_window=z_window,
            use_cross_attn=use_cross_attn,
        )
        self.last = nn.Sequential(
            nn.Conv2d(dim, output_channels, 1, padding_mode="reflect"),
            nn.Sigmoid(),
        )
        self.low_pe = CoordEncoding2D(levels=low_pe_levels)
        pe_ch = int(low_pe_levels) * 4
        self.low_head = nn.Sequential(
            nn.Conv2d(pe_ch, low_hidden, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(low_hidden, 1, 1, padding_mode="reflect"),
            nn.Sigmoid(),
        )
        self._last_low = None

    def get_inr_low(self):
        return self._last_low

    def forward(self, x):
        feat = self.layers(x)
        out = self.last(feat)
        b, _, h, w = out.shape
        pe = self.low_pe(h, w, device=out.device, dtype=out.dtype).expand(b, -1, -1, -1)
        self._last_low = self.low_head(pe)
        return out
