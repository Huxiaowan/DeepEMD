import torch
import torch.nn as nn
import torch.nn.functional as F


class ZPyramidBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = max(1, int(num_levels))
        self.proj = nn.Conv2d(in_channels * self.num_levels, out_channels, kernel_size=1, padding_mode="reflect")
        self.bn = nn.BatchNorm2d(out_channels, 0.1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feats = [x]
        for level in range(1, self.num_levels):
            scale = 2 ** level
            pooled = F.avg_pool2d(x, kernel_size=scale, stride=scale, ceil_mode=True)
            up = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(up)
        out = torch.cat(feats, dim=1)
        out = self.proj(out)
        out = self.bn(out)
        return self.act(out)
