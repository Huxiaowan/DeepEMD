import torch
import torch.nn as nn


class _HybridBlock(nn.Module):
    def __init__(self, channels: int, spatial_kernel: int = 3):
        super().__init__()
        padding = spatial_kernel // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size=spatial_kernel, padding=padding, groups=channels, padding_mode="reflect")
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, padding_mode="reflect")
        self.bn = nn.BatchNorm2d(channels, 0.1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        return x + y


class Hybrid2D1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 32,
        num_blocks: int = 3,
        spatial_kernel: int = 3,
        z_kernel: int = 3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(hidden_dim, 0.1),
            nn.LeakyReLU(0.2, True),
        )
        self.blocks = nn.ModuleList([_HybridBlock(hidden_dim, spatial_kernel=spatial_kernel) for _ in range(max(1, int(num_blocks)))])
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding_mode="reflect"),
            nn.BatchNorm2d(hidden_dim, 0.1),
            nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, padding_mode="reflect"),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        for block in self.blocks:
            y = block(y)
        y = self.channel_mixer(y)
        return self.head(y)
