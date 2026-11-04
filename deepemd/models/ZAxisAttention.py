import torch
import torch.nn as nn


class ZAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, z_window: int = 7):
        super().__init__()
        self.q = nn.Conv1d(dim, dim, kernel_size=1)
        self.k = nn.Conv1d(dim, dim, kernel_size=1)
        self.v = nn.Conv1d(dim, dim, kernel_size=1)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim, 0.1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w)
        q = self.q(seq)
        k = self.k(seq)
        v = self.v(seq)
        # Attention over Z-axis (channels)
        # q: (B, C, L), k: (B, C, L)
        # (B, C, L) @ (B, L, C) -> (B, C, C)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / max(1.0, float(h * w) ** 0.5), dim=-1)
        out = torch.bmm(attn, v)
        out = self.proj(out).view(b, c, h, w)
        out = self.norm(out + x)
        return self.act(out)
