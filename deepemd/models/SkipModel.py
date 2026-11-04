import torch
import torch.nn as nn


def conv(in_ch, out_ch, kernel, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=(kernel-1)//2, padding_mode='reflect'),
        nn.BatchNorm2d(out_ch, 0.1),
        nn.LeakyReLU(0.2, True)
    )


def conv3d(in_ch, out_ch, kernel=(3,3,3), stride=(1,1,1)):
    """3D卷积辅助函数，用于真正的3D CNN"""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
                 padding=(k//2 for k in kernel), padding_mode='reflect'),
        nn.BatchNorm3d(out_ch, 0.1),
        nn.LeakyReLU(0.2, True)
    )


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        # 使用PyTorch内置多头注意力模块
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True  # PyTorch 1.10+ 支持
        )
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img):
        """
        输入: [B, 2*C, H, W] (concatenated feature maps)
        输出: 交叉注意力增强后的特征图 [B, 2*C, H, W]
        """
        B, C2, H, W = img.shape
        C = C2 // 2  # Split channels into two halves

        # Split into two halves
        img_a = img[:, :C]  # [B, C, H, W]
        img_b = img[:, C:]  # [B, C, H, W]

        # Step 1: 图像分块展平为序列 [B, N, C]
        a_seq = img_a.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        b_seq = img_b.flatten(2).permute(0, 2, 1)

        # Step 2: 使用PyTorch内置注意力计算
        attn_output, _ = self.cross_attn(
            query=a_seq,   # Query来自img_a
            key=b_seq,     # Key来自img_b
            value=b_seq,   # Value来自img_b
            need_weights=False
        )

        # Step 3: 残差连接 + 层归一化
        output_a = self.norm(a_seq + attn_output)  # [B, H*W, C]

        # Step 4: 恢复空间维度
        output_a = output_a.permute(0, 2, 1).view(B, C, H, W)

        # Concat with img_b
        output = torch.cat([output_a, img_b], dim=1)  # [B, 2*C, H, W]
        return output

# PSNR: 29.90, Time: 140.28s = 2.34min


class ConcatBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, depth=4, is_first=False,
                 use_self_attn=False, num_heads=8, attn_window_size=16,
                 use_z_attn=False, use_z_pyramid=False, z_window=7,
                 use_cross_attn=True):
        super().__init__()

        # 动态确定输入通道
        input_ch = in_channels if is_first else out_channels

        # 分支1：始终1x1卷积
        self.branch1 = conv(input_ch, out_channels, 1)

        # 分支2：基础结构
        self.branch2 = nn.Sequential(
            conv(input_ch, out_channels, 3, stride=2),
            conv(out_channels, out_channels, 3),
            ConcatBlock(out_channels=out_channels, depth=depth-1,
                       use_self_attn=use_self_attn, num_heads=num_heads,
                       attn_window_size=attn_window_size,
                       use_z_attn=use_z_attn, use_z_pyramid=use_z_pyramid,
                       z_window=z_window, use_cross_attn=use_cross_attn) if depth > 0 else
            nn.Identity(),
        )
        self.up = nn.Sequential(
            CrossAttention(embed_dim=out_channels // 2, num_heads=max(1, out_channels // 16)) if (depth == 0 and use_cross_attn) else nn.Identity(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # 后处理模块
        self.fusion = nn.Sequential(
            nn.BatchNorm2d(2*out_channels, 0.1),
            conv(2*out_channels, out_channels, 3),
            conv(out_channels, out_channels, 1)
        )

        # Self-Attention（可选）
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = SelfAttentionBlock(out_channels, num_heads=num_heads, window_size=attn_window_size)

        # Z轴模块（可选）
        self.use_z_attn = use_z_attn
        self.use_z_pyramid = use_z_pyramid

        if use_z_attn:
            from deepemd.models.ZAxisAttention import ZAttentionBlock
            self.z_attn_block = ZAttentionBlock(dim=out_channels, num_heads=4, z_window=z_window)

        if use_z_pyramid:
            from deepemd.models.ZAxisPyramid import ZPyramidBlock
            self.z_pyramid_block = ZPyramidBlock(in_channels=out_channels, out_channels=out_channels, num_levels=3)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x2 = self.up(x2)
        x = self.fusion(torch.cat([x1, x2], 1))
        # 添加Self-Attention（如果启用）
        if self.use_self_attn:
            x = self.self_attn(x)

        # 添加Z轴模块（如果启用）
        if self.use_z_attn:
            x = self.z_attn_block(x)

        if self.use_z_pyramid:
            x = self.z_pyramid_block(x)

        return x


class Skip(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, dim=64,
                 use_self_attn=False, num_heads=8, depth=4, attn_window_size=16,
                 use_z_attn=False, use_z_pyramid=False, z_window=7,
                 use_hybrid_2d1d=False, use_cross_attn=True):
        super().__init__()

        # 保存配置
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_hybrid_2d1d = use_hybrid_2d1d

        # 如果使用混合2D+Z-1D模式
        if use_hybrid_2d1d:
            from deepemd.models.Hybrid2D1D import Hybrid2D1D
            self.layers = Hybrid2D1D(
                in_channels=input_channels,
                out_channels=output_channels,
                hidden_dim=dim,
                num_blocks=depth,
                spatial_kernel=3,
                z_kernel=3
            )
            self.last = nn.Identity()
        else:
            # 标准模式
            self.layers = ConcatBlock(input_channels, dim, depth=depth, is_first=True,
                                     use_self_attn=use_self_attn, num_heads=num_heads,
                                     attn_window_size=attn_window_size,
                                     use_z_attn=use_z_attn, use_z_pyramid=use_z_pyramid,
                                     z_window=z_window, use_cross_attn=use_cross_attn)
            self.last = nn.Sequential(
                nn.Conv2d(dim, output_channels, 1, padding_mode='reflect'),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.layers(x)
        x = self.last(x)
        return x


class SelfAttentionBlock(nn.Module):
    """空间自注意力块 - 高效版，使用分块窗口处理"""
    def __init__(self, dim=64, num_heads=8, window_size=16):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # 用于保存原始尺寸信息
        self.H_orig = None
        self.W_orig = None

    def forward(self, x):
        B, C, H, W = x.shape
        self.H_orig = H
        self.W_orig = W

        # Padding到window_size的倍数
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

        B, C, H_pad, W_pad = x.shape

        # 分割成窗口，保留空间维度
        x_windows = self.window_partition(x)  # [B, num_h, num_w, window_size^2, C]

        B, num_h, num_w, N, C = x_windows.shape

        # 合并空间维度以便批量处理
        x_windows = x_windows.view(B, num_h * num_w, N, C)  # [B, num_windows, N, C]

        # Layer norm
        x_windows_norm = self.norm(x_windows)

        # 展平为[B*num_windows, N, C]以便QKV投影
        x_windows_norm = x_windows_norm.view(B * num_h * num_w, N, C)

        # QKV projection
        qkv = self.qkv(x_windows_norm).reshape(B * num_h * num_w, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*num_windows, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 分块处理窗口
        chunk_size = min(50, num_h * num_w)  # 每次最多处理50个窗口
        output_chunks = []

        for chunk_idx, chunk_start in enumerate(range(0, num_h * num_w, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, num_h * num_w)
            actual_chunk_size = chunk_end - chunk_start

            # 提取当前chunk的QKV
            q_chunk = q[chunk_start:chunk_end]  # [actual_chunk_size, num_heads, N, head_dim]
            k_chunk = k[chunk_start:chunk_end]
            v_chunk = v[chunk_start:chunk_end]

            # Attention计算 (每个窗口内)
            attn = (q_chunk @ k_chunk.transpose(-2, -1)) * self.scale  # [actual_chunk_size, num_heads, N, N]
            attn = attn.softmax(dim=-1)

            x_chunk = (attn @ v_chunk)  # [actual_chunk_size, num_heads, N, head_dim]
            x_chunk = x_chunk.reshape(actual_chunk_size, N, C)  # [actual_chunk_size, N, C]
            x_chunk = self.proj(x_chunk)

            output_chunks.append(x_chunk)

        # 合并所有chunks
        x_windows_out = torch.cat(output_chunks, dim=0)  # [B*num_windows, N, C]

        # 恢复空间维度
        x_windows_out = x_windows_out.view(B, num_h, num_w, N, C)

        # 残差连接
        x_windows = x_windows.view(B, num_h, num_w, N, C)
        x_windows_out = x_windows + x_windows_out

        # 恢复图像
        x = self.window_unpartition(x_windows_out, H_pad, W_pad, B)

        # 裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :self.H_orig, :self.W_orig]

        return x

    def window_partition(self, x):
        """将图像分成窗口，返回[B, num_h, num_w, window_size^2, C]"""
        B, C, H, W = x.shape
        num_h = H // self.window_size
        num_w = W // self.window_size

        x = x.view(B, C, num_h, self.window_size, num_w, self.window_size)
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, num_h, num_w, window_size, window_size, C]
        windows = windows.view(B, num_h, num_w, self.window_size * self.window_size, C)
        return windows

    def window_unpartition(self, windows, H_pad, W_pad, B):
        """从窗口恢复图像"""
        num_h = H_pad // self.window_size
        num_w = W_pad // self.window_size

        windows = windows.view(B, num_h, num_w, self.window_size, self.window_size, -1)
        windows = windows.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, num_h, window_size, num_w, window_size]
        x = windows.view(B, -1, H_pad, W_pad)
        return x
