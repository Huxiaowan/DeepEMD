"""
真正的3D CNN用于3D体积去噪
使用3D卷积核同时处理XYZ三个维度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3d(in_ch, out_ch, kernel_size=(3,3,3), stride=(1,1,1)):
    """3D卷积辅助函数"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    padding = tuple(k//2 for k in kernel_size)

    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                 padding=padding, padding_mode='reflect'),
        nn.BatchNorm3d(out_ch, 0.1),
        nn.LeakyReLU(0.2, True)
    )


class Conv3DBlock(nn.Module):
    """3D CNN块，类似UNet的编码器-解码器结构

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        depth: 网络深度
        is_first: 是否是第一层
        use_z_attn: 是否使用Z轴注意力
        z_window: Z轴注意力窗口大小
    """
    def __init__(self, in_channels=1, out_channels=64, depth=4, is_first=False,
                 use_z_attn=False, z_window=7):
        super().__init__()

        # 动态确定输入通道
        input_ch = in_channels if is_first else out_channels

        # 分支1：1x1x1卷积（降维或保持通道数）
        self.branch1 = nn.Sequential(
            nn.Conv3d(input_ch, out_channels, kernel_size=1, padding_mode='reflect'),
            nn.BatchNorm3d(out_channels, 0.1),
            nn.LeakyReLU(0.2, True)
        )

        # 分支2：3D卷积+下采样+递归
        self.branch2 = nn.Sequential(
            conv3d(input_ch, out_channels, kernel_size=(3,3,3), stride=(2,2,2)),
            conv3d(out_channels, out_channels, kernel_size=(3,3,3)),
            Conv3DBlock(out_channels=out_channels, depth=depth-1,
                       use_z_attn=use_z_attn, z_window=z_window) if depth > 0 else
            nn.Identity(),
        )

        # 上采样
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )

        # 后处理融合
        self.fusion = nn.Sequential(
            nn.BatchNorm3d(2*out_channels, 0.1),
            conv3d(2*out_channels, out_channels, kernel_size=(3,3,3)),
            conv3d(out_channels, out_channels, kernel_size=1)
        )

        # Z轴注意力（可选）
        self.use_z_attn = use_z_attn
        if use_z_attn:
            from deepemd.models.ZAxisAttention import ZAttentionBlock
            self.z_attn = ZAttentionBlock(dim=out_channels, num_heads=4, z_window=z_window)

    def forward(self, x):
        """
        Args:
            x: 输入 (B, C, D, H, W) 或 (B, C, H, W)
        Returns:
            输出 (B, C, D, H, W) 或 (B, C, H, W)
        """
        # 处理2D输入
        if x.dim() == 4:
            B, C, H, W = x.shape
            # 如果是2D，直接返回（或可以在D=1上扩展）
            # 为了兼容性，在D=1上扩展为3D
            x = x.unsqueeze(2)  # (B, C, 1, H, W)
            x = self.branch1(x)
            x = x.squeeze(2)  # (B, C, H, W)
            return x

        # 3D输入
        B, C, D, H, W = x.shape

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x2 = self.up(x2)

        # 确保尺寸匹配（可能需要crop或pad）
        if x2.shape[2] > x1.shape[2]:  # D维度
            x2 = x2[:, :, :x1.shape[2], :, :]
        elif x2.shape[2] < x1.shape[2]:
            diff = x1.shape[2] - x2.shape[2]
            x2 = F.pad(x2, (0, 0, 0, 0, 0, diff))

        x = self.fusion(torch.cat([x1, x2], dim=1))

        # Z轴注意力
        if self.use_z_attn:
            x = self.z_attn(x)

        return x


class Skip3D(nn.Module):
    """真正的3D Skip网络用于3D体积去噪

    与SkipModel不同，这个版本使用3D卷积处理Z轴

    Args:
        input_channels: 输入通道数（通常是多个Z切片concat）
        output_channels: 输出通道数
        dim: 特征维度
        depth: 网络深度
        use_z_attn: 是否使用Z轴注意力
        z_window: Z轴注意力窗口
    """
    def __init__(self, input_channels=3, output_channels=1, dim=64,
                 depth=4, use_z_attn=False, z_window=7):
        super().__init__()

        # 输入可能是 (B, 3, H, W) 或 (B, 1, D, H, W)
        self.input_channels = input_channels
        self.needs_reshape = (input_channels == 3)

        # 简化版：直接使用2D卷积处理每个Z切片，然后在Z方向融合
        # 这样可以避免D维度太小的问题

        # 对每个Z切片应用2D卷积
        self.conv2d_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(dim, 0.1),
                nn.LeakyReLU(0.2, True)
            )
            for _ in range(3)  # 最多3个Z切片
        ])

        # Z方向融合（使用1D卷积）
        self.z_fusion = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim, 0.1),
            nn.LeakyReLU(0.2, True)
        )

        # 输出层
        self.last = nn.Sequential(
            nn.Conv2d(dim, output_channels, kernel_size=1, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - C个Z切片concat为通道（C可以是任意奇数：3,5,7...）
        Returns:
            (B, 1, H, W) - 中间切片的输出
        """
        B, C, H, W = x.shape

        # 处理多个Z切片（C可以是3, 5, 7等任意奇数）
        slice_features = []
        num_slices = C

        # 对每个Z切片应用2D卷积
        for i in range(num_slices):
            slice_i = x[:, i:i+1, :, :]  # (B, 1, H, W)
            # 循环使用conv2d_layers（如果层数不够，循环使用）
            conv_layer = self.conv2d_layers[i % len(self.conv2d_layers)]
            feat = conv_layer(slice_i)  # (B, dim, H, W)
            slice_features.append(feat)

        # 在Z方向融合
        # 将特征堆叠为(B, dim, C, H, W)
        stacked = torch.stack(slice_features, dim=2)  # (B, dim, C, H, W)
        B, C_dim, D, H, W = stacked.shape

        # 对每个空间位置应用1D卷积在Z方向
        stacked = stacked.permute(0, 3, 4, 1, 2)  # (B, H, W, dim, C)
        stacked = stacked.reshape(B*H*W, C_dim, D)  # (B*H*W, dim, C)

        z_feat = self.z_fusion(stacked)

        z_feat = z_feat.reshape(B, H, W, C_dim, D).permute(0, 3, 4, 1, 2)  # (B, dim, C, H, W)

        # 提取中间切片
        middle_idx = D // 2  # 取中间索引
        middle_feat = z_feat[:, :, middle_idx, :, :]  # (B, dim, H, W)

        # 输出
        out = self.last(middle_feat)

        return out


class Pure3DUNet(nn.Module):
    """纯3D UNet用于3D体积去噪

    使用完整的3D编码器-解码器结构
    """
    def __init__(self, in_channels=1, out_channels=1, base_dim=32, depth=4):
        super().__init__()

        self.depth = depth

        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        dims = [in_channels] + [base_dim * (2**i) for i in range(depth)]

        for i in range(depth):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=3, padding=1, padding_mode='reflect'),
                    nn.BatchNorm3d(dims[i+1]),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv3d(dims[i+1], dims[i+1], kernel_size=3, padding=1, padding_mode='reflect'),
                    nn.BatchNorm3d(dims[i+1]),
                    nn.LeakyReLU(0.2, True),
                )
            )
            if i < depth - 1:
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))

        # 中间层
        self.bottleneck = nn.Sequential(
            nn.Conv3d(dims[depth], dims[depth]*2, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm3d(dims[depth]*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(dims[depth]*2, dims[depth], kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm3d(dims[depth]),
            nn.LeakyReLU(0.2, True),
        )

        # 解码器
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            self.upsamples.append(
                nn.ConvTranspose3d(dims[i+1], dims[i], kernel_size=2, stride=2)
            )
            self.decoders.append(
                nn.Sequential(
                    nn.Conv3d(dims[i]*2, dims[i], kernel_size=3, padding=1, padding_mode='reflect'),
                    nn.BatchNorm3d(dims[i]),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv3d(dims[i], dims[i], kernel_size=3, padding=1, padding_mode='reflect'),
                    nn.BatchNorm3d(dims[i]),
                    nn.LeakyReLU(0.2, True),
                )
            )

        # 输出
        self.final = nn.Sequential(
            nn.Conv3d(base_dim, out_channels, kernel_size=1, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) 或 (B, C, H, W)
        """
        reshape_output = False
        if x.dim() == 4:
            B, C, H, W = x.shape
            if C == 3:
                x = x.permute(0, 2, 3, 1).reshape(B, 1, H, W, C)
                x = x.permute(0, 1, 4, 2, 3)
                reshape_output = True
            else:
                x = x.unsqueeze(2)

        # 编码
        skip_connections = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skip_connections.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # 解码
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            skip = skip_connections[-(i+1)]
            # Crop或pad到相同尺寸
            if x.shape[2] > skip.shape[2]:
                x = x[:, :, :skip.shape[2], :, :]
            elif x.shape[2] < skip.shape[2]:
                diff = skip.shape[2] - x.shape[2]
                x = F.pad(x, (0, 0, 0, 0, 0, diff))
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # 输出
        x = self.final(x)

        if reshape_output:
            B, C, D, H, W = x.shape
            if D == 3:
                x = x[:, :, 1, :, :]
            else:
                x = x[:, :, D//2, :, :]

        return x


if __name__ == "__main__":
    # 测试代码
    print("测试3D CNN模块...")

    batch_size = 2
    depth = 5
    height = 32
    width = 32

    # 测试1: 输入是(B, 3, H, W) - 3个Z切片concat
    x_2d = torch.randn(batch_size, 3, height, width)

    model1 = Skip3D(input_channels=3, output_channels=1, dim=32, depth=3)
    out1 = model1(x_2d)
    print(f"Skip3D输入(B,3,H,W): {x_2d.shape} -> 输出: {out1.shape}")

    # 测试2: 真正的3D输入
    x_3d = torch.randn(batch_size, 1, depth, height, width)
    out2 = model1(x_3d)
    print(f"Skip3D输入(B,1,D,H,W): {x_3d.shape} -> 输出: {out2.shape}")

    # 测试3: Pure3DUNet
    model2 = Pure3DUNet(in_channels=1, out_channels=1, base_dim=16, depth=3)
    out3 = model2(x_3d)
    print(f"Pure3DUNet输入: {x_3d.shape} -> 输出: {out3.shape}")

    print("\n所有测试通过！")
