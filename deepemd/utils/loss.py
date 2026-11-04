import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss_b(nn.Module):
    def __init__(self, mode="loglike", sigma=25):
        super(CustomLoss_b, self).__init__()
        self.mode = mode
        self.sigma = sigma

    def forward(self, output, truth):
        if self.mode == "mse":
            loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
        elif self.mode == "loglike":
            eps = 1e-5
            channel = 3+6
            (N, C, H, W), T = truth.shape, truth.shape[1]//channel
            C = channel
            output, truth = output.reshape(N, T, C, H, W), truth.reshape(N, T, C, H, W)
            output, truth = output.permute(0, 2, 1, 3, 4), truth.permute(0, 2, 1, 3, 4)
            N, C, T, H, W = truth.shape
            mean = output[0:N, 0:C, 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)
            var = output[0:N, C:C+int(C*(C+1)/2), 0:T, 0:H, 0:W].permute(0, 2, 3, 4, 1)
            truth = truth.permute(0, 2, 3, 4, 1).reshape(N, T, H, W, C, 1)

            ax = torch.zeros(N, T, H, W, int(C*C)).to(output.device)
            I = torch.eye(C).reshape(1, 1, 1, 1, C, C).repeat(N, T, H, W, 1, 1).to(output.device)
            idx1 = 0
            for i in range(C):
                idx2 = idx1 + C - i
                ax[0:N, 0:T, 0:H, 0:W, int(i * C):int(i * C) + C - i] = var[0:N, 0:T, 0:H, 0:W, idx1:idx2]
                idx1 = idx2
            ax = ax.reshape(N, T, H, W, C, C)

            sigma2I = (((self.sigma**2) + eps) * I.permute(1, 2, 3, 4, 5, 0)).permute(5, 0, 1, 2, 3, 4)
            variance = torch.matmul(ax.transpose(4, 5), ax) + sigma2I  # (sigma**2)*I

            likelihood = 0.5 * torch.matmul(
                torch.matmul((truth - mean).transpose(4, 5), torch.inverse(variance)), (truth - mean)
            )
            likelihood = likelihood.reshape(N, T, H, W)
            likelihood += 0.5 * torch.log(torch.det(variance))
            loss = torch.mean(likelihood)

        return loss


class CustomLoss(nn.Module):
    def __init__(self, mode="loglike", sigma=25):
        super(CustomLoss, self).__init__()
        self.mode = mode
        self.sigma = sigma

    def forward(self, output, truth):
        if self.mode == "mse":
            loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
        elif self.mode == "loglike":
            eps = 1e-5
            N, C, H, W = truth.shape
            mean = output[0:N, 0:C, 0:H, 0:W].permute(0, 2, 3, 1).reshape(N, H, W, C, 1)
            var = output[0:N, C:C+int(C*(C+1)/2), 0:H, 0:W].permute(0, 2, 3, 1)
            truth = truth.permute(0, 2, 3, 1).reshape(N, H, W, C, 1)

            ax = torch.zeros(N, H, W, int(C*C)).to(output.device)
            I = torch.eye(C).reshape(1, 1, 1, C, C).repeat(N, H, W, 1, 1).to(output.device)
            idx1 = 0
            for i in range(C):
                idx2 = idx1 + C - i
                ax[0:N, 0:H, 0:W, int(i * C):int(i * C) + C - i] = var[0:N, 0:H, 0:W, idx1:idx2]
                idx1 = idx2
            ax = ax.reshape(N, H, W, C, C)

            sigma2I = (((self.sigma**2) + eps) * I.permute(1, 2, 3, 4, 0)).permute(4, 0, 1, 2, 3)
            variance = torch.matmul(ax.transpose(3, 4), ax) + sigma2I  # (sigma**2)*I

            likelihood = 0.5 * torch.matmul(
                torch.matmul((truth - mean).transpose(3, 4), torch.inverse(variance)), (truth - mean)
            )
            likelihood = likelihood.reshape(N, H, W)
            likelihood += 0.5 * torch.log(torch.det(variance))
            loss = torch.mean(likelihood)

        return loss


def loss_function(output, truth, mode="loglike", sigma=30):
    if (mode == "mse"):
        loss = F.mse_loss(output, truth, reduction="sum") / (truth.size(0) * 2)
    elif (mode == "loglike"):
        eps = 1e-5
        N, C, H, W = truth.shape
        mean = output[0:N, 0:C, 0:H, 0:W].permute(0, 2, 3, 1).reshape(N, H, W, C, 1)
        var = output[0:N, C:C+int(C*(C+1)/2), 0:H, 0:W].permute(0, 2, 3, 1)
        truth = truth.permute(0, 2, 3, 1).reshape(N, H, W, C, 1)
        ax = torch.zeros(N, H, W, int(C*C)).to(output.device)
        I = torch.eye(C).reshape(1, 1, 1, C, C).repeat(N, H, W, 1, 1).to(output.device)
        idx1 = 0
        for i in range(C):
            idx2 = idx1 + C-i
            ax[0:N, 0:H, 0:W, int(i*C):int(i*C)+C-i] = var[0:N, 0:H, 0:W, idx1:idx2]
            idx1 = idx2
        ax = ax.reshape(N, H, W, C, C)
        sigma2I = (((sigma**2)+eps)*I.permute(1, 2, 3, 4, 0)).permute(4, 0, 1, 2, 3)
        variance = torch.matmul(ax.transpose(3, 4), ax) + sigma2I  # (sigma**2)*I
        likelihood = 0.5*torch.matmul(torch.matmul((truth-mean).transpose(3, 4), torch.inverse(variance)), (truth-mean))
        likelihood = likelihood.reshape(N, H, W)
        likelihood += 0.5*torch.log(torch.det(variance))
        loss = torch.mean(likelihood)
        # loss = torch.mean(likelihood.mean(dim=(1,2)) - 0.1*sigma)
    return loss


class CombinedLoss(nn.Module):
    """组合损失函数：MSE + SSIM + Total Variation

    Args:
        alpha: MSE损失权重 (默认1.0)
        beta: SSIM损失权重 (默认0.1)
        gamma: TV正则权重 (默认0.01)
        window_size: SSIM计算窗口大小 (默认11)
    """
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01, window_size=11):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.window_size = window_size
        self.mse_loss = nn.MSELoss()

    def gaussian_window(self, window_size, sigma=1.5):
        """生成高斯窗口用于SSIM计算"""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / (2 * sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, channels):
        """创建2D高斯窗口"""
        _1D_window = self.gaussian_window(self.window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channels, 1, self.window_size, self.window_size).contiguous()
        return window

    def ssim_loss(self, img1, img2):
        """计算SSIM损失"""
        if img1.dim() == 4:
            channels = img1.size(1)
        elif img1.dim() == 3:
            channels = 1
            img1 = img1.unsqueeze(1)
            img2 = img2.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected tensor dimensions: {img1.dim()}")

        window = self.create_window(channels).to(img1.device)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()

    def total_variation_loss(self, img):
        """计算Total Variation正则化损失"""
        if img.dim() == 4:
            batch_size = img.size(0)
        else:
            batch_size = 1
            img = img.unsqueeze(0)

        # 水平方向的TV
        tv_h = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        # 垂直方向的TV
        tv_v = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

        return tv_h + tv_v

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        ssim = self.ssim_loss(output, target)
        tv = self.total_variation_loss(output)

        total_loss = self.alpha * mse + self.beta * ssim + self.gamma * tv
        return total_loss
