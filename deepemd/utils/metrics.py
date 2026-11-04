import torch
import math
from skimage.metrics import structural_similarity


def test(noisy_img, clean_img):
    psnr = compare_psnr(clean_img, noisy_img)
    ssim = cal_ssim(clean_img, noisy_img)
    snr = compare_snr(clean_img, noisy_img)
    return psnr, ssim, snr


def test_batch(noisy_img, clean_img):
    psnr = []
    ssim = []
    snr = []
    batch_size = noisy_img.shape[0]
    step = 3 if batch_size >= 3 else batch_size
    for i in range(0, batch_size, step):
        clean_slice = clean_img[i:i+step]
        noisy_slice = noisy_img[i:i+step]
        if clean_slice.dim() == 4:
            clean_slice = clean_slice.squeeze(0)
        if noisy_slice.dim() == 4:
            noisy_slice = noisy_slice.squeeze(0)
        psnr.append(compare_psnr(clean_slice, noisy_slice))
        ssim.append(cal_ssim(clean_slice, noisy_slice))
        snr.append(compare_snr(clean_slice, noisy_slice))
    return psnr, ssim, snr

def compare_psnr(a, b):
    a = torch.clip(a, 0, 1)
    b = torch.clip(b, 0, 1)
    if a.dim() == 3:
        x = torch.mean((a - b) ** 2, dim=[-3, -2, -1])
    else:
        x = torch.mean((a - b) ** 2)
    return torch.mean(20 * torch.log(1 / torch.sqrt(x)) / math.log(10)).item()


def compare_snr(clean_img, denoised_img):
    clean_img = torch.clip(clean_img, 0, 1)
    denoised_img = torch.clip(denoised_img, 0, 1)
    
    noise = clean_img - denoised_img
    
    if clean_img.dim() == 3:
        signal_power = torch.mean(clean_img ** 2, dim=[-3, -2, -1])
        noise_power = torch.mean(noise ** 2, dim=[-3, -2, -1])
    else:
        signal_power = torch.mean(clean_img ** 2)
        noise_power = torch.mean(noise ** 2)
    
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return torch.mean(snr).item()


def cal_ssim(pack1, pack2):
    if pack1.dim() == 3:
        C, H, W = pack1.shape
        pack1 = pack1.detach().cpu().numpy().transpose(1, 2, 0)
        pack2 = pack2.detach().cpu().numpy().transpose(1, 2, 0)
        return structural_similarity(pack1, pack2, channel_axis=2, data_range=1)
    else:
        if pack1.dim() == 4:
            _, H, W = pack1.shape
        else:
            H, W = pack1.shape
        pack1 = pack1.detach().cpu().numpy()
        pack2 = pack2.detach().cpu().numpy()
        return structural_similarity(pack1, pack2, data_range=1)
