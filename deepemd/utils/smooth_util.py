import torch


def smooth_out_withmask(out_avg,out,mask,exp_weight=0.99):
    # Smoothing
    if out_avg is None:
        # 避免除零错误：当mask全为0时，直接使用out
        denominator = 1 - mask
        # 将分母中的0替换为1，避免除零
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        out_avg = torch.div(torch.mul(out.detach(), 1-mask), denominator)
    else:
        out_avg=torch.where((~torch.isnan(out_avg)) & (mask==0),out.detach()*(1-exp_weight)+out_avg*exp_weight,out_avg)
        out_avg=torch.where(torch.isnan(out_avg) & (mask==0),out.detach(),out_avg)
    return out_avg

