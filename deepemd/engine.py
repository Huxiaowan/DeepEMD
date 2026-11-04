import argparse
import numpy as np
import torch.nn.functional as torch_F
import random
from deepemd.utils.util import get_args, get_model
from deepemd.utils.metrics import test_batch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from deepemd.utils.smooth_util import smooth_out_withmask
import torch.optim as optim
import torch
import time
import os
from tqdm import tqdm


if os.environ.get('DEEPEMD_REQUIRE_CUDA') == '1' and not torch.cuda.is_available():
    raise SystemExit('Selected CUDA GPU is unavailable. Check PyTorch/CUDA installation and --gpu.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_deterministic(seed=3407, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_multi_frame_tiff(frames, output_path, is_16bit=False):
    """保存多个帧为单个多帧TIFF文件

    Args:
        frames: tensor列表，每个元素是(H, W)或(1, H, W)的tensor (值在 [0, 1])
        output_path: 输出文件路径
        is_16bit: 是否以 16-bit 保存
    """
    from PIL import Image
    import numpy as np

    pil_images = []
    for f in frames:
        if f.dim() == 3 and f.size(0) == 1:
            f = f.squeeze(0)
        arr = f.cpu().numpy()
        arr = np.clip(arr, 0.0, 1.0)
        
        if is_16bit:
            arr = (arr * 65535.0).astype(np.uint16)
            img = Image.fromarray(arr, mode='I;16')
        else:
            arr = (arr * 255.0).astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
            
        pil_images.append(img)

    # 保存所有帧到单个TIFF文件
    first_frame = pil_images[0]
    first_frame.save(output_path, save_all=True, append_images=pil_images[1:], format='TIFF')
    print(f"已保存多帧TIFF: {output_path} (共{len(frames)}帧, 16bit={is_16bit})")


def pad_temporal_replicate(frames_tensor, pad_frames):
    if pad_frames <= 0:
        return frames_tensor
    left = frames_tensor[:1].repeat(pad_frames, 1, 1, 1)
    right = frames_tensor[-1:].repeat(pad_frames, 1, 1, 1)
    return torch.cat([left, frames_tensor, right], dim=0)


def pd_pack(x, scale):
    if scale <= 1:
        return x
    B, C, H, W = x.shape
    if H % scale != 0 or W % scale != 0:
        raise ValueError(f"pd_pack requires H,W divisible by scale, got {(H, W)} scale={scale}")
    xs = []
    for dy in range(scale):
        for dx in range(scale):
            xs.append(x[:, :, dy::scale, dx::scale])
    return torch.cat(xs, dim=0)


def pd_unpack(x_pd, scale, batch_size):
    if scale <= 1:
        return x_pd
    Bp, C, Hs, Ws = x_pd.shape
    if Bp != batch_size * scale * scale:
        raise ValueError(f"pd_unpack got batch {Bp}, expected {batch_size * scale * scale}")
    chunks = torch.chunk(x_pd, scale * scale, dim=0)
    out = torch.empty((batch_size, C, Hs * scale, Ws * scale), device=x_pd.device, dtype=x_pd.dtype)
    idx = 0
    for dy in range(scale):
        for dx in range(scale):
            out[:, :, dy::scale, dx::scale] = chunks[idx]
            idx += 1
    return out

def _make_blend_window(tile_h, tile_w, device, dtype):
    """2D Hann window for smooth tile blending (avoids seams)."""
    wy = torch.hann_window(tile_h, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    wx = torch.hann_window(tile_w, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    return torch.outer(wy, wx).view(1, 1, tile_h, tile_w)


def _tile_starts(extent, tile, stride):
    """Overlapping start positions covering [0, extent); last tile snaps to edge."""
    if extent <= tile:
        return [0]
    starts = list(range(0, extent - tile + 1, stride))
    if starts[-1] != extent - tile:
        starts.append(extent - tile)
    return starts


def tiled_infer(net_fill, img, tile_size, overlap, ensemble_k=16):
    """Blind-spot tiled inference: stitch overlapping spatial tiles, and within
    each tile ensemble-average the network's predictions ONLY at masked
    (blind-spot) locations across `ensemble_k` masked passes.

    This mirrors the out_avg estimator (predictions accumulated where mask==0),
    which is the validated DeepEMD output mechanism — running the blind-spot
    network with the mask disabled instead collapses to ~identity (no denoising).

    img: (1, C, H, W) -> denoised (1, C, H, W).
    """
    was_training = net_fill.training
    net_fill.train()  # keep mask generation active (forward masks when net.training)
    if hasattr(net_fill, 'set_mask_context'):
        net_fill.set_mask_context('train')  # use train_mask_ratio (validated ratio)
    if hasattr(net_fill, 'set_mask_enabled'):
        net_fill.set_mask_enabled(True)

    B, C, H, W = img.shape
    ts_h = min(tile_size, H)
    ts_w = min(tile_size, W)
    stride_h = max(1, int(ts_h * (1 - overlap)))
    stride_w = max(1, int(ts_w * (1 - overlap)))
    ys = _tile_starts(H, ts_h, stride_h)
    xs = _tile_starts(W, ts_w, stride_w)

    out_acc = None
    wgt_acc = None
    win = _make_blend_window(ts_h, ts_w, img.device, img.dtype)

    with torch.no_grad():
        for ty in ys:
            for tx in xs:
                tile = img[:, :, ty:ty + ts_h, tx:tx + ts_w]
                # Blind-spot ensemble within the tile
                t_acc = None
                t_cnt = None
                for _ in range(max(1, ensemble_k)):
                    out_tile = net_fill(tile)
                    m = net_fill.get_mask()
                    out_c = out_tile.size(1)
                    if t_acc is None:
                        t_acc = torch.zeros_like(out_tile)
                        t_cnt = torch.zeros_like(out_tile)
                    if m.size(1) == 1 and out_c > 1:
                        m = m.expand(-1, out_c, -1, -1)
                    elif m.size(1) != out_c:
                        if out_c == 1:
                            mid_idx = m.size(1) // 2
                            m = m[:, mid_idx:mid_idx + 1]
                        else:
                            m = m[:, :out_c]
                    blind = (1.0 - m)  # 1 where pixel was masked (predicted)
                    t_acc = t_acc + out_tile * blind
                    t_cnt = t_cnt + blind
                # blind-spot mean; fall back to input where never masked
                if out_tile.size(1) == tile.size(1):
                    fallback = tile
                elif out_tile.size(1) == 1:
                    mid_idx = tile.size(1) // 2
                    fallback = tile[:, mid_idx:mid_idx + 1]
                else:
                    fallback = out_tile
                tile_out = torch.where(t_cnt > 0, t_acc / t_cnt.clamp_min(1e-8), fallback)
                if out_acc is None:
                    out_acc = torch.zeros((B, tile_out.size(1), H, W), device=img.device, dtype=img.dtype)
                    wgt_acc = torch.zeros((B, 1, H, W), device=img.device, dtype=img.dtype)
                out_acc[:, :, ty:ty + ts_h, tx:tx + ts_w] += tile_out * win
                wgt_acc[:, :, ty:ty + ts_h, tx:tx + ts_w] += win

    out = out_acc / wgt_acc.clamp_min(1e-8)

    if not was_training:
        net_fill.eval()
    return out


def train(args, data, net_fill, optimizer_fill, criterion_fill, original_h, original_w, padded_h, padded_w):
    noisy_input = data['img_noisy_torch']

    # 处理输入维度：确保是4维(B, C, H, W)
    if noisy_input.dim() == 4 and noisy_input.size(0) == 1:
        img_noisy_torch = noisy_input
    elif noisy_input.dim() == 4:
        # 可能是(3, 1, H, W)单通道3帧，需要转为(3, H, W)
        if noisy_input.size(1) == 1:
            img_noisy_torch = noisy_input.squeeze(1).unsqueeze(0)  # (3,1,H,W) -> (3,H,W) -> (1,3,H,W)
        else:
            img_noisy_torch = noisy_input.unsqueeze(0)
    else:
        img_noisy_torch = noisy_input.unsqueeze(0)

    # 提取中间帧用于loss计算（参考0521版本line 51）
    C = img_noisy_torch.size(1)
    if C == 3:  # 单通道3帧：(1, 3, H, W)
        img_noisy_for_loss = img_noisy_torch[:, 1:2]  # 取中间帧：(1, 1, H, W)
        # GT也取中间帧
        gt_input = data['img_gt_torch']
        if gt_input.dim() == 4 and gt_input.size(1) == 1:
            img_gt_torch = gt_input[1:2]  # (3,1,H,W) -> (1,1,H,W)
        else:
            img_gt_torch = gt_input[1:2]
    else:  # 多通道：(1, C, H, W) = C个单通道帧（C可以是5, 7, 9等）
        # 取中间帧
        mid_idx = C // 2
        img_noisy_for_loss = img_noisy_torch[:, mid_idx:mid_idx+1]  # (1, C, H, W) -> (1, 1, H, W)
        gt_input = data['img_gt_torch']
        if gt_input.dim() == 4 and gt_input.size(1) == 1:
            img_gt_torch = gt_input[mid_idx:mid_idx+1]  # (C,1,H,W) -> (1,1,H,W)
        else:
            img_gt_torch = gt_input[mid_idx:mid_idx+1]

    out_avg = img_noisy_torch.clone().detach() if args.load_initial == 'noisy' else None

    pd_scale = int(getattr(args, 'pd_scale', 1) or 1)
    if pd_scale > 1:
        base_batch = img_noisy_torch.size(0)
        img_noisy_torch = pd_pack(img_noisy_torch, pd_scale)
        img_noisy_for_loss = pd_pack(img_noisy_for_loss, pd_scale)
        if out_avg is not None:
            out_avg = pd_pack(out_avg, pd_scale)

    # Spatial tiling: train on random patches (fixed cost) + tiled inference,
    # so DeepEMD scales to large frames like SUPPORT / SRDTrans.
    tile_size = int(getattr(args, 'tile_size', 0) or 0)
    tile_overlap = getattr(args, 'tile_overlap', None)
    tile_overlap = 0.5 if tile_overlap is None else float(tile_overlap)
    cur_h, cur_w = img_noisy_torch.shape[-2:]
    tiling = (tile_size > 0 and pd_scale <= 1 and (cur_h > tile_size or cur_w > tile_size))
    if tiling:
        # snap tile to multiple of 64 (network downsample) and clamp to frame
        ts = max(64, (tile_size // 64) * 64)
        ts = min(ts, cur_h, cur_w)
    else:
        ts = 0

    start = time.time()
    refine_num_iter = int(getattr(args, 'refine_num_iter', 0) or 0)
    stages = [('main', int(args.num_iter))]
    if refine_num_iter > 0:
        stages.append(('refine', refine_num_iter))

    for stage_name, num_iter in stages:
        if stage_name == 'refine':
            if hasattr(net_fill, 'mask_type'):
                net_fill.mask_type = getattr(args, 'refine_mask_type', net_fill.mask_type)
            if hasattr(net_fill, 'mask_replace'):
                net_fill.mask_replace = getattr(args, 'refine_mask_replace', net_fill.mask_replace)
            if hasattr(net_fill, 'mask_fixed'):
                net_fill.mask_fixed = bool(getattr(args, 'refine_mask_fixed', net_fill.mask_fixed))
            if hasattr(net_fill, 'mask_block_size'):
                net_fill.mask_block_size = int(getattr(args, 'refine_mask_block_size', net_fill.mask_block_size) or net_fill.mask_block_size)
            if hasattr(net_fill, 'mask_center_only'):
                net_fill.mask_center_only = bool(getattr(args, 'refine_mask_center_only', net_fill.mask_center_only))
            if hasattr(net_fill, 'mask_share_across_channels'):
                net_fill.mask_share_across_channels = bool(getattr(args, 'refine_mask_share_across_channels', net_fill.mask_share_across_channels))
            if hasattr(net_fill, 'train_mask_ratio'):
                refine_train_ratio = getattr(args, 'refine_train_mask_ratio', None)
                if refine_train_ratio is None:
                    refine_train_ratio = getattr(args, 'train_mask_ratio', None)
                if refine_train_ratio is not None:
                    net_fill.train_mask_ratio = refine_train_ratio

        if hasattr(net_fill, 'infer_mask_ratio') and hasattr(args, 'infer_mask_ratio'):
            net_fill.infer_mask_ratio = getattr(args, 'infer_mask_ratio', net_fill.infer_mask_ratio)

        if hasattr(net_fill, 'set_mask_context'):
            net_fill.set_mask_context('train')
        if hasattr(net_fill, 'set_mask_enabled'):
            key = 'refine_mask_apply_train' if stage_name == 'refine' else 'mask_apply_train'
            net_fill.set_mask_enabled(bool(getattr(args, key, True)))

        for _ in range(num_iter):
            if tiling:
                # random spatial patch -> training cost independent of frame size
                ty = random.randint(0, cur_h - ts)
                tx = random.randint(0, cur_w - ts)
                cur_in = img_noisy_torch[:, :, ty:ty + ts, tx:tx + ts]
                cur_loss_target = img_noisy_for_loss[:, :, ty:ty + ts, tx:tx + ts]
            else:
                cur_in = img_noisy_torch
                cur_loss_target = img_noisy_for_loss

            out = net_fill(cur_in)  # 输入全部帧
            mask = net_fill.get_mask()
            if out.size(1) == 1:
                out_for_loss = out
            else:
                out_mid_idx = out.size(1) // 2
                out_for_loss = out[:, out_mid_idx:out_mid_idx+1]

            if mask.size(1) == 1:
                mask_for_loss = mask
            else:
                mask_mid_idx = mask.size(1) // 2
                mask_for_loss = mask[:, mask_mid_idx:mask_mid_idx+1]

            if args.is_smooth and not tiling:
                out_avg = smooth_out_withmask(out_avg, out, mask, exp_weight=args.exp_weight)

            optimizer_fill.zero_grad()
            loss_fill = criterion_fill(torch.mul(out_for_loss, 1-mask_for_loss), torch.mul(cur_loss_target, 1-mask_for_loss))
            loss_fill.backward()
            optimizer_fill.step()

    end = time.time()

    use_infer_output = bool(getattr(args, 'use_infer_output', False))
    infer_ensemble_k = int(getattr(args, 'infer_ensemble_k', 1) or 1)
    mask_apply_infer = bool(getattr(args, 'mask_apply_infer', False))
    if (getattr(args, 'infer_mask_ratio', None) is not None) or (infer_ensemble_k > 1) or mask_apply_infer:
        use_infer_output = True

    out_final = out_avg
    if tiling:
        # out_avg is unused when tiling; produce output via blind-spot tiled inference
        tile_k = int(getattr(args, 'tile_infer_k', 16) or 16)
        out_final = tiled_infer(net_fill, img_noisy_torch, ts, tile_overlap, ensemble_k=tile_k)
    elif use_infer_output:
        was_training = net_fill.training
        net_fill.eval()
        if hasattr(net_fill, 'set_mask_context'):
            net_fill.set_mask_context('infer')
        if hasattr(net_fill, 'set_mask_enabled'):
            net_fill.set_mask_enabled(mask_apply_infer)
        outs = []
        with torch.no_grad():
            for _ in range(infer_ensemble_k):
                outs.append(net_fill(img_noisy_torch))
        out_final = torch.mean(torch.stack(outs, dim=0), dim=0)
        if was_training:
            net_fill.train()
        if hasattr(net_fill, 'set_mask_context'):
            net_fill.set_mask_context('train')
        if hasattr(net_fill, 'set_mask_enabled'):
            net_fill.set_mask_enabled(bool(getattr(args, 'mask_apply_train', True)))

    if pd_scale > 1:
        out_final = pd_unpack(out_final, pd_scale, base_batch)

    # Crop to original dimensions and extract middle frame for metrics
    denoised_cropped = out_final[:, :, :original_h, :original_w]
    img_gt_torch_cropped = img_gt_torch[:, :, :original_h, :original_w]

    # Extract middle frame for comparison (GT is already middle frame)
    # out_avg可能是1通道(Skip3D)或3通道或更多通道
    if denoised_cropped.size(1) == 1:  # Skip3D等输出1通道
        denoised = denoised_cropped  # 直接使用
    elif denoised_cropped.size(1) == 3:  # single-channel 3 frames: extract middle
        denoised = denoised_cropped[:, 1:2]  # (1,3,H,W) -> (1,1,H,W)
    else:  # multi-channel (5, 7, 9等): extract middle frame
        mid_idx = denoised_cropped.size(1) // 2
        denoised = denoised_cropped[:, mid_idx:mid_idx+1]  # 提取中间帧

    if getattr(args, '_has_gt', True):
        psnr, ssim, snr = test_batch(denoised.detach(), img_gt_torch_cropped.detach())
    else:
        psnr, ssim, snr = [float('nan')], [float('nan')], [float('nan')]
    return psnr, ssim, snr, out_final[0].detach(), denoised.detach(), end-start


def pad_to_multiple(image, multiple=64):
    h, w = image.shape[-2:]
    m = int(multiple)
    pad_h = (m - h % m) % m
    pad_w = (m - w % m) % m
    # 使用 constant (0) 填充，防止自监督训练中的边缘信息泄露（恒等映射风险）
    return torch_F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)


def _prod(values):
    out = 1
    for value in values:
        out *= int(value)
    return out


def _array_to_tcyx(arr, axes=None):
    """Convert a TIFF series array to (T, C, Y, X)."""
    arr = np.asarray(arr)
    axes = axes.upper() if axes and len(axes) == arr.ndim else None

    if axes and "Y" in axes and "X" in axes:
        y_old = axes.index("Y")
        x_old = axes.index("X")
        frame_labels = {"T", "Z", "Q", "I"}
        channel_labels = {"C", "S"}

        frame_old = [
            i for i, label in enumerate(axes)
            if label in frame_labels and arr.shape[i] > 1
        ]
        channel_old = [
            i for i, label in enumerate(axes)
            if label in channel_labels and arr.shape[i] > 1
        ]
        used = set(frame_old + channel_old + [y_old, x_old])
        other_old = [i for i in range(arr.ndim) if i not in used and arr.shape[i] > 1]
        frame_old.extend(other_old)
        keep = set(frame_old + channel_old + [y_old, x_old])

        drop = [i for i in range(arr.ndim) if i not in keep and arr.shape[i] == 1]
        if drop:
            arr = np.squeeze(arr, axis=tuple(drop))

            def remap(index):
                return index - sum(1 for d in drop if d < index)

            frame_axes = [remap(i) for i in frame_old]
            channel_axes = [remap(i) for i in channel_old]
            y_axis = remap(y_old)
            x_axis = remap(x_old)
        else:
            frame_axes = frame_old
            channel_axes = channel_old
            y_axis = y_old
            x_axis = x_old

        perm = frame_axes + channel_axes + [y_axis, x_axis]
        # Any non-singleton axis not identified above is safest as a frame axis.
        extra = [i for i in range(arr.ndim) if i not in perm and arr.shape[i] > 1]
        if extra:
            frame_axes.extend(extra)
            perm = frame_axes + channel_axes + [y_axis, x_axis]
        singleton_extra = [i for i in range(arr.ndim) if i not in perm and arr.shape[i] == 1]
        if singleton_extra:
            arr = np.squeeze(arr, axis=tuple(singleton_extra))

            def remap2(index):
                return index - sum(1 for d in singleton_extra if d < index)

            frame_axes = [remap2(i) for i in frame_axes]
            channel_axes = [remap2(i) for i in channel_axes]
            y_axis = remap2(y_axis)
            x_axis = remap2(x_axis)
            perm = frame_axes + channel_axes + [y_axis, x_axis]

        arr = np.transpose(arr, perm)
        t = _prod(arr.shape[:len(frame_axes)]) if frame_axes else 1
        c0 = len(frame_axes)
        c1 = c0 + len(channel_axes)
        c = _prod(arr.shape[c0:c1]) if channel_axes else 1
        h, w = arr.shape[-2:]
        return arr.reshape(t, c, h, w)

    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr[None, None, :, :]
    if arr.ndim == 3:
        # H,Y,C / Y,X,S style single image; otherwise assume grayscale stack.
        if arr.shape[-1] <= 4 and arr.shape[0] > 16 and arr.shape[1] > 16:
            return arr.transpose(2, 0, 1)[None, :, :, :]
        return arr[:, None, :, :]
    if arr.ndim == 4:
        # T,Y,X,C
        if arr.shape[-1] <= 4 and arr.shape[1] > 16 and arr.shape[2] > 16:
            return arr.transpose(0, 3, 1, 2)
        # T,C,Y,X
        if arr.shape[1] <= 4:
            return arr
        # C,T,Y,X
        if arr.shape[0] <= 4 and arr.shape[2] > 16 and arr.shape[3] > 16:
            return arr.transpose(1, 0, 2, 3)
        h, w = arr.shape[-2:]
        return arr.reshape(_prod(arr.shape[:-2]), 1, h, w)
    if arr.ndim > 4:
        h, w = arr.shape[-2:]
        return arr.reshape(_prod(arr.shape[:-2]), 1, h, w)
    raise ValueError(f"Unsupported TIFF array shape: {arr.shape}")


def _normalize_tcyx(arr):
    dtype = arr.dtype
    is_16bit = bool(np.issubdtype(dtype, np.integer) and np.iinfo(dtype).bits > 8)
    arr = arr.astype(np.float32, copy=False)

    if np.issubdtype(dtype, np.integer):
        denom = float(np.iinfo(dtype).max)
        if denom <= 0:
            denom = 1.0
        arr = arr / denom
    elif np.issubdtype(dtype, np.bool_):
        arr = arr.astype(np.float32)
    else:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            mn = float(finite.min())
            mx = float(finite.max())
            if mn < 0.0 or mx > 1.0:
                if mx > mn:
                    arr = (arr - mn) / (mx - mn)
                else:
                    arr = np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(arr)), is_16bit


def _selected_range(frame_count, frame_start=0, frame_end=None, max_frames=0):
    start = int(frame_start or 0)
    start = max(0, min(start, int(frame_count)))
    if frame_end is None or int(frame_end) < 0:
        end = int(frame_count)
    else:
        end = max(start, min(int(frame_end), int(frame_count)))
    max_frames = int(max_frames or 0)
    if max_frames > 0:
        end = min(end, start + max_frames)
    if end <= start:
        end = min(int(frame_count), start + 1)
    return start, end


def _series_frame_count(shape, axes=None):
    axes = axes.upper() if axes and len(axes) == len(shape) else None
    if axes and "Y" in axes and "X" in axes:
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        frame_labels = {"T", "Z", "Q", "I"}
        channel_labels = {"C", "S"}
        frame_dims = [
            int(size)
            for i, (label, size) in enumerate(zip(axes, shape))
            if i not in {y_axis, x_axis}
            and label not in channel_labels
            and (label in frame_labels or int(size) > 1)
        ]
        return _prod(frame_dims) if frame_dims else 1
    if len(shape) <= 2:
        return 1
    if len(shape) == 3 and not (shape[-1] <= 4 and shape[0] > 16 and shape[1] > 16):
        return int(shape[0])
    return _prod(shape[:-2])


def _can_read_selected_pages(shape, axes, page_count):
    axes = axes.upper() if axes and len(axes) == len(shape) else None
    if len(shape) != 3:
        return False
    if page_count < int(shape[0]):
        return False
    if axes is None:
        return not (shape[-1] <= 4 and shape[0] > 16 and shape[1] > 16)
    return axes in {"ZYX", "TYX", "QYX", "IYX"}


def _load_tiff_frames_tifffile(tiff_path, frame_start=0, frame_end=None, max_frames=0):
    try:
        import tifffile
    except ImportError:
        return None

    try:
        with tifffile.TiffFile(tiff_path) as tif:
            series = tif.series[0]
            axes = getattr(series, "axes", None)
            frame_count = _series_frame_count(tuple(series.shape), axes)
            start, end = _selected_range(frame_count, frame_start, frame_end, max_frames)
            print(
                f"tifffile准备读取: shape={series.shape}, axes={axes}, "
                f"frames={start}:{end}/{frame_count}",
                flush=True,
            )

            if _can_read_selected_pages(tuple(series.shape), axes, len(tif.pages)):
                frames = []
                for page_idx in range(start, end):
                    if page_idx == start or page_idx == end - 1 or (page_idx - start) % 10 == 0:
                        print(f"tifffile逐页读取: page={page_idx} ({page_idx - start + 1}/{end - start})", flush=True)
                    page = np.asarray(tif.pages[page_idx].asarray())
                    page = np.squeeze(page)
                    if page.ndim != 2:
                        page = _array_to_tcyx(page)[0, 0]
                    frames.append(page)
                arr = np.stack(frames, axis=0)[:, None, :, :]
                print(f"tifffile按页读取完成: shape={arr.shape}, axes=TCYX", flush=True)
                return _normalize_tcyx(arr)

            arr = series.asarray()
            print(f"tifffile读取完成: shape={arr.shape}, axes={axes}", flush=True)
    except Exception as exc:
        print(f"tifffile读取失败，回退到PIL: {exc}", flush=True)
        return None

    tcyx = _array_to_tcyx(arr, axes=axes)
    start, end = _selected_range(tcyx.shape[0], frame_start, frame_end, max_frames)
    tcyx = tcyx[start:end]
    return _normalize_tcyx(tcyx)


def _load_tiff_frames_pil(tiff_path, frame_start=0, frame_end=None, max_frames=0):
    tiff = Image.open(tiff_path)
    frames = []
    is_16bit = False

    if tiff.mode == 'I;16' or tiff.mode == 'I':
        is_16bit = True

    start, end = _selected_range(tiff.n_frames, frame_start, frame_end, max_frames)
    print(f"PIL读取TIFF帧: frames={start}:{end}/{tiff.n_frames}", flush=True)

    for i in range(start, end):
        tiff.seek(i)
        frame = tiff.copy()

        if is_16bit:
            arr = np.array(frame).astype(np.float32) / 65535.0
            if arr.ndim == 2:
                arr = arr[None, ...]
            else:
                arr = arr.transpose(2, 0, 1)
            frame_tensor = torch.from_numpy(arr)
        else:
            frame_tensor = to_tensor(frame)

        frames.append(frame_tensor)

    return torch.stack(frames), is_16bit


def load_tiff_frames(tiff_path, frame_start=0, frame_end=None, max_frames=0):
    """加载TIFF图像栈，优先用 tifffile 识别 OME/ImageJ/BigTIFF 的真实帧维。"""
    loaded = _load_tiff_frames_tifffile(tiff_path, frame_start, frame_end, max_frames)
    if loaded is not None:
        return loaded
    return _load_tiff_frames_pil(tiff_path, frame_start, frame_end, max_frames)


def _series_shape_to_tcyx_meta(shape, axes=None):
    axes = axes.upper() if axes and len(axes) == len(shape) else None
    if axes and "Y" in axes and "X" in axes:
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        h = int(shape[y_axis])
        w = int(shape[x_axis])
        c = _prod(
            int(size)
            for label, size in zip(axes, shape)
            if label in {"C", "S"} and int(size) > 1
        )
        if c <= 0:
            c = 1
        t = _series_frame_count(shape, axes)
        return t, c, h, w

    if len(shape) == 2:
        return 1, 1, int(shape[0]), int(shape[1])
    if len(shape) == 3:
        if shape[-1] <= 4 and shape[0] > 16 and shape[1] > 16:
            return 1, int(shape[-1]), int(shape[0]), int(shape[1])
        return int(shape[0]), 1, int(shape[1]), int(shape[2])
    if len(shape) == 4:
        if shape[-1] <= 4 and shape[1] > 16 and shape[2] > 16:
            return int(shape[0]), int(shape[-1]), int(shape[1]), int(shape[2])
        if shape[1] <= 4:
            return int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
        if shape[0] <= 4 and shape[2] > 16 and shape[3] > 16:
            return int(shape[1]), int(shape[0]), int(shape[2]), int(shape[3])
    return _prod(shape[:-2]), 1, int(shape[-2]), int(shape[-1])


def inspect_tiff_meta(tiff_path, frame_start=0, frame_end=None, max_frames=0):
    try:
        import tifffile
        with tifffile.TiffFile(tiff_path) as tif:
            series = tif.series[0]
            shape = tuple(int(x) for x in series.shape)
            axes = getattr(series, "axes", None)
            dtype = np.dtype(series.dtype)
            page_count = len(tif.pages)
    except Exception:
        with Image.open(tiff_path) as img:
            shape = (int(getattr(img, "n_frames", 1)), int(img.height), int(img.width))
            axes = "TYX"
            dtype = np.dtype(np.uint16 if img.mode in {"I;16", "I"} else np.uint8)
            page_count = int(getattr(img, "n_frames", 1))

    total_frames, channels, height, width = _series_shape_to_tcyx_meta(shape, axes)
    start, end = _selected_range(total_frames, frame_start, frame_end, max_frames)
    return {
        "shape": shape,
        "axes": axes,
        "dtype": dtype,
        "page_count": page_count,
        "total_frames": total_frames,
        "start": start,
        "end": end,
        "frames": end - start,
        "channels": channels,
        "height": height,
        "width": width,
    }


def should_stream_tiff(args, tiff_path, frame_start=0, frame_end=None, max_frames=0):
    meta = inspect_tiff_meta(tiff_path, frame_start, frame_end, max_frames)
    selected_float_bytes = meta["frames"] * meta["channels"] * meta["height"] * meta["width"] * 4
    threshold_gb = float(getattr(args, "stream_tiff_threshold_gb", 6.0) or 6.0)
    force_stream = bool(getattr(args, "stream_tiff", False))
    use_stream = force_stream or selected_float_bytes > threshold_gb * (1024 ** 3)
    print(
        "TIFF元数据: "
        f"shape={meta['shape']}, axes={meta['axes']}, dtype={meta['dtype']}, "
        f"frames={meta['start']}:{meta['end']}/{meta['total_frames']}, "
        f"C={meta['channels']}, H={meta['height']}, W={meta['width']}, "
        f"float32约={selected_float_bytes / (1024 ** 3):.2f}GB, "
        f"stream={use_stream}",
        flush=True,
    )
    return use_stream, meta


class StreamingTiffFrames:
    def __init__(self, path, frame_start=0, frame_end=None, max_frames=0):
        self.path = path
        self.tif = None
        self.series = None
        self.memmap = None
        self.pil = None

        try:
            import tifffile
            self.tif = tifffile.TiffFile(path)
            self.series = self.tif.series[0]
            self.axes = getattr(self.series, "axes", None)
            self.shape = tuple(int(x) for x in self.series.shape)
            self.dtype = np.dtype(self.series.dtype)
            total, channels, height, width = _series_shape_to_tcyx_meta(self.shape, self.axes)
            self.start, self.end = _selected_range(total, frame_start, frame_end, max_frames)
            self.n_frames = self.end - self.start
            self.channels = channels
            self.height = height
            self.width = width
            self.is_16bit = bool(np.issubdtype(self.dtype, np.integer) and np.iinfo(self.dtype).bits > 8)
            self.use_pages = _can_read_selected_pages(self.shape, self.axes, len(self.tif.pages))
            if not self.use_pages:
                try:
                    self.memmap = tifffile.memmap(path, series=0)
                    print(f"流式TIFF memmap: shape={self.memmap.shape}, dtype={self.memmap.dtype}", flush=True)
                except Exception as exc:
                    self.memmap = None
                    print(f"流式TIFF memmap不可用: {exc}", flush=True)
            print(
                f"流式TIFF打开: shape={self.shape}, axes={self.axes}, "
                f"frames={self.start}:{self.end}/{total}, pages={len(self.tif.pages)}, "
                f"use_pages={self.use_pages}, use_memmap={self.memmap is not None}",
                flush=True,
            )
            if not self.use_pages and self.memmap is None:
                raise RuntimeError(
                    "当前TIFF不能逐页读取，也不能memmap切片；无法安全流式处理，"
                    "否则会整栈加载导致OOM。请先裁剪/转换为普通多页TIFF。"
                )
        except Exception:
            if self.tif is not None:
                self.tif.close()
                self.tif = None
                raise
            self.tif = None
            self.series = None
            self.memmap = None
            self.pil = Image.open(path)
            total = int(getattr(self.pil, "n_frames", 1))
            self.start, self.end = _selected_range(total, frame_start, frame_end, max_frames)
            self.n_frames = self.end - self.start
            self.channels = 1
            self.height = int(self.pil.height)
            self.width = int(self.pil.width)
            self.is_16bit = self.pil.mode in {"I;16", "I"}
            self.dtype = np.dtype(np.uint16 if self.is_16bit else np.uint8)
            self.axes = "TYX"
            self.shape = (total, self.height, self.width)
            print(f"流式PIL打开: frames={self.start}:{self.end}/{total}", flush=True)

    def close(self):
        if self.tif is not None:
            self.tif.close()
        if self.pil is not None:
            self.pil.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def read_frame(self, local_index):
        local_index = max(0, min(int(local_index), self.n_frames - 1))
        src_index = self.start + local_index
        if self.use_pages:
            arr = np.asarray(self.tif.pages[src_index].asarray())
        elif self.memmap is not None:
            arr = np.asarray(self.memmap[src_index])
        else:
            self.pil.seek(src_index)
            arr = np.asarray(self.pil.copy())

        tcyx = _array_to_tcyx(arr)
        tensor, _ = _normalize_tcyx(tcyx)
        return tensor[0]

    def read_window(self, center_index, window_frames):
        frames = []
        pad = int(window_frames) // 2
        for offset in range(int(window_frames)):
            idx = center_index + offset - pad
            idx = max(0, min(idx, self.n_frames - 1))
            frames.append(self.read_frame(idx))
        return torch.stack(frames, dim=0)


def save_tiff_frame_writer(writer, frame, is_16bit=False):
    if frame.dim() == 3 and frame.size(0) == 1:
        frame = frame.squeeze(0)
    arr = frame.detach().cpu().numpy()
    arr = np.squeeze(arr)
    arr = np.clip(arr, 0.0, 1.0)
    if is_16bit:
        out = (arr * 65535.0 + 0.5).astype(np.uint16)
    else:
        out = (arr * 255.0 + 0.5).astype(np.uint8)
    writer.write(out, photometric="minisblack")


def write_run_report(args, output_base, frames, seconds, before, after):
    import json
    import math
    from pathlib import Path
    report = {
        "framework": "DeepEMD", "input": args.noisy_path,
        "gt": args.gt_path if args._has_gt else None,
        "frames": frames, "config": args.config,
        "optimization_seconds": seconds,
        "quality_metrics": None,
        "evaluation": "Clean-reference metrics" if args._has_gt else "No clean reference; use comparison.pdf for visual inspection.",
    }
    if args._has_gt:
        def values(items):
            return {key: float(value) if math.isfinite(float(value)) else None
                    for key, value in zip(("psnr", "ssim", "snr"), items)}
        report["quality_metrics"] = {"before": values(before), "after": values(after)}
        print("[DeepEMD] Reference metrics:", report["quality_metrics"])
    Path(output_base, "run_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[DeepEMD] Completed {frames} frames; optimization time {seconds:.2f}s.", flush=True)


def denoise_tiff_streaming(args, meta):
    args._has_gt = os.path.abspath(args.noisy_path) != os.path.abspath(args.gt_path)
    try:
        import tifffile
    except ImportError as exc:
        raise SystemExit("Streaming TIFF output needs tifffile. Install it with: pip install tifffile") from exc

    noisy_tiff_path = args.noisy_path
    gt_tiff_path = args.gt_path
    model_name = getattr(args, 'model_name', 'model')
    requested_frame_start = int(getattr(args, 'frame_start', 0) or 0)
    requested_frame_end = int(getattr(args, 'frame_end', -1) or -1)
    requested_max_frames = int(getattr(args, 'max_frames', 0) or 0)

    same_gt = os.path.abspath(noisy_tiff_path) == os.path.abspath(gt_tiff_path)
    noisy_source = StreamingTiffFrames(
        noisy_tiff_path,
        frame_start=requested_frame_start,
        frame_end=requested_frame_end,
        max_frames=requested_max_frames,
    )
    gt_source = noisy_source if same_gt else StreamingTiffFrames(
        gt_tiff_path,
        frame_start=requested_frame_start,
        frame_end=requested_frame_end,
        max_frames=requested_max_frames,
    )

    try:
        is_16bit = noisy_source.is_16bit
        real_frames = min(noisy_source.n_frames, gt_source.n_frames)
        sample = noisy_source.read_frame(0)
        original_h, original_w = int(sample.shape[-2]), int(sample.shape[-1])
        channels = int(sample.shape[0]) if sample.dim() == 3 else 1
        window_frames = args.input_depth if channels == 1 else max(1, args.input_depth // channels)
        pad_frames = window_frames // 2

        if channels == 1:
            setattr(args, 'length', args.input_depth)
            if not hasattr(args, 'multchannel'):
                setattr(args, 'multchannel', False)
            frame_multiplier = 1
        else:
            setattr(args, 'length', max(1, args.input_depth // channels))
            setattr(args, 'multchannel', True)
            frame_multiplier = channels

        pd_scale = int(getattr(args, 'pd_scale', 1) or 1)
        pad_multiple = 64 * max(1, pd_scale)

        first_window = noisy_source.read_window(0, window_frames)
        if channels == 1:
            first_noisy = first_window
        else:
            first_noisy = first_window.view(first_window.size(0) * first_window.size(1), *first_window.size()[2:])
        first_padded = pad_to_multiple(first_noisy, multiple=pad_multiple)
        padded_h, padded_w = first_padded.shape[-2:]
        print(
            f"流式处理: frames={real_frames}, window={window_frames}, "
            f"frame_shape=({channels},{original_h},{original_w}), "
            f"padded=({padded_h},{padded_w}), pad_multiple={pad_multiple}",
            flush=True,
        )

        resume = False
        net_fill, lr_groups = get_model(args, resume=resume, device=device)
        optimizer_fill = optim.Adam(lr_groups, lr=args.lr)

        loss_type = getattr(args, 'loss_type', 'mse')
        if loss_type == 'combined':
            from deepemd.utils.loss import CombinedLoss
            alpha = getattr(args, 'loss_alpha', 1.0)
            beta = getattr(args, 'loss_beta', 0.1)
            gamma = getattr(args, 'loss_gamma', 0.01)
            criterion_fill = CombinedLoss(alpha=alpha, beta=beta, gamma=gamma).to(device)
            print(f"使用组合损失函数: alpha={alpha}, beta={beta}, gamma={gamma}")
        else:
            criterion_fill = torch.nn.MSELoss().to(device)
            print("使用MSE损失函数" if loss_type == 'mse' else f"未知loss_type: {loss_type}, 使用MSE损失")

        date_str = time.strftime("%Y%m%d")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(args.noisy_path))[0]
        noisy_filename = os.path.basename(args.noisy_path)
        if 'averageN=' in noisy_filename:
            import re
            n_value_match = re.search(r'averageN=(\d+)', noisy_filename)
            n_value = int(n_value_match.group(1)) if n_value_match else 1
        else:
            n_value = getattr(args, 'N_value', 1)
        output_tag = str(getattr(args, 'output_tag', '') or '').strip()
        output_base = getattr(args, 'out_dir', f'./outputs/{dataset_name}/{date_str}')
        if output_tag:
            output_base = os.path.join(output_base, output_tag)
        output_dir = os.path.join(output_base, 'denoised_tiff')
        log_dir = os.path.join(output_base, 'logs')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        def make_window(source, i):
            window = source.read_window(i, window_frames)
            if channels == 1:
                img = window
            else:
                img = window.view(window.size(0) * window.size(1), *window.size()[2:])
            return pad_to_multiple(img, multiple=pad_multiple)

        psnr_before_list = []
        ssim_before_list = []
        snr_before_list = []
        if same_gt:
            psnr_before = ssim_before = snr_before = float("nan")
            print("Calculating metrics before denoising skipped: GT is the same TIFF placeholder.")
        else:
            print("Calculating metrics before denoising (streaming)...")
            for i in tqdm(range(0, real_frames), desc='Before metrics'):
                noisy_mid = noisy_source.read_frame(i).unsqueeze(0)
                clean_mid = gt_source.read_frame(i).unsqueeze(0)
                if noisy_mid.shape[-2:] != clean_mid.shape[-2:]:
                    min_h = min(noisy_mid.shape[-2], clean_mid.shape[-2])
                    min_w = min(noisy_mid.shape[-1], clean_mid.shape[-1])
                    noisy_mid = noisy_mid[..., :min_h, :min_w]
                    clean_mid = clean_mid[..., :min_h, :min_w]
                psnr_b, ssim_b, snr_b = test_batch(noisy_mid, clean_mid)
                psnr_before_list.extend(psnr_b)
                ssim_before_list.extend(ssim_b)
                snr_before_list.extend(snr_b)
            psnr_before = np.mean(psnr_before_list)
            ssim_before = np.mean(ssim_before_list)
            snr_before = np.mean(snr_before_list)
        if args._has_gt:
            print(f"Before Denoising - PSNR: {psnr_before:.2f}, SSIM: {ssim_before:.4f}, SNR: {snr_before:.2f}")

        psnr = []
        ssim = []
        snr = []
        time_all = 0

        if args.pretrain:
            for i in tqdm(range(0, real_frames), desc='Pretraining'):
                noisy = make_window(noisy_source, i)
                clean = noisy if same_gt else make_window(gt_source, i)
                noisy_dev = noisy.to(device)
                clean_dev = noisy_dev if same_gt else clean.to(device)
                data = {'img_noisy_torch': noisy_dev, 'img_gt_torch': clean_dev}
                _, _, _, _, _, time_cost = train(args, data, net_fill, optimizer_fill, criterion_fill, original_h, original_w, padded_h, padded_w)
                time_all += time_cost

        output_tmp = os.path.join(output_dir, 'denoised_partial.tif')
        if os.path.exists(output_tmp):
            os.remove(output_tmp)

        with tifffile.TiffWriter(output_tmp, bigtiff=True) as writer:
            for i in tqdm(range(0, real_frames), desc='Denoising'):
                noisy = make_window(noisy_source, i)
                clean = noisy if same_gt else make_window(gt_source, i)
                noisy_dev = noisy.to(device)
                clean_dev = noisy_dev if same_gt else clean.to(device)
                data = {'img_noisy_torch': noisy_dev, 'img_gt_torch': clean_dev}
                psnr_out, ssim_out, snr_out, out_avg, denoised, time_cost = train(
                    args, data, net_fill, optimizer_fill, criterion_fill, original_h, original_w, padded_h, padded_w
                )
                psnr += psnr_out
                ssim += ssim_out
                snr += snr_out
                time_all += time_cost

                if denoised.dim() == 4 and denoised.size(0) == 1:
                    denoised_middle = denoised[0, 0]
                elif denoised.dim() == 4:
                    denoised_middle = denoised[0]
                else:
                    denoised_middle = denoised
                save_tiff_frame_writer(writer, denoised_middle, is_16bit=is_16bit)

        psnr_after = np.mean(psnr)
        ssim_after = np.mean(ssim)
        snr_after = np.mean(snr)

        output_filename = 'denoised.tif'
        output_path = os.path.join(output_dir, output_filename)
        os.replace(output_tmp, output_path)
        print(f"已流式保存多帧TIFF: {output_path} (共{real_frames}帧, 16bit={is_16bit})")

        model_save_path = os.path.join(output_base, 'models', f'{model_name}.pth')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        net_fill.save_model(model_save_path)

        write_run_report(args, output_base, real_frames, time_all,
                         (psnr_before, ssim_before, snr_before),
                         (psnr_after, ssim_after, snr_after))
    finally:
        if gt_source is not noisy_source:
            gt_source.close()
        noisy_source.close()


def denoise_tiff(args):
    args._has_gt = os.path.abspath(args.noisy_path) != os.path.abspath(args.gt_path)
    noisy_tiff_path = args.noisy_path
    gt_tiff_path = args.gt_path
    model_name = getattr(args, 'model_name', 'model')
    requested_frame_start = int(getattr(args, 'frame_start', 0) or 0)
    requested_frame_end = int(getattr(args, 'frame_end', -1) or -1)
    requested_max_frames = int(getattr(args, 'max_frames', 0) or 0)

    use_stream, stream_meta = should_stream_tiff(
        args,
        noisy_tiff_path,
        frame_start=requested_frame_start,
        frame_end=requested_frame_end,
        max_frames=requested_max_frames,
    )
    if use_stream:
        return denoise_tiff_streaming(args, stream_meta)

    print(f"加载噪声TIFF: {noisy_tiff_path}", flush=True)
    noisy_frames_tensor, is_16bit_noisy = load_tiff_frames(
        noisy_tiff_path,
        frame_start=requested_frame_start,
        frame_end=requested_frame_end,
        max_frames=requested_max_frames,
    )
    print(f"噪声帧数: {len(noisy_frames_tensor)}, 形状: {noisy_frames_tensor.shape}, 16bit: {is_16bit_noisy}", flush=True)

    if os.path.abspath(noisy_tiff_path) == os.path.abspath(gt_tiff_path):
        print("GT路径与噪声TIFF相同，复用已加载的输入作为无GT占位。", flush=True)
        gt_frames_tensor = noisy_frames_tensor.clone()
        is_16bit_gt = is_16bit_noisy
    else:
        print(f"加载GT TIFF: {gt_tiff_path}", flush=True)
        gt_frames_tensor, is_16bit_gt = load_tiff_frames(
            gt_tiff_path,
            frame_start=requested_frame_start,
            frame_end=requested_frame_end,
            max_frames=requested_max_frames,
        )
    print(f"GT帧数: {len(gt_frames_tensor)}, 形状: {gt_frames_tensor.shape}, 16bit: {is_16bit_gt}", flush=True)
    
    # 统一使用噪声图的位深作为输出标准
    is_16bit = is_16bit_noisy
    
    if noisy_frames_tensor.size(0) != gt_frames_tensor.size(0):
        min_frames = min(noisy_frames_tensor.size(0), gt_frames_tensor.size(0))
        print(f"警告：噪声帧数({noisy_frames_tensor.size(0)})与GT帧数({gt_frames_tensor.size(0)})不匹配，裁剪到共同帧数 {min_frames}")
        noisy_frames_tensor = noisy_frames_tensor[:min_frames]
        gt_frames_tensor = gt_frames_tensor[:min_frames]
    if noisy_frames_tensor.shape[-2:] != gt_frames_tensor.shape[-2:]:
        min_h = min(noisy_frames_tensor.shape[-2], gt_frames_tensor.shape[-2])
        min_w = min(noisy_frames_tensor.shape[-1], gt_frames_tensor.shape[-1])
        print(f"警告：噪声尺寸{tuple(noisy_frames_tensor.shape[-2:])}与GT尺寸{tuple(gt_frames_tensor.shape[-2:])}不匹配，裁剪到共同尺寸 {(min_h, min_w)}")
        noisy_frames_tensor = noisy_frames_tensor[..., :min_h, :min_w]
        gt_frames_tensor = gt_frames_tensor[..., :min_h, :min_w]

    real_frames = noisy_frames_tensor.size(0)
    window_frames = args.input_depth if noisy_frames_tensor.size(1) == 1 else args.input_depth // 3
    pad_frames = window_frames // 2

    noisy_frames_tensor = pad_temporal_replicate(noisy_frames_tensor, pad_frames)
    gt_frames_tensor = pad_temporal_replicate(gt_frames_tensor, pad_frames)

    if noisy_frames_tensor.size(1) == 1:
        noisy_img = noisy_frames_tensor
        clean_img = gt_frames_tensor
        setattr(args, 'length', args.input_depth)
        if not hasattr(args, 'multchannel'):
            setattr(args, 'multchannel', False)
        frame_multiplier = 1
    else:
        noisy_img = noisy_frames_tensor.view(noisy_frames_tensor.size(0) * noisy_frames_tensor.size(1), *noisy_frames_tensor.size()[2:])
        clean_img = gt_frames_tensor.view(gt_frames_tensor.size(0) * gt_frames_tensor.size(1), *gt_frames_tensor.size()[2:])
        setattr(args, 'length', args.input_depth // 3)
        setattr(args, 'multchannel', True)
        frame_multiplier = 3
    
    print(f"noisy_img shape before spatial pad: {noisy_img.shape}")
    
    pd_scale = int(getattr(args, 'pd_scale', 1) or 1)
    pad_multiple = 64 * max(1, pd_scale)
    
    noisy_padded = pad_to_multiple(noisy_img, multiple=pad_multiple)
    clean_padded = pad_to_multiple(clean_img, multiple=pad_multiple)
    
    padded_h, padded_w = noisy_padded.shape[-2:]
    print(f"noisy_img shape after spatial pad: {noisy_padded.shape} (multiple of {pad_multiple})")

    resume = False
    net_fill, lr_groups = get_model(args, resume=resume, device=device)

    optimizer_fill = optim.Adam(lr_groups, lr=args.lr)

    # 根据配置选择损失函数
    loss_type = getattr(args, 'loss_type', 'mse')
    if loss_type == 'combined':
        from deepemd.utils.loss import CombinedLoss
        # 可配置的权重参数
        alpha = getattr(args, 'loss_alpha', 1.0)
        beta = getattr(args, 'loss_beta', 0.1)
        gamma = getattr(args, 'loss_gamma', 0.01)
        criterion_fill = CombinedLoss(alpha=alpha, beta=beta, gamma=gamma).to(device)
        print(f"使用组合损失函数: alpha={alpha}, beta={beta}, gamma={gamma}")
    elif loss_type == 'mse':
        criterion_fill = torch.nn.MSELoss().to(device)
        print("使用MSE损失函数")
    else:
        # 默认使用MSE
        criterion_fill = torch.nn.MSELoss().to(device)
        print(f"未知loss_type: {loss_type}, 使用MSE损失")

    psnr = []
    ssim = []
    snr = []
    time_all = 0

    original_h, original_w = noisy_img.shape[-2:]

    # 创建输出目录结构: outputs/数据集名称/YYYYMMDD/denoised_tiff/ 和 outputs/数据集名称/YYYYMMDD/logs/
    date_str = time.strftime("%Y%m%d")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 从GT路径中提取数据集名称（例如：dataset/0121/GTvolume.tif → 0121）
    dataset_name = os.path.splitext(os.path.basename(args.noisy_path))[0]
    # 从噪声文件名中提取N值（例如：lowSNRvolume_averageN=1.tif → N=1）
    noisy_filename = os.path.basename(args.noisy_path)
    if 'averageN=' in noisy_filename:
        import re
        n_value_match = re.search(r'averageN=(\d+)', noisy_filename)
        n_value = int(n_value_match.group(1)) if n_value_match else 1
    else:
        n_value = getattr(args, 'N_value', 1)
    output_tag = str(getattr(args, 'output_tag', '') or '').strip()
    output_base = getattr(args, 'out_dir', f'./outputs/{dataset_name}/{date_str}')
    if output_tag:
        output_base = os.path.join(output_base, output_tag)
    output_dir = os.path.join(output_base, 'denoised_tiff')
    log_dir = os.path.join(output_base, 'logs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    usable_frames = real_frames

    # 计算去噪前的指标
    if args._has_gt:
        print("Calculating metrics before denoising...")
        psnr_before_list = []
        ssim_before_list = []
        snr_before_list = []

        for i in range(0, usable_frames):
            if frame_multiplier == 1:
                noisy_mid = noisy_padded[i + pad_frames]
                clean_mid = clean_padded[i + pad_frames]
            else:
                base = (i + pad_frames) * frame_multiplier
                noisy_mid = noisy_padded[base:base + 1]
                clean_mid = clean_padded[base:base + 1]

            # 裁剪到原始尺寸
            noisy_mid_cropped = noisy_mid[:, :original_h, :original_w].unsqueeze(0)
            clean_mid_cropped = clean_mid[:, :original_h, :original_w].unsqueeze(0)

            if i == 0:
                check_mse = torch.mean((noisy_mid_cropped - clean_mid_cropped)**2).item()
                print(f"Alignment Check (Frame 0 MSE): {check_mse:.6f}")
                if args.noisy_path == args.gt_path and check_mse > 1e-6:
                    print("FATAL ERROR: Noisy and GT are not spatially/temporally aligned!")
                    exit(1)

            psnr_b, ssim_b, snr_b = test_batch(noisy_mid_cropped, clean_mid_cropped)
            psnr_before_list.extend(psnr_b)
            ssim_before_list.extend(ssim_b)
            snr_before_list.extend(snr_b)

        psnr_before = np.mean(psnr_before_list)
        ssim_before = np.mean(ssim_before_list)
        snr_before = np.mean(snr_before_list)

        print(f"Before Denoising - PSNR: {psnr_before:.2f}, SSIM: {ssim_before:.4f}, SNR: {snr_before:.2f}")

    else:
        psnr_before = ssim_before = snr_before = float('nan')
        print('[DeepEMD] No clean reference: quality metrics are disabled.')

    if args.pretrain:
        for i in tqdm(range(0, usable_frames), desc='Pretraining'):
            if frame_multiplier == 1:
                # 单通道：取3帧（前帧+当前帧+后帧）
                noisy = noisy_padded[i:i+args.length]
                clean = clean_padded[i:i+args.length]
            else:
                # 多通道：取args.length个RGB帧（每个RGB是3通道）
                noisy = noisy_padded[i*frame_multiplier:(i+args.length)*frame_multiplier, :]
                clean = clean_padded[i*frame_multiplier:(i+args.length)*frame_multiplier, :]
            data = {'img_noisy_torch': noisy.to(device), 'img_gt_torch': clean.to(device)}
            psnr_out, ssim_out, snr_out, out_avg, denoised, time_cost = train(args, data, net_fill, optimizer_fill, criterion_fill, original_h, original_w, padded_h, padded_w)
            time_all += time_cost

    # 收集所有去噪帧
    denoised_frames = []

    for i in tqdm(range(0, usable_frames), desc='Denoising'):
        if frame_multiplier == 1:
            # 单通道：取3帧（前帧+当前帧+后帧）
            noisy = noisy_padded[i:i+args.length]
            clean = clean_padded[i:i+args.length]
        else:
            # 多通道：取args.length个RGB帧
            noisy = noisy_padded[i*frame_multiplier:(i+args.length)*frame_multiplier, :]
            clean = clean_padded[i*frame_multiplier:(i+args.length)*frame_multiplier, :]
        data = {'img_noisy_torch': noisy.to(device), 'img_gt_torch': clean.to(device)}
        psnr_out, ssim_out, snr_out, out_avg, denoised, time_cost = train(args, data, net_fill, optimizer_fill, criterion_fill, original_h, original_w, padded_h, padded_w)

        psnr += psnr_out
        ssim += ssim_out
        snr += snr_out
        time_all += time_cost

        # 提取中间帧并收集
        if denoised.dim() == 4 and denoised.size(0) == 1:
            denoised_middle = denoised[0, 0]  # (1, 1, H, W) -> (H, W)
        elif denoised.dim() == 4:
            denoised_middle = denoised[0]  # 其他4维情况
        else:
            denoised_middle = denoised  # 3维情况直接使用

        denoised_frames.append(denoised_middle.cpu())

    psnr_after = np.mean(psnr)
    ssim_after = np.mean(ssim)
    snr_after = np.mean(snr)

    # 保存为多帧TIFF文件
    output_filename = 'denoised.tif'
    output_path = os.path.join(output_dir, output_filename)
    output_tmp = os.path.join(output_dir, 'denoised_partial.tif')
    save_multi_frame_tiff(denoised_frames, output_tmp, is_16bit=is_16bit)
    os.replace(output_tmp, output_path)

    # 保存模型
    model_save_path = os.path.join(output_base, 'models', f'{model_name}.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    net_fill.save_model(model_save_path)

    write_run_report(args, output_base, usable_frames, time_all,
                     (psnr_before, ssim_before, snr_before),
                     (psnr_after, ssim_after, snr_after))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepEMD internal denoising engine')
    parser.add_argument('--config', type=str, required=True,
                        help='Explicit legacy-engine YAML; use run_deepemd.py for the public INR default.')
    parser.add_argument('--noisy_path', type=str, required=True,
                        help='Path to noisy TIFF file')
    parser.add_argument('--gt_path', type=str, required=True,
                        help='Path to ground truth TIFF file')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory to save denoised TIFFs.')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name for saving (default: model)')
    parser.add_argument('--output_tag', type=str, default=None)
    parser.add_argument('--frame_start', type=int, default=None)
    parser.add_argument('--frame_end', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', type=int, default=None)
    parser.add_argument('--tile_size', type=int, default=None,
                        help='Spatial tile size for large frames (0/None=full frame). '
                             'Trains on random patches and stitches tiled inference.')
    parser.add_argument('--tile_overlap', type=float, default=None,
                        help='Tile overlap ratio for stitched inference (default 0.5)')

    cli_args = parser.parse_args()
    config_args = get_args(cli_args.config)
    for key, value in vars(cli_args).items():
        if key != 'config':
            setattr(config_args, key, value)
    config_args.config = cli_args.config

    config_args.resume = False
    seed = getattr(config_args, 'seed', None)
    if seed is None:
        seed = 2024
        setattr(config_args, 'seed', seed)
    deterministic = getattr(config_args, 'deterministic', None)
    if deterministic is None:
        deterministic = True
        setattr(config_args, 'deterministic', deterministic)
    set_deterministic(int(seed), bool(int(deterministic)) if isinstance(deterministic, (int, np.integer)) else bool(deterministic))

    denoise_tiff(config_args)
