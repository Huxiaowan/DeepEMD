import torch
import numpy as np
import torch.nn.functional as F
from deepemd.models.Fill_network_arch import Fill_network
import torch.nn as nn
from math import ceil
import yaml


def write_logs(string,log_dir='log.txt'):
    print(string)
    with open(log_dir,'a') as f:
        f.write(string+'\n')


class DictToClass(object):
    @classmethod
    def _to_class(cls, _obj):
        _obj_ = type('new', (object,), _obj)
        [setattr(_obj_, key, cls._to_class(value)) if isinstance(value, dict) else setattr(_obj_, key, value) for
         key, value in _obj.items()]
        return _obj_


def get_args(yml_path):
    ## setup
    args=DictToClass._to_class(
                yaml.load(open(yml_path,'r',encoding='utf-8').read(),Loader=yaml.FullLoader)
                )
    return args


def tensor2plot(input):
    return torch.clip(input,0,1).cpu().squeeze(0).permute(1,2,0)


def comp_params(net):
    '''Compute number of parameters'''
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)



def get_model(args,resume=True,device='cuda'):
    # 如果resume=False（从头训练），强制使用我们设置的multchannel参数
    if not resume:
        multchannel = getattr(args, 'multchannel', True)
    else:
        multchannel = True if 'multchannel' in args.modelpath else False

    # 获取Self-Attention参数
    use_self_attn = getattr(args, 'use_self_attn', False)
    num_heads = getattr(args, 'num_heads', 8)
    depth = getattr(args, 'depth', 4)
    dim = getattr(args, 'dim', 64)
    attn_window_size = getattr(args, 'attn_window_size', 16)  # 轻量级注意力窗口大小
    use_cross_attn = getattr(args, 'use_cross_attn', True)  # 默认开启交叉注意力

    # 获取Z轴增强参数
    use_z_attn = getattr(args, 'use_z_attn', False)
    use_z_pyramid = getattr(args, 'use_z_pyramid', False)
    z_window = getattr(args, 'z_window', 7)
    use_hybrid_2d1d = getattr(args, 'use_hybrid_2d1d', False)

    # 获取3D CNN参数
    use_3d = getattr(args, 'use_3d', False)
    use_pure_3d = getattr(args, 'use_pure_3d', False)

    mask_type = getattr(args, 'mask_type', 'bernoulli')
    mask_replace = getattr(args, 'mask_replace', 'zero')
    mask_fixed = getattr(args, 'mask_fixed', False)
    mask_block_size = getattr(args, 'mask_block_size', 16)
    mask_share_across_channels = getattr(args, 'mask_share_across_channels', False)
    mask_patchsz = getattr(args, 'mask_patchsz', 1)
    mask_center_only = getattr(args, 'mask_center_only', False)
    train_mask_ratio = getattr(args, 'train_mask_ratio', None)
    infer_mask_ratio = getattr(args, 'infer_mask_ratio', None)
    mask_ratio = getattr(args, 'mask_ratio', None)
    if mask_ratio is None:
        mask_ratio = train_mask_ratio if train_mask_ratio is not None else [15, 15]
        setattr(args, 'mask_ratio', mask_ratio)

    net_fill=Fill_network(in_chans=args.input_depth,mask_ratio=mask_ratio,
                       multchannel=multchannel,
                       patchsz=mask_patchsz,
                       mask_type=mask_type,
                       mask_replace=mask_replace,
                       mask_fixed=mask_fixed,
                       mask_block_size=mask_block_size,
                       mask_share_across_channels=mask_share_across_channels,
                       mask_center_only=mask_center_only,
                       train_mask_ratio=train_mask_ratio,
                       infer_mask_ratio=infer_mask_ratio,
                       network=args.modelpath.split('_')[-1].split('.')[0],
                       use_self_attn=use_self_attn,
                       num_heads=num_heads,
                       depth=depth,
                       dim=dim,
                       attn_window_size=attn_window_size,
                       use_z_attn=use_z_attn,
                       use_z_pyramid=use_z_pyramid,
                       z_window=z_window,
                       use_hybrid_2d1d=use_hybrid_2d1d,
                       use_3d=use_3d,
                       use_pure_3d=use_pure_3d,
                       use_cross_attn=use_cross_attn)
    net_fill=net_fill.to(device)
    comp_params(net_fill)

    lr_groups=[]
    if resume:
        saved_state_dict = torch.load(args.modelpath, map_location = device)['params']
        net_fill.load_state_dict(saved_state_dict,strict=True)
    lr_groups.append({'params':net_fill.parameters(),'lr':args.lr})

    return net_fill,lr_groups
