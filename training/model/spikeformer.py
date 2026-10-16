from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(r'/media/data/huawenhui/Spike-Driven-Transformer-main/Spike-Driven-Transformer-main')
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module import *
from torch import einsum



class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        in_channels=3,
        num_classes=5000,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=4,
        pooling_stat="1111",
        attn_mode="direct_xor",
        spike_mode="lif",
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        pretrained=False,
        pretrained_cfg=None,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                )
                for j in range(depths)
            ]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))  # 1,1,768
        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x, _, hook = patch_embed(x, hook=hook)
        #print(f'before blk shape:{x.shape}')
        for blk in block:
            x, _, hook = blk(x, hook=hook)  #forward features shape:torch.Size([4, 16, 512, 14, 14]) forward flatten features shape:torch.Size([4, 16, 512, 196]) forward mean features shape:torch.Size([4, 16, 512])
        cls_token = x.flatten(3).mean(3).unsqueeze(-1)  # 由于CLS是全局信息因此改为这个结构 [T,bs,dim]
        x = x.flatten(3)  # 由于这个网络是卷积实现的 所以有别于常规的Transformer 这个形状是 ([4, 16, 512, 196])
        x = torch.cat((cls_token, x), dim =-1)  #最后一步就会变成[4,16,512,197] #此时和常规VIT类似了
        x = x.permute(0, 1, 3, 2)
        #x = x.flatten(3).mean(3)
        return x, hook

    def forward(self, x, hook=None):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x, hook = self.forward_features(x, hook=hook)
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()
        x = self.head(x)  # 得到每一个位置的token的logits
        x = x.mean(0)  # 维度应该是[bs, 197, codebooksize]
        return x


@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model

# model = SpikeDrivenTransformer(img_size_h=224, img_size_w=224,
#         patch_size=16, embed_dims=768, num_heads=8, mlp_ratios=4,
#         in_channels=3, num_classes=524288, qkv_bias=False, depths=1,
#         )
# image_sample = torch.randn(1, 3, 224, 224)
# output = model(image_sample)
# print("模型输出形状:", output.shape)
