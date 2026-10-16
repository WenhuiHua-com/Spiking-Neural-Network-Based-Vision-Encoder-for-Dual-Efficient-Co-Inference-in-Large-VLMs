import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from timm.models.layers import to_2tuple

class PulseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x 预期是0或1的二进制脉冲张量 [N, C_in, H_in, W_in]
        x_float = x.float()

        N, C_in, H_in, W_in = x_float.shape

        H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = torch.zeros(N, self.out_channels, H_out, W_out, device=x.device, dtype=x.dtype)

        # 对输入进行padding，以便在循环中可以统一处理边界
        padded_x = F.pad(x_float, (self.padding, self.padding, self.padding, self.padding))

        for n in range(N):
            for out_c in range(self.out_channels):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        window_input = padded_x[n, :, h_start:h_end, w_start:w_end]
                        kernel_weights = self.weight[out_c, :, :, :]

                        current_sum = 0.0
                        # 核心的“条件累加”逻辑
                        for c_in in range(C_in):
                            for k_h in range(self.kernel_size):
                                for k_w in range(self.kernel_size):
                                    if window_input[c_in, k_h, k_w] == 1:
                                        current_sum += kernel_weights[c_in, k_h, k_w]

                        output[n, out_c, h_out, w_out] = current_sum

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output
        
class MS_SPS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
        spike_mode="lif",
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        if spike_mode == "lif":
            self.proj_lif1 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.proj_lif1 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        if spike_mode == "lif":
            self.proj_lif2 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.proj_lif2 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.proj_lif3 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.proj_lif3 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.rpe_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

    def forward(self, x, hook=None):
        T, B, _, H, W = x.shape
        ratio = 1
        x = self.proj_conv(x.flatten(0, 1))  # have some fire value
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x)
        if hook is not None:
            hook[self._get_name() + "_lif"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif2(x)
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        x_feat = x
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())
        if hook is not None:
            hook[self._get_name() + "_lif3"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W), hook
