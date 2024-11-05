import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
affine_par = True
import functools

import sys, os
from pdb import set_trace
in_place = True

class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)




class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.gn1 = nn.GroupNorm(16, inplanes)
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1,1,1),
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.relu = nn.ReLU(inplace=in_place)

        self.gn2 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1,1,1),
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)


        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class unet3D(nn.Module):
    def __init__(self, layers, num_classes=3, weight_std = False, bch=32):
        self.inplanes = bch*4
        self.weight_std = weight_std
        super(unet3D, self).__init__()

        self.conv1 = conv3x3x3(1, bch, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, bch, bch, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, bch, bch*2, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, bch*2, bch*4, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, bch*4, bch*8, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, bch*8, bch*8, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, bch*8),
            nn.ReLU(inplace=in_place),
            conv3x3x3(bch*8, bch*8, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, bch*8, bch*4, 1, stride=(1, 1, 1))
        self.x4_resb = self._make_layer(NoBottleneck, bch*4, bch*2, 1, stride=(1, 1, 1))
        self.x2_resb = self._make_layer(NoBottleneck, bch*2, bch, 1, stride=(1, 1, 1))
        self.x1_resb = self._make_layer(NoBottleneck, bch, bch, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, bch),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(bch, 8, kernel_size=1)
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(16, bch*8),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool3d((1,1,1))
        )
        
        self.head = nn.Sequential(
            nn.Conv3d(8,8,kernel_size=1,stride=1,padding=0,bias=True),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(8,8,kernel_size=1,stride=1,padding=0,bias=True),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(8,2,kernel_size=1,stride=1,padding=0,bias=True)
        )
        # self.controller = nn.Conv3d(256+7, 162, kernel_size=1, stride=1, padding=0)
        self._init_weights()

    def _init_weights(self):
        print("initialize weights ... ")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.normal_(m.weight)
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]]=1
        return task_encoding.cuda()

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, input, task_id):

        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)

        x = self.fusionConv(x)

        # x8
        x = self.upsamplex2(x)
        if x.shape != skip3.shape:
            x = F.interpolate(x, size=skip3.shape[-3:], align_corners=True, mode='trilinear')
        x = x + skip3
        x = self.x8_resb(x)

        # x4
        x = self.upsamplex2(x)
        if x.shape != skip2.shape:
            x = F.interpolate(x, size=skip2.shape[-3:], align_corners=True, mode='trilinear')
        x = x + skip2
        x = self.x4_resb(x)

        # x2
        x = self.upsamplex2(x)
        if x.shape != skip1.shape:
            x = F.interpolate(x, size=skip1.shape[-3:], align_corners=True, mode='trilinear')
        x = x + skip1
        x = self.x2_resb(x)

        # x1
        x = self.upsamplex2(x)
        if x.shape != skip0.shape:
            x = F.interpolate(x, size=skip0.shape[-3:], align_corners=True, mode='trilinear')
        x = x + skip0
        x = self.x1_resb(x)

        head_inputs = self.precls_conv(x)
        logits = self.head(head_inputs)
        return logits

def UNet3D(num_classes=1, weight_std=False):
    print("Using Singlehead 8,8,2")
    model = unet3D([1, 2, 2, 2, 2], num_classes, weight_std)
    return model