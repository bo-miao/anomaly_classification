import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import Module, Parameter
import torchvision.models as models

# personal pkg
from .block import *
from .memory_patch import *


class Unet_Free(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free, self).__init__()

        # n_channel = args.c
        # t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        n_channel = 3
        t_length = 3
        self.encoder = Encoder_Free(t_length, n_channel)
        self.decoder = Decoder_Free(t_length, n_channel)
        self.memory = Memory()
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, memory=None, train=True):

        feature, skip1, skip2, skip3 = self.encoder(x)
        # 1/8 resolution
        read_memory, memory, memory_loss = self.memory(feature, memory, train)
        updated_feature = torch.cat((feature, read_memory), dim=1)
        reconstructed_image = self.decoder(updated_feature, skip1, skip2, skip3)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss,
                'memory_loss': memory_loss,
                }
        return reconstructed_image, loss, memory


class ResUnet(nn.Module):
    def __init__(self, criterion=None, args=None, filters=[16, 32, 64, 32]):
        super(ResUnet, self).__init__()

        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        channel = n_channel * t_length
        # channel = 9
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)
        # HERE TO ADD MEM [3]
        self.upsample_1 = Upsample(filters[3]*2, filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 3, 1, 1),
            nn.Tanh(),
        )

        self.memory = Memory()
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, memory=None, train=True):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge         # Decode HERE TO ADD MEMORY 1/8 resolution
        x4 = self.bridge(x3)
        read_memory, memory, memory_loss = self.memory(x4, memory, train)
        x4 = torch.cat((x4, read_memory), dim=1)
        x4 = self.upsample_1(x4)

        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        reconstructed_image = self.output_layer(x10)

        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss,
                'memory_loss': memory_loss,
                }
        return reconstructed_image, loss, memory


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="4"
    import time
    m=Unet_Free(criterion=nn.MSELoss(reduction='none'))
    a = torch.rand(10,9,256,256)
    aa,bb,cc=m(a, a[:,:3], torch.rand(32*32,16))
    print(bb['memory_loss'].view(bb['memory_loss'].shape[0],-1).mean(1).shape,bb['pixel_loss'].shape)
