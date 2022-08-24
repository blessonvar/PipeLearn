import torch.nn as nn

from networks.vgg import Convolutional, Dense
from networks.resnet import ResBasicBlock, ResBottleneck


class LayerConfig:
    def __init__(self, layer_type, in_channel_num=None, out_channel_num=None, kernel_size=None, padding=1, stride=1,
                 drop_out=None, bias=True, pool_type=None, pool_kernel_size=None, pool_stride=None, pool_padding=0):
        self.layer_type = layer_type
        self.in_channel_num = in_channel_num
        self.out_channel_num = out_channel_num
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.drop_out = drop_out
        self.bias = bias
        self.pool_type = pool_type
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def make_layers(self):
        if self.layer_type == 'Convolutional':
            return Convolutional(self.in_channel_num, self.out_channel_num,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 stride=self.stride,
                                 bias=self.bias,
                                 pool_type=self.pool_type,
                                 pool_kernel_size=self.pool_kernel_size,
                                 pool_stride=self.pool_stride,
                                 pool_padding=self.pool_padding)
        if self.layer_type == 'ResBasicBlock':
            return ResBasicBlock(self.in_channel_num, self.out_channel_num,
                                 stride=self.stride,
                                 pool_type=self.pool_type,
                                 pool_kernel_size=self.pool_kernel_size,
                                 pool_stride=self.pool_stride,
                                 pool_padding=self.pool_padding)
        if self.layer_type == 'ResBottleneck':
            return ResBottleneck(self.in_channel_num, self.out_channel_num,
                                 stride=self.stride,
                                 pool_type=self.pool_type,
                                 pool_kernel_size=self.pool_kernel_size,
                                 pool_stride=self.pool_stride,
                                 pool_padding=self.pool_padding)
        if self.layer_type == 'Dense':
            return Dense(self.in_channel_num, self.out_channel_num, self.drop_out)
