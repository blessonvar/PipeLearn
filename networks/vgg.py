import torch.nn as nn

from networks.network import Network


class VGG(Network):
    def __init__(self, config, model_type="server"):
        super(VGG, self).__init__(config, model_type)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class Convolutional(nn.Module):
    def __init__(self, in_channel_num=None, out_channel_num=None, kernel_size=None, padding=1, stride=1,
                 bias=True, pool_type=None, pool_kernel_size=None, pool_stride=None, pool_padding=0):
        super(Convolutional, self).__init__()
        self.in_channel_num = in_channel_num
        self.out_channel_num = out_channel_num
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        # set up layers
        self.conv = nn.Conv2d(self.in_channel_num, self.out_channel_num,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              stride=self.stride,
                              bias=self.bias)
        self.bn = nn.BatchNorm2d(self.out_channel_num)
        self.relu = nn.ReLU(inplace=True)
        if pool_type == 'maxpool':
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        elif pool_type == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        else:
            self.pool = None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.pool:
            out = self.pool(out)
        return out


class Dense(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, drop_out=None):
        super(Dense, self).__init__()
        self.in_channel_num = in_channel_num
        self.out_channel_num = out_channel_num
        self.drop_out = drop_out
        self.linear = nn.Linear(self.in_channel_num, self.out_channel_num)
        self.drop_out_layer = nn.Dropout(drop_out) if drop_out else None

    def forward(self, x):
        out = self.linear(x)
        if self.drop_out_layer:
            out = self.drop_out_layer(x)
        return out
