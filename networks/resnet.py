import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from networks.network import Network


class ResNet(Network):
    def __init__(self, config, model_type="server"):
        super(ResNet, self).__init__(config, model_type)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResBasicBlock(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, stride,
                 pool_type=None, pool_kernel_size=None, pool_stride=None, pool_padding=0):
        super(ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_num)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel_num, out_channel_num, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_num)
        self.conv3 = None
        self.bn3 = None
        if stride != 1 or in_channel_num != out_channel_num:
            self.conv3 = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=1, stride=stride, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel_num)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride
        if pool_type == 'maxpool':
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        elif pool_type == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        else:
            self.pool = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv3 and self.bn3:
            identity = self.conv3(identity)
            identity = self.bn3(identity)
        out += identity
        out = self.relu2(out)
        if self.pool:
            out = self.pool(out)
        return out


class ResBottleneck(nn.Module):
    def __init__(self,  in_channel_num, out_channel_num, stride,
                 pool_type=None, pool_kernel_size=None, pool_stride=None, pool_padding=0):
        expansion = 4
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_num)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel_num, out_channel_num, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_num)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channel_num, out_channel_num * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel_num * expansion)
        self.conv4 = None
        self.bn4 = None
        if stride != 1 or in_channel_num != out_channel_num * expansion:
            self.conv4 = nn.Conv2d(in_channel_num, out_channel_num * expansion, kernel_size=1, stride=stride, bias=False)
            self.bn4 = nn.BatchNorm2d(out_channel_num * expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.stride = stride
        if pool_type == 'maxpool':
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        elif pool_type == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        else:
            self.pool = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.conv4 and self.bn4:
            identity = self.conv4(identity)
            identity = self.bn4(identity)
        out += identity
        out = self.relu3(out)
        if self.pool:
            out = self.pool(out)
        return out
