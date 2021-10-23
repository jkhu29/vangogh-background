import torch.nn as nn
import torchvision
from collections import namedtuple


def _make_layer(block, num_layers, **kwargs):
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


class ConvReLU(nn.Module):
    """ConvReLU: conv 64 * 3 * 3 + leakyrelu"""
    def __init__(self, in_channels, out_channels, withbn=False):
        super(ConvReLU, self).__init__()
        self.withbn = withbn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pad(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTranReLU(nn.Module):
    """ConvTranReLU: conv trans 64 * 3 * 3 + leakyrelu"""
    def __init__(self, in_channels, out_channels, withbn=False):
        super(ConvTranReLU, self).__init__()
        self.withbn = withbn
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.convtran = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.convtran(x)
        x = self.pad(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x


class ResBlock(nn.Module):
    """ResBlock used by CartoonGAN"""
    def __init__(self, num_conv=5, channels=64):
        super(ResBlock, self).__init__()
        self.conv_relu = _make_layer(ConvReLU, num_layers=num_conv, in_channels=channels, out_channels=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_relu(x)
        x = self.conv(x) + x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, bias=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(self.avg_pool(x))


class CAB(nn.Module):
    def __init__(self, num_features, reduction: int = 8):
        super(CAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return self.module(x)


class CartoonGAN_G(nn.Module):
    """the G of CartoonGAN, use conv-transpose to up sample"""
    def __init__(self, in_channels=3, out_channels=32, num_resblocks=5):
        super(CartoonGAN_G, self).__init__()

        self.in_channels = in_channels
        self.num_resblocks = num_resblocks

        self.conv_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=self.in_channels, out_channels=out_channels)

        # down sample
        self.conv_relu2 = _make_layer(ConvReLU, num_layers=2, in_channels=out_channels, out_channels=out_channels)
        self.conv_relu3 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels * 2, withbn=True)

        self.res1 = _make_layer(ResBlock, num_layers=self.num_resblocks, num_conv=1, channels=out_channels * 2)

        # up sample
        self.convup_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels * 2, out_channels=out_channels, withbn=True)
        self.convup_relu2 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels)

        self.conv4 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_relu1(x)
        res1 = x
        x = self.conv_relu2(x)
        res2 = x
        x = self.conv_relu3(x)
        x = self.res1(x)
        x = self.convup_relu1(x)
        x = x + res2
        x = self.convup_relu2(x)
        x = x + res1
        del res2, res1
        x = self.conv4(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


if __name__ == "__main__":
    from torchsummary import summary
    a = CartoonGAN_G().cuda()
    summary(a, (3, 64, 64))
