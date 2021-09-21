import torch
import torch.nn as nn


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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTranReLU(nn.Module):
    """ConvTranReLU: conv trans 64 * 3 * 3 + leakyrelu"""
    def __init__(self, in_channels, out_channels, withbn=False):
        super(ConvTranReLU, self).__init__()
        self.withbn = withbn
        self.convtran = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.convtran(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    """ResBlock used by CartoonGAN and DeCartoonGAN"""
    def __init__(self, num_conv=5, channels=64):
        super(ResBlock, self).__init__()

        self.conv_relu = _make_layer(ConvReLU, num_layers=num_conv, in_channels=channels, out_channels=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.dropout = nn.Dropout2d(0.5, inplace=True)

    def forward(self, x):
        x = self.conv_relu(x)
        res = x
        if self.training:
            x = self.dropout(x)
        x = self.conv(x) + res
        return x


class CartoonGAN_G(nn.Module):
    """the G of CartoonGAN, use conv-transpose to up sample"""
    def __init__(self, in_channels=3, out_channels=64, num_resblocks=8):
        super(CartoonGAN_G, self).__init__()

        self.in_channels = in_channels + 4
        self.num_resblocks = num_resblocks

        self.conv_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=self.in_channels, out_channels=out_channels)

        # down sample
        self.conv_relu2 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels * 2, withbn=True)
        self.conv_relu3 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels * 2, out_channels=out_channels * 4)

        self.res1 = _make_layer(ResBlock, num_layers=self.num_resblocks, num_conv=2, channels=out_channels * 4)

        # up sample
        self.convup_relu1 = _make_layer(ConvTranReLU, num_layers=1, in_channels=out_channels * 4, out_channels=out_channels * 2, withbn=True)
        self.convup_relu2 = _make_layer(ConvTranReLU, num_layers=1, in_channels=out_channels * 2, out_channels=out_channels)

        self.conv4 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_relu1(x)
        res = x
        x = self.conv_relu2(x)
        x = self.conv_relu3(x)
        x = self.res1(x)
        x = self.convup_relu1(x)
        x = self.convup_relu2(x)
        x = res + x
        del res
        x = self.conv4(x)
        return x


class DeCartoonGAN_G(CartoonGAN_G):
    """the G of DeCartoonGAN, use conv-transpose to up sample"""
    def __init__(self, in_channels=3, out_channels=64, num_resblocks=4):
        super(DeCartoonGAN_G, self).__init__()
        self.in_channels = in_channels
        self.num_resblocks = num_resblocks

        self.conv_relu1 = _make_layer(ConvReLU, num_layers=1, in_channels=self.in_channels, out_channels=out_channels)

        # down sample
        self.conv_relu2 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels, out_channels=out_channels * 2, withbn=True)
        self.conv_relu3 = _make_layer(ConvReLU, num_layers=1, in_channels=out_channels * 2, out_channels=out_channels * 4)

        self.res1 = _make_layer(ResBlock, num_layers=self.num_resblocks, num_conv=2, channels=out_channels * 4)

        # up sample
        self.convup_relu1 = _make_layer(ConvTranReLU, num_layers=1, in_channels=out_channels * 4, out_channels=out_channels * 2, withbn=True)
        self.convup_relu2 = _make_layer(ConvTranReLU, num_layers=1, in_channels=out_channels * 2, out_channels=out_channels)

        self.conv4 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)


class GAN_D(nn.Module):
    """GAN_D: VGG19"""
    def __init__(self):
        super(GAN_D, self).__init__()
        self.net_d = [nn.Conv2d(3, 64, kernel_size=3, padding=1)]
        self.net_d.extend([nn.LeakyReLU(0.2)])
        self._conv_block(64, 64, with_stride=True)
        self._conv_block(64, 128)
        self._conv_block(128, 128, with_stride=True)
        self._conv_block(128, 256)
        self._conv_block(256, 256, with_stride=True)
        self._conv_block(256, 512)
        self._conv_block(512, 512, with_stride=True)
        self.net_d.extend([nn.AdaptiveAvgPool2d(1)])
        self.net_d.extend([nn.Conv2d(512, 1024, kernel_size=1)])
        self.net_d.extend([nn.LeakyReLU(0.2)])
        self.net_d.extend([nn.Conv2d(1024, 1, kernel_size=1)])
        self.net_d = nn.Sequential(*self.net_d)

    def _conv_block(self, in_channels, out_channels, with_batch=True, with_stride=False):
        if with_stride:
            self.net_d.extend([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)])
        else:
            self.net_d.extend([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)])

        if with_batch:
            self.net_d.extend([nn.BatchNorm2d(out_channels)])
        self.net_d.extend([nn.LeakyReLU(0.2, inplace=True)])

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net_d(x).view(batch_size))


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)
