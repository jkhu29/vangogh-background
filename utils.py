import numpy as np
import torch
import torch.nn as nn


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, 0.1)


def upsampling(img, x, y):
    func = nn.Upsample(size=[x, y], mode='bilinear', align_corners=True)
    return func(img)


def generate_noise(size, channels=1, type='gaussian', scale=2, noise=None):
    if type == 'gaussian':
        noise = torch.randn(channels, size[0], round(size[1] / scale), round(size[2] / scale))
        noise = upsampling(noise, size[1], size[2])
    if type == 'gaussian_mixture':
        noise1 = torch.randn(channels, size[0], size[1], size[2]) + 5
        noise2 = torch.randn(channels, size[0], size[1], size[2])
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(channels, size[0], size[1], size[2])
    return noise * 10.


def concat_noise(img, *args):
    noise = generate_noise(*args)
    if isinstance(img, torch.Tensor):
        noise = noise.to(img.device)
    else:
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    mixed_img = torch.cat((img, noise), 1)
    return mixed_img


def calc_gram(x):
    n, c, h, w = x.shape
    f = x.view(n, c, w * h)
    f_trans = f.transpose(1, 2)
    gram = f.bmm(f_trans).div_(c * h * w)
    return gram


def rgb2lum(arr):
    small = np.where(arr <= 0.04045)
    big = np.where(arr > 0.04045)
    arr[small] /= 12.92
    arr[big] = ((arr[big] + 0.055) / 1.055) ** 2.4
    return arr


def lum(image):
    """
    turn BGR to Lum
    :param image: image in sRGB area, range 255
    :return: image in Lum
    """
    assert image.shape[0] == 3, "make sure the layout is (c, h, w), BGR"
    _, h, w = image.shape
    image = image.astype(np.float)
    v_b = image[0, ...] / 255
    v_g = image[1, ...] / 255
    v_r = image[2, ...] / 255
    print(rgb2lum(v_r))
    l_image = 0.2126 * rgb2lum(v_r) + 0.7152 * rgb2lum(v_g) + 0.0722 * rgb2lum(v_b)
    return l_image


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
