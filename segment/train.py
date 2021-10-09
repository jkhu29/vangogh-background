import random
import warnings

import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm

import losses
from models import U2Net

import sys
sys.path.append("..")
import config
import utils


opt = config.get_options()

# deveice init
CUDA_ENABLE = torch.cuda.is_available()
if CUDA_ENABLE and opt.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
elif CUDA_ENABLE and not opt.cuda:
    warnings.warn("WARNING: You have CUDA device, so you should probably run with --cuda")
elif not CUDA_ENABLE and opt.cuda:
    assert CUDA_ENABLE, "ERROR: You don't have a CUDA device"

device = torch.device('cuda:0' if CUDA_ENABLE else 'cpu')

# seed init
manual_seed = opt.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# dataset init, train file need .tfrecord
description = {
    "inputs": "byte",
    "labels": "byte",
}
train_dataset = TFRecordDataset("train.tfrecord", None, description)
# do not shuffle
train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True
)
length = 1700

# models init
model = U2Net().to(device)
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 0.1)
        nn.init.constant_(m.bias, 0)

# optim and scheduler init
model_optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=1e-8, weight_decay=1)
model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=opt.niter)

img_test = cv2.imread("00001.png")
img_test = cv2.resize(img_test, (192*3, 192*4))
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
img_test = img_test.transpose(2, 0, 1)
img_test = torch.from_numpy(img_test).reshape(1, 3, 192*4, 192*3).float().to(device)

# train model
print("-----------------train-----------------")
for epoch in range(opt.niter):
    model.train()
    epoch_losses = utils.AverageMeter()
    epoch_loss0 = utils.AverageMeter()

    with tqdm(total=(length - length % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))

        for record in train_dataloader:
            inputs = record["inputs"].reshape(
                opt.batch_size,
                3,
                192*4,
                192*3,
            )
            inputs = inputs.float().to(device) / 255
            labels = record["labels"].reshape(
                opt.batch_size,
                1,
                192*4,
                192*3,
            )
            labels = labels.float().to(device) / 255

            d0, d1, d2, d3, d4, d5, d6 = model(inputs)
            # print(d0, d1, d2, d3, d4, d5, d6, labels)

            model_optimizer.zero_grad()
            loss0, loss = losses.muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
            loss.backward()

            model_optimizer.step()
            epoch_losses.update(loss.item(), len(inputs))
            epoch_loss0.update(loss0.item(), len(inputs))

            t.set_postfix(
                loss=epoch_losses.avg,
                loss0=epoch_loss0.avg,
            )
            t.update(len(inputs))

    model_scheduler.step()
    
    model.eval()
    with torch.no_grad():
        label_pred, _, _, _, _, _, _ = model(img_test)
        cv2.imwrite("epoch_{}.png".format(epoch), label_pred[0][0].cpu().numpy()*255)
    torch.save(model.state_dict(), "u2net_epoch_{}.pth".format(epoch))
