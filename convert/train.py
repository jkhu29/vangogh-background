import cv2
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm

from models import CartoonGAN_G, VGG16

import sys

sys.path.append("..")
import config
import utils


class BasicStyleTransfer(object):
    def __init__(self, style_img_path, content_img_path):
        super(BasicStyleTransfer, self).__init__()
        opt = config.get_options()
        self.device = torch.device("cuda" if opt.cuda else "cpu")
        self.niter = opt.niter
        self.batch_size = opt.batch_size
        self.workers = opt.workers
        self.batch_scale = opt.batch_scale
        self.lr = opt.lr
        self.output_dir = opt.output_dir

        # train model init
        self.model_transfer = CartoonGAN_G().to(self.device)
        self.vgg = VGG16().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        # self.model_transfer.apply(utils.weights_init)
        model_params = torch.load("pretrain.pth")
        self.model_transfer.load_state_dict(model_params)

        # prepare init
        self.prep = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.40760392, 0.45795686, 0.48501961],
                    # subtract imagenet mean, for BGR
                    std=[0.225, 0.224, 0.229]
                )
            ]
        )
        style_img = torch.FloatTensor(cv2.imread(style_img_path).transpose(2, 0, 1)).to(self.device)
        features_style = self.vgg(self.prep(style_img.unsqueeze(0)/255)*255)
        self.grams_style = [utils.calc_gram(feature_style.repeat(self.batch_size, 1, 1, 1)) for feature_style in features_style]
        if content_img_path is not None:
            content_img = self.prep(torch.FloatTensor(cv2.imread(content_img_path).transpose(2, 0, 1))).to(self.device)
            _, h, w = content_img.shape
            self.photo = torch.rand(
                self.batch_size,
                3,
                h,
                w
            ).float().to(self.device) * 255
            self.features_content = self.vgg(content_img.unsqueeze(0)).relu3_3.repeat(self.batch_size, 1, 1, 1)

        # criterion init
        self.criterion = nn.MSELoss()
        self.style_weight = 30
        self.content_weight = 1

        # dataset init
        self.data_length = 10000

        # optim init
        self.optimizer_transfer = optim.Adam(
            self.model_transfer.parameters(), lr=self.lr
        )

        # lr init
        self.model_scheduler_transfer = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_transfer, T_max=self.niter
        )

    def train_batch(self):
        print("-----------------train-----------------")
        for epoch in range(self.niter):
            self.model_transfer.train()

            epoch_losses_style = utils.AverageMeter()
            epoch_losses_content = utils.AverageMeter()

            with tqdm(total=(self.data_length - self.data_length % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch + 1, self.niter))
                for _ in range(self.data_length):
                    # --------------------
                    # model transfer train
                    # --------------------
                    vangogh = self.model_transfer(self.photo)
                    features_vangogh = self.vgg(vangogh)

                    # get loss_content
                    loss_content = self.criterion(self.features_content, features_vangogh.relu3_3)

                    # get loss_style
                    loss_style = 0
                    for feature_vangogh, gram_style in zip(features_vangogh, self.grams_style):
                        gram_vangogh = utils.calc_gram(feature_vangogh)
                        loss_style += self.criterion(gram_vangogh, gram_style[:self.batch_size, :, :])

                    loss_total = self.style_weight * loss_style + \
                                 self.content_weight * loss_content

                    self.optimizer_transfer.zero_grad()
                    loss_total.backward()
                    self.optimizer_transfer.step()
                    epoch_losses_style.update(loss_style.item(), self.batch_size)
                    epoch_losses_content.update(loss_content.item(), self.batch_size)

                    t.set_postfix(
                        loss_style='{:.6f}'.format(epoch_losses_style.avg),
                        loss_content='{:.6f}'.format(epoch_losses_content.avg),
                    )
                    t.update(self.batch_size)

            self.model_scheduler_transfer.step()


class BasicFastStyleTransfer(BasicStyleTransfer):
    def __init__(self, style_img_path, content_img_path=None):
        super(BasicFastStyleTransfer, self).__init__(style_img_path=style_img_path, content_img_path=content_img_path)
        description = {
            "vangogh": "byte",
            "photo": "byte",
            "size": "int",
        }
        train_dataset = TFRecordDataset("train.tfrecord", None, description, shuffle_queue_size=100)
        self.train_dataloader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        self.data_length = 4000

    def train_batch(self):
        print("-----------------train-----------------")
        for epoch in range(self.niter):
            self.model_transfer.train()

            epoch_losses_style = utils.AverageMeter()
            epoch_losses_content = utils.AverageMeter()
            epoch_losses_tv = utils.AverageMeter()

            with tqdm(total=(self.data_length - self.data_length % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch + 1, self.niter))
                for record in self.train_dataloader:
                    photo = record["photo"].reshape(
                        self.batch_size,
                        3,
                        record["size"][0],
                        record["size"][0]
                    ).float().to(self.device)

                    # --------------------
                    # model transfer train
                    # --------------------
                    vangogh = self.model_transfer(photo)
                    photo_p = self.prep(photo / 255) * 255
                    vangogh_p = self.prep(vangogh / 255) * 255
                    features_photo = self.vgg(photo_p)
                    features_vangogh = self.vgg(vangogh_p)
                    del photo_p, vangogh_p

                    # get loss_content
                    loss_content = self.criterion(features_photo.relu2_2, features_vangogh.relu2_2)
                    
                    # get loss tv
                    diff_i = torch.sum(torch.abs(vangogh[..., 1:] - vangogh[..., :-1]))
                    diff_j = torch.sum(torch.abs(vangogh[..., 1:, :] - vangogh[..., :-1, :]))
                    loss_tv = diff_i + diff_j

                    # get loss_style
                    loss_style = 0
                    for feature_vangogh, gram_style in zip(features_vangogh, self.grams_style):
                        gram_vangogh = utils.calc_gram(feature_vangogh)
                        loss_style += self.criterion(gram_vangogh, gram_style[:self.batch_size, :, :])

                    loss_total = self.style_weight * loss_style + \
                                 self.content_weight * loss_content + 5e-6 * loss_tv

                    self.optimizer_transfer.zero_grad()
                    loss_total.backward()
                    self.optimizer_transfer.step()
                    epoch_losses_style.update(loss_style.item(), self.batch_size)
                    epoch_losses_content.update(loss_content.item(), self.batch_size)
                    epoch_losses_tv.update(loss_tv.item(), self.batch_size)

                    t.set_postfix(
                        loss_style='{:.6f}'.format(epoch_losses_style.avg),
                        loss_content='{:.6f}'.format(epoch_losses_content.avg),
                        loss_tv='{:.6f}'.format(epoch_losses_tv.avg),
                    )
                    t.update(self.batch_size)

            self.model_scheduler_transfer.step()

            torch.save(
                self.model_transfer.state_dict(),
                "{}/photo2vangogh_snapshot_{}.pth".format(self.output_dir, epoch)
            )


if __name__ == "__main__":
    cartoon = BasicFastStyleTransfer(style_img_path="vangogh.jpg")
    cartoon.train_batch()
