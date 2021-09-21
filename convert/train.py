import copy
import itertools

import torch
from torch import nn
from torch import optim
from torch.utils.data import dataloader
from torch.autograd import Variable
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm

from models import CartoonGAN_G, DeCartoonGAN_G, GAN_D

import sys
sys.path.append("..")
import config
import utils


class BasicCycleGAN(object):
    def __init__(self):
        super(BasicCycleGAN, self).__init__()
        opt = config.get_options()
        self.device = torch.device("cuda" if opt.cuda else "cpu")
        self.niter = opt.niter
        self.batch_size = opt.batch_size
        self.workers = opt.workers
        self.batch_scale = opt.batch_scale
        self.lr = opt.lr
        self.output_dir = opt.output_dir

        self.target_fake = Variable(torch.rand(self.batch_size) * 0.3).to(self.device)
        self.target_real = Variable(torch.rand(self.batch_size) * 0.5 + 0.7).to(self.device)

        # cyclegan for bgan, init
        self.model_g_x2y = CartoonGAN_G().to(self.device)
        self.model_g_y2x = DeCartoonGAN_G().to(self.device)
        self.model_d_x = GAN_D().to(self.device)
        self.model_d_y = GAN_D().to(self.device)

        self.model_g_x2y.apply(utils.weights_init)
        self.model_g_y2x.apply(utils.weights_init)
        self.model_d_x.apply(utils.weights_init)
        self.model_d_y.apply(utils.weights_init)

        # criterion init
        self.criterion_generate = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # dataset init
        description = {
            "vangogh": "byte",
            "photo": "byte",
            "size": "int",
        }
        train_dataset = TFRecordDataset("train.tfrecord", None, description)
        self.train_dataloader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True
        )
        self.data_length = 10000  # ans of dataset.py

        # valid_dataset = TFRecordDataset("valid.tfrecord", None, description)
        # self.valid_dataloader = dataloader.DataLoader(
        #     dataset=valid_dataset,
        #     batch_size=1
        # )

        # optim init
        self.optimizer_g = optim.Adam(
            itertools.chain(self.model_g_x2y.parameters(), self.model_g_y2x.parameters()),
            lr=self.lr, betas=(0.75, 0.999)
        )
        self.optimizer_d_x = optim.Adam(
            self.model_d_x.parameters(),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_d_y = optim.Adam(
            self.model_d_y.parameters(),
            lr=self.lr, betas=(0.5, 0.999)
        )

        # lr init
        self.model_scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=self.niter
        )
        self.model_scheduler_d_x = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d_x, T_max=self.niter
        )
        self.model_scheduler_d_y = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d_y, T_max=self.niter
        )

    def train_batch(self):
        cnt = 0
        for epoch in range(self.niter):
            self.model_g_x2y.train()
            self.model_g_y2x.train()

            epoch_losses_g = utils.AverageMeter()
            epoch_losses_d_x = utils.AverageMeter()
            epoch_losses_d_y = utils.AverageMeter()

            with tqdm(total=(self.data_length - self.data_length % self.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch + 1, self.niter))

                for record in self.train_dataloader:
                    self.model_d_x.eval()
                    self.model_d_y.eval()

                    cnt += 1
                    vangogh = record["vangogh"].reshape(
                        self.batch_size,
                        3,
                        record["size"][0],
                        record["size"][0]
                    ).float().to(self.device)
                    photo = record["photo"].reshape(
                        self.batch_size,
                        3,
                        record["size"][0],
                        record["size"][0]
                    ).float().to(self.device)

                    vangogh_noise = utils.concat_noise(vangogh, (4, 128, 128), vangogh.size()[0])
                    photo_noise = utils.concat_noise(photo, (4, 128, 128), photo.size()[0])

                    # --------------------
                    # generator train(2 * model_g)
                    # --------------------
                    loss_total, vangogh_fake, photo_fake = self._calc_loss_g(vangogh_noise, vangogh, photo_noise, photo)

                    if cnt % self.batch_scale == 0:
                        self.optimizer_g.zero_grad()
                        loss_total.backward()
                        epoch_losses_g.update(loss_total.item(), self.batch_size)
                        self.optimizer_g.step()

                    # --------------------
                    # discriminator vangogh train(model_d_x)
                    # --------------------
                    self.model_d_x.train()
                    loss_total_d_x = self._calc_loss_d(vangogh_fake, vangogh)

                    if cnt % self.batch_scale == 0:
                        self.optimizer_d_x.zero_grad()
                        loss_total_d_x.backward()
                        epoch_losses_d_x.update(loss_total_d_x.item(), self.batch_size)
                        self.optimizer_d_x.step()

                    # --------------------
                    # discriminator photo train(model_d_y)
                    # --------------------
                    self.model_d_y.train()
                    loss_total_d_y = self._calc_loss_d(photo_fake, photo)

                    if cnt % self.batch_scale == 0:
                        self.optimizer_d_y.zero_grad()
                        loss_total_d_y.backward()
                        epoch_losses_d_y.update(loss_total_d_y.item(), self.batch_size)
                        self.optimizer_d_y.step()

                    t.set_postfix(
                        loss_g='{:.6f}'.format(epoch_losses_g.avg),
                        loss_d_vangogh='{:.6f}'.format(epoch_losses_d_x.avg),
                        loss_d_photo='{:.6f}'.format(epoch_losses_d_y.avg)
                    )
                    t.update(self.batch_size)

            self.model_scheduler_g.step()
            self.model_scheduler_d_x.step()
            self.model_scheduler_d_y.step()

            torch.save(
                self.model_g_x2y.state_dict(),
                "{}/photo2vangogh_snapshot_{}.pth".format(self.output_dir, epoch)
            )

    def _calc_loss_g(self, vangogh_noise, vangogh, photo_noise, photo):
        # loss identity(ATTN!: `a_same = model_b2a(a)`)
        vangogh_same = self.model_g_x2y(vangogh_noise)  # model_g_x2y: photo --> vangogh
        loss_identity_vangogh = self.criterion_identity(vangogh_same, vangogh)

        photo_fake = self.model_g_y2x(photo)  # model_g_y2x: vangogh --> photo
        loss_identity_photo = self.criterion_identity(photo_fake, photo)

        # loss gan(ATTN!: `a_fake = model_b2a(b)`)
        vangogh_fake = self.model_g_x2y(photo_noise)
        vangogh_fake_feature = self.model_d_y(vangogh_fake)  # get vangogh features
        loss_gan_x2y = self.criterion_generate(vangogh_fake_feature, self.target_real)

        photo_fake = self.model_g_y2x(vangogh)
        photo_fake_feature = self.model_d_x(photo_fake)  # get photo features
        loss_gan_y2x = self.criterion_generate(photo_fake_feature, self.target_real)

        photo_fake_noise = utils.concat_noise(photo_fake, (4, 128, 128), photo_fake.size()[0])

        # loss cycle(ATTN!: `a_recover = model_b2a(b_fake)`)
        vangogh_recover = self.model_g_x2y(photo_fake_noise)  # recover the vangogh: vangogh->photo->vangogh
        loss_cycle_x2y = self.criterion_cycle(vangogh_recover, vangogh) * 2

        photo_recover = self.model_g_y2x(vangogh_fake)  # recover the photo: photo->vangogh->photo
        loss_cycle_y2x = self.criterion_cycle(photo_recover, photo) * 2

        # loss total
        loss_total = loss_identity_vangogh + loss_identity_photo + \
                     loss_gan_x2y + loss_gan_y2x + \
                     loss_cycle_x2y + loss_cycle_y2x

        return loss_total, vangogh_recover, photo_recover

    def _calc_loss_d(self, vangogh_fake, vangogh):
        # loss real
        pred_vangogh_real = self.model_d_x(vangogh)
        loss_real = self.criterion_generate(pred_vangogh_real, self.target_real)

        # loss fake
        vangogh_fake_ = copy.deepcopy(vangogh_fake.data)
        pred_vangogh_fake = self.model_d_x(vangogh_fake_.detach())
        loss_fake = self.criterion_generate(pred_vangogh_fake, self.target_fake)

        # loss rbl
        loss_rbl = - torch.log(abs(loss_real - loss_fake)) \
                   - torch.log(abs(1 - loss_fake - loss_real))

        # loss total
        loss_total = (loss_real + loss_fake) * 0.5 + loss_rbl * 0.01

        return loss_total


if __name__ == "__main__":
    cartoon = BasicCycleGAN()
    cartoon.train_batch()
