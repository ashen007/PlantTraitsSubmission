import numpy as np
import torch
import tqdm
from move import move_to
from torch import nn
from torch.autograd import Variable
from models.colearn_effvit_small import CoLearnEffnet
from train import AvgMeter
from loss import NTXentLoss
from loss_structure import loss_structrue, loss_structrue_t


class Compile(object):

    def __init__(self,
                 init_lr,
                 epochs,
                 batch_size,
                 train_loader,
                 save_to,
                 val_loader=None,
                 metrics=None
                 ):
        self.init_lr = init_lr
        self.epochs = epochs
        self.train_loader = train_loader
        self.save_to = save_to
        self.val_loader = val_loader
        self.metrics = metrics
        self.batch_size = batch_size

        # model
        self.model = CoLearnEffnet().cuda()
        self.optim1 = torch.optim.AdamW(self.model.parameters(), lr=self.init_lr)
        self.optim2 = torch.optim.AdamW(list(self.model.reg_head.parameters()), lr=self.init_lr / 5)
        self.lr_scheduler1 = torch.optim.lr_scheduler.OneCycleLR(self.optim1,
                                                                 init_lr,
                                                                 steps_per_epoch=len(train_loader),
                                                                 epochs=epochs,
                                                                 pct_start=0.1,
                                                                 anneal_strategy='cos',
                                                                 div_factor=1e1,
                                                                 final_div_factor=1e1)
        self.lr_scheduler2 = torch.optim.lr_scheduler.OneCycleLR(self.optim2,
                                                                 init_lr,
                                                                 steps_per_epoch=len(train_loader),
                                                                 epochs=epochs,
                                                                 pct_start=0.1,
                                                                 anneal_strategy='cos',
                                                                 div_factor=1e1,
                                                                 final_div_factor=1e1)

        # loss
        self.smoothed_l1_loss = nn.SmoothL1Loss()
        self.ntxent_loss = NTXentLoss(self.batch_size, 0.5, True)

        self.param_v = None
        self.track_sup_loss = AvgMeter()
        self.track_sim_loss = AvgMeter()
        self.track_tot_loss = AvgMeter()
        self.best_score = np.inf

        state = torch.load('./ckpts/best_ckpt_colearn_effvit_288.pth')
        self.model.load_state_dict(state['model_state_dict'])

    def fit(self):
        for epoch in range(self.epochs):
            if self.metrics is not None:
                for key, val in self.metrics.items():
                    val.reset()

            self.track_sup_loss.reset()
            self.track_sim_loss.reset()
            self.track_tot_loss.reset()
            self.model.train()

            # model train
            for raw, img1, img2, Y in tqdm.tqdm(self.train_loader, desc='train steps '):
                pos_1, pos_2 = (Variable(img1).to('cuda', non_blocking=True),
                                Variable(img2).to('cuda', non_blocking=True))
                raw = Variable(raw).to('cuda', non_blocking=True)
                Y = move_to(Y, 'cuda')

                feat, outs, logits = self.model(raw)

                if self.param_v is None:
                    loss_feat = loss_structrue(outs.detach(), logits)
                else:
                    loss_feat = loss_structrue_t(outs.detach(), logits, self.param_v)

                if self.model.training:
                    self.optim2.zero_grad()
                    loss_feat.backward()
                    self.optim2.step()
                    self.lr_scheduler2.step()

                # self learning
                out_1 = self.model(pos_1, ignore_feat=True, forward_fc=False)
                out_2 = self.model(pos_2, ignore_feat=True, forward_fc=False)
                loss_con = self.ntxent_loss(out_1, out_2)

                self.track_sim_loss.update(loss_con)

                # supervised learning
                _, logits = self.model(raw, ignore_feat=True)
                loss_sup = self.smoothed_l1_loss(logits, Y)

                self.track_sup_loss.update(loss_sup)

                # loss
                loss = loss_sup + loss_con

                self.track_tot_loss.update(loss)

                if self.model.training:
                    self.optim1.zero_grad()
                    loss.backward()
                    self.optim1.step()
                    self.lr_scheduler1.step()

                if self.metrics is not None:
                    for key, val in self.metrics.items():
                        val.update(logits, Y)

            if self.metrics is not None:
                dtl = [f'{key}: {val.compute().item():.7f}' for key, val in self.metrics.items()]
            else:
                dtl = []

            print(f"epoch: {epoch + 1}, "
                  f"training sim loss: {self.track_sim_loss.avg:.7f}, ",
                  f"training sup loss: {self.track_sup_loss.avg:.7f}, ",
                  f"training total loss: {self.track_tot_loss.avg:.7f} ",
                  " ".join(dtl))

            # model evaluation
            if self.val_loader is not None:
                if self.metrics is not None:
                    for key, val in self.metrics.items():
                        val.reset()

                self.track_sup_loss.reset()
                self.track_sim_loss.reset()
                self.track_tot_loss.reset()
                self.model.eval()

                for raw, img1, img2, Y in tqdm.tqdm(self.val_loader, desc='valid steps '):
                    pos_1, pos_2 = (Variable(img1).to('cuda', non_blocking=True),
                                    Variable(img2).to('cuda', non_blocking=True))
                    raw = Variable(raw).to('cuda', non_blocking=True)
                    Y = move_to(Y, 'cuda')

                    with torch.no_grad():
                        feat, outs, logits = self.model(raw)

                        if self.param_v is None:
                            loss_feat = loss_structrue(outs.detach(), logits)
                        else:
                            loss_feat = loss_structrue_t(outs.detach(), logits, self.param_v)

                        # self learning
                        out_1 = self.model(pos_1, ignore_feat=True, forward_fc=False)
                        out_2 = self.model(pos_2, ignore_feat=True, forward_fc=False)
                        loss_con = self.ntxent_loss(out_1, out_2)

                        self.track_sim_loss.update(loss_con)

                        # supervised learning
                        _, logits = self.model(raw, ignore_feat=True)
                        loss_sup = self.smoothed_l1_loss(logits, Y)

                        self.track_sup_loss.update(loss_sup)

                        # loss
                        loss = loss_sup + loss_con

                        self.track_tot_loss.update(loss)

                    if self.metrics is not None:
                        for key, val in self.metrics.items():
                            val.update(logits, Y)

                if self.metrics is not None:
                    dtl = [f'{key}: {val.compute().item():.7f}' for key, val in self.metrics.items()]
                else:
                    dtl = []

                print(f"epoch: {epoch + 1}, "
                      f"validation sim loss: {self.track_sim_loss.avg:.7f}, ",
                      f"validation sup loss: {self.track_sup_loss.avg:.7f}, ",
                      f"validation total loss: {self.track_tot_loss.avg:.7f} ",
                      " ".join(dtl))

            if self.track_sup_loss.avg < self.best_score:
                torch.save({'model_state_dict': self.model.state_dict()}, self.save_to)

                self.best_score = self.track_sup_loss.avg
