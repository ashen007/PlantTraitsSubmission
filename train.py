import numpy as np
import torch
import tqdm
from torch import nn
from move import move_to


class AvgMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val.sum()
        self.count += val.numel()
        self.avg = self.sum / self.count


class Compile(object):

    def __init__(self,
                 model,
                 loss,
                 optimizer,
                 init_lr,
                 weight_decay,
                 epochs,
                 batch_size,
                 train_loader,
                 save_to,
                 val_loader=None,
                 metrics=None):
        self.epochs = epochs
        self.model = model.cuda()
        self.loss = loss()
        self.optimizer = optimizer(params=model.parameters(),
                                   lr=init_lr,
                                   weight_decay=weight_decay)
        # self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
        #                                                         init_lr,
        #                                                         steps_per_epoch=len(train_loader),
        #                                                         epochs=epochs,
        #                                                         pct_start=0.1,
        #                                                         anneal_strategy='cos',
        #                                                         div_factor=1e3,
        #                                                         final_div_factor=1e4)
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                                len(train_loader),
        #                                                                0)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                       factor=0.5,
                                                                       patience=3,
                                                                       cooldown=2,
                                                                       verbose=True)
        self.metrics = metrics
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.track_loss = AvgMeter()
        self.best_score = np.inf
        self.save_to = save_to

    def fit(self):

        for epoch in range(self.epochs):
            # reset metrics
            if self.metrics is not None:
                for key, val in self.metrics.items():
                    val.reset()

            self.track_loss.reset()
            self.model.train()

            for X, Y in tqdm.tqdm(self.train_loader, desc='steps '):
                X = move_to(X, 'cuda')
                Y = move_to(Y, 'cuda')

                y_pred = self.model(X)
                loss = self.loss(y_pred, Y)
                self.track_loss.update(loss)

                if self.model.training:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # self.lr_scheduler.step()

                for key, val in self.metrics.items():
                    val.update(y_pred, Y)

            dtl = [f'{key}: {val.compute().item():.7f}' for key, val in self.metrics.items()]
            print(f"epoch: {epoch + 1}, "
                  f"training loss: {self.track_loss.avg:.7f}, ",
                  " ".join(dtl))

            if self.val_loader is not None:
                if self.metrics is not None:
                    for key, val in self.metrics.items():
                        val.reset()

                self.track_loss.reset()
                self.model.eval()

                for X, Y in tqdm.tqdm(self.val_loader, desc='steps '):
                    X = move_to(X, 'cuda')
                    Y = move_to(Y, 'cuda')

                    with torch.no_grad():
                        y_pred = self.model(X)
                        loss = self.loss(y_pred, Y)
                        self.track_loss.update(loss)

                    for key, val in self.metrics.items():
                        val.update(y_pred, Y)

                dtl = [f'{key}: {val.compute().item():.7f}' for key, val in self.metrics.items()]
                print(f"epoch: {epoch + 1}, "
                      f"validation loss: {self.track_loss.avg:.7f}",
                      " ".join(dtl))

            self.lr_scheduler.step(self.track_loss.avg)

            # torch.save({'model_state_dict': self.model.state_dict(),
            #             }, 'last_checkpoint.pth')

            if self.track_loss.avg < self.best_score:
                torch.save({'model_state_dict': self.model.state_dict(),
                            }, self.save_to)

                self.best_score = self.track_loss.avg
