import os
import torch
import pathlib
import time

from argparse import ArgumentParser

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer

#from in_out.datasets import load_dataset
from in_out.datasets_label import load_dataset
from networks.dreem_net import DreemNet
from metrics.specificity import Specificity
from metrics.sensitivity import Sensitivity
from metrics.auc import Auc
from metrics.f1score import F1score

from torch.nn import BCEWithLogitsLoss, MSELoss, L1Loss
from torch.optim import Adam



class Model(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        path_to_results = os.path.normpath(os.path.join(os.path.dirname(__file__), '../results'))
        self.output_dir = os.path.join(path_to_results, hparams.experiment)
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        (self.dataset_train, self.dataset_val, self.dataset_test) = load_dataset(hparams.fold)
        self.loader_train = DataLoader(self.dataset_train, batch_size=hparams.batch_size, shuffle=True,
                                       num_workers=0, pin_memory=True, drop_last=True)
        self.loader_val = DataLoader(self.dataset_val, batch_size=len(self.dataset_val), shuffle=False,
                                     num_workers=0, pin_memory=True, drop_last=True)
        self.loader_test = DataLoader(self.dataset_test, batch_size=len(self.dataset_test), shuffle=False,
                                      num_workers=0, pin_memory=True, drop_last=True)

        self.loss = BCEWithLogitsLoss()  # 0, 1
        # self.loss = MSELoss() # Regression

        # Plot curves
        self.metrics = {
            'spe': Specificity(),
            'sen': Sensitivity(),
            'auc': Auc(),
            'f1score' : F1score()
        }
        # self.metrics = {
        #     'mae': L1Loss()
        # }

        self.normalizer = float(hparams.batch_size) / len(self.dataset_train) #  Could be a problem
        self.metrics_avg = {key: 0.0 for key in self.metrics.keys()}
        self.metrics_avg['loss'] = 0.0

        self.net = DreemNet(n_channels=hparams.n_channels, n_virtual_channels=hparams.n_virtual_channels,
                            convolution_size=hparams.convolution_size, pool_size=hparams.pool_size,
                            n_hidden_channels=hparams.n_hidden_channels)
        # self.net = CataNet()

    def forward(self, x):
        return self.net(x)

    def on_epoch_start(self):
        self.metrics_avg = {key: 0.0 for key in self.metrics_avg.keys()}

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_yp = self(batch_x)  # FORWARD (don't call it explicitly! :))
        loss = self.loss(batch_yp, batch_y)

        metrics = {key: value(batch_yp.detach(), batch_y) for key, value in self.metrics.items()}
        metrics['loss'] = loss.detach()
        self.metrics_avg = {key: value + metrics[key] * self.normalizer for key, value in self.metrics_avg.items()}
        for key, value in metrics.items():
            self.logger.experiment.add_scalars(key, {'train': value}, self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            batch_x, batch_y = batch
            batch_yp = self(batch_x)
            loss = self.loss(batch_yp, batch_y)

            metrics = {key: value(batch_yp, batch_y) for key, value in self.metrics.items()}
            metrics['loss'] = loss.detach()

            return metrics

    def validation_end(self, outputs):  # validation_epoch_end

        metrics = {key: torch.stack([x[key] for x in outputs]).mean() for key in self.metrics_avg.keys()}
        for key, value in metrics.items():
            self.logger.experiment.add_scalars(key, {'val': value}, self.global_step)

        return {'val_loss': metrics['loss']}

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            batch_x, batch_y = batch
            batch_yp = self(batch_x)
            loss = self.loss(batch_yp, batch_y)

            metrics = {key: value(batch_yp, batch_y) for key, value in self.metrics.items()}
            metrics['loss'] = loss

            return metrics

    def test_end(self, outputs):
        metrics = {key: torch.stack([x[key] for x in outputs]).mean() for key in self.metrics_avg.keys()}
        for key, value in metrics.items():
            self.logger.experiment.add_scalars(key, {'test': value}, self.global_step)
        return {'test_loss': metrics['loss']}

    def train_dataloader(self):
        return self.loader_train

    def val_dataloader(self):
        return self.loader_val

    def test_dataloader(self):
        return self.loader_test

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)



def main(hparams):

    model = Model(hparams)

    trainer = Trainer(
        default_save_path=model.output_dir,
        min_epochs=hparams.n_epochs, max_epochs=hparams.n_epochs,
        # fast_dev_run=True
    )

    start_time = time.time()
    trainer.fit(model)
    end_time = time.time()

    if end_time - start_time > 60 * 60 * 24:
        print('>> Training took: %s' %
              time.strftime("%d days, %H hours, %M minutes and %S seconds", time.gmtime(end_time - start_time)))
    elif end_time - start_time > 60 * 60:
        print('>> Training took: %s' %
              time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(end_time - start_time)))
    elif end_time - start_time > 60:
        print('>> Training took: %s' % time.strftime("%M minutes and %S seconds", time.gmtime(end_time - start_time)))
    else:
        print('>> Training took: %s' % time.strftime("%S seconds", time.gmtime(end_time - start_time)))

    trainer.test()


if __name__ == '__main__':
    parser = ArgumentParser()

    # parser.add_argument("--experiment", type=str, default='1__debug', help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='3__performance_modes', help="name of the experiment")
    parser.add_argument("--experiment", type=str, default='4__label_collison', help="name of the experiment")

    parser.add_argument("--fold", type=int, default=0, help="5-fold index: choose between 0, 1, 2, 3 or 4.")

    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs")  # default=100
    parser.add_argument("--batch_size", type=int, default=2**5, help="size of the batches")  # default=32=2**5, 2**10
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")  # default=1e-3

    parser.add_argument("--n_virtual_channels", type=int, default=10)  # default=59
    parser.add_argument("--convolution_size", type=int, default=64)  # default=16; 64
    parser.add_argument("--pool_size", type=int, default=16)  # default=8; 16
    parser.add_argument("--n_hidden_channels", type=int, default=8)  # default=8,
    parser.add_argument("--n_time_series", type=int, default=320)  # default=320 # number of time points
    parser.add_argument("--n_channels", type=int, default=59)  # default=59 for EEG

    hparams = parser.parse_args()
    main(hparams)

# An epoch is a measure of the number of times all training data is used once to update the parameters.
#
# The actual batch size that we choose depends on many things. We want our batch size to be large enough
# to not be too "noisy", but not so large as to make each iteration too expensive to run.
#
# People often choose batch sizes of the form  n=2^k  so that it is easy to half or double the batch size.
# We'll choose a batch size of 32 and train the network again.
