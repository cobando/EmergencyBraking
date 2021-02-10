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
# from in_out.datasets_sampled import load_dataset
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

        (self.dataset_train, self.dataset_val, self.dataset_test) = load_dataset(hparams.fold,hparams.window_size)
        self.loader_train = DataLoader(self.dataset_train, batch_size=hparams.batch_size, shuffle=True,
                                       num_workers=3, pin_memory=True, drop_last=True)
        self.loader_val = DataLoader(self.dataset_val, batch_size=hparams.batch_size, shuffle=False,
                                     num_workers=3, pin_memory=True, drop_last=False)
        self.loader_test = DataLoader(self.dataset_test, batch_size=hparams.batch_size, shuffle=False,
                                      num_workers=3, pin_memory=True, drop_last=False)

        self.loss = BCEWithLogitsLoss()  # 0, 1
        # self.loss = MSELoss() # Regression

        # Plot curves
        self.metrics = {
            'spe': Specificity(),
            'sen': Sensitivity(),
            'auc': Auc(),
            'f1score': F1score()
        }
        # self.metrics = {
        #     'mae': L1Loss()
        # }

        self.normalizer = float(hparams.batch_size) / len(self.dataset_train) #  Could be a problem
        self.metrics_avg = {key: 0.0 for key in self.metrics.keys()}
        self.metrics_avg['loss'] = 0.0

        self.net = DreemNet(n_channels=hparams.n_channels, n_virtual_channels=hparams.n_virtual_channels,
                            convolution_size=hparams.convolution_size, pool_size=hparams.pool_size,
                            n_hidden_channels=hparams.n_hidden_channels, window_size=hparams.window_size)
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

    def validation_epoch_end(self, outputs):  # before called validation_end

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

    def test_epoch_end(self, outputs): # before called test_end
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
        return Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)  # Default 1e-2, maybe increase it?



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
    # parser.add_argument("--experiment", type=str, default='4__label_collison', help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='5__windowsize_label_braking', help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='6__filter_braking_collision', help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='7__label_braking_drive', help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='8__label_collison_balanced', help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='9__windowsize_label_collision', help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='10__windowsize_label_collision_balanced', help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='11__windowsize_label_collision_filtered', help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='12__windowsize_label_collision_filtered_balanced', help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='13__windowsize_label_collision_filtered_balanced_smallNN',
    #                     help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='14__windowsize_label_collision_filtered_balanced_bigNN',
    #                     help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='15__windowsize_label_collision_modes',
    #                     help="name of the experiment")
    # parser.add_argument("--experiment", type=str, default='16__windowsize_label_collision_ERPcnhls',
    #                     help="name of the experiment") # and with EOG (but it is not working yet)
    # parser.add_argument("--experiment", type=str, default='17__sampleDataKL_collision_driving',
    #                     help="name of the experiment") #
    parser.add_argument("--experiment", type=str, default='18__allsubjects_label_braking',
                        help="name of the experiment") #


    # parser.add_argument("--experiment", type=str, default='18__sampleDataKL_persubject_collision_driving',
    #                     help="name of the experiment")  #

    parser.add_argument("--fold", type=int, default=0, help="5-fold index: choose between 0, 1, 2, 3 or 4.")

    parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs")  # default=100 (I put 500), after studies = 10000
    parser.add_argument("--batch_size", type=int, default=2**7, help="size of the batches")  # default=32=2**5, after studies 2**9
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")  # default=1e-3, after studies 1e-5

    parser.add_argument("--n_virtual_channels", type=int, default=1)  # default=59
    parser.add_argument("--convolution_size", type=int, default=2**4)  # default2**4=16 = 80ms of record; paper:64 = 2**6
    parser.add_argument("--n_hidden_channels", type=int, default=2**3)  # default=8 = 2**3(same as in paper), (2**1 worked for small wsize)
    parser.add_argument("--pool_size", type=int, default=2**3)  # default=8; paper:16 = 2**4, (2**2 worked for small wsize)

    # parser.add_argument("--n_time_series", type=int, default=320)  # default=320 # number of time points
    parser.add_argument("--n_channels", type=int, default=59)  # default=59 for EEG, change for modes, or EMG, or others

    parser.add_argument("--window_size", type=int, default=320)  # default=320 for EEG, the size is in number of points, 1ponint = 5ms, 301 for MNE process data

    hparams = parser.parse_args()
    main(hparams)

# An epoch is a measure of the number of times all training data is used once to update the parameters.
#
# The actual batch size that we choose depends on many things. We want our batch size to be large enough
# to not be too "noisy", but not so large as to make each iteration too expensive to run.
#
# People often choose batch sizes of the form  n=2^k  so that it is easy to half or double the batch size.
# We'll choose a batch size of 32 and train the network again.
