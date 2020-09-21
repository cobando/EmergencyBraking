import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from sklearn.metrics import roc_auc_score

class Auc(_Loss):

    def __init__(self):
        super(Auc, self).__init__()

    def forward(self, yp, y):
        # yp_thresholded = (torch.sigmoid(yp) > 0.5) * 1.0
        # fpr, tpr, thresholds = metrics.roc_curve(y, yp_thresholded, pos_label=2)
        # auc = roc_auc_score(y.numpy(), yp_thresholded.numpy())
        try:
            auc = roc_auc_score(y.numpy(), torch.sigmoid(yp).numpy())
            # auc = roc_auc_score(y.numpy(), yp.numpy())
        except ValueError:
            auc = float("nan")

        return torch.from_numpy(np.array(auc))
