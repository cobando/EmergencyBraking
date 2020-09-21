import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from sklearn.metrics import f1_score

class F1score(_Loss):

    def __init__(self):
        super(F1score, self).__init__()

    def forward(self, yp, y):
        yp_thresholded = (torch.sigmoid(yp) > 0.5) * 1.0
        # yp_thresholded = (yp > 0.5) * 1.0
        try:
            f1score = f1_score(y.numpy(), yp_thresholded.numpy())
            # f1score = f1_score(y.numpy(), yp.numpy())
        except ValueError:
            f1score = float("nan")

        return torch.from_numpy(np.array(f1score))
