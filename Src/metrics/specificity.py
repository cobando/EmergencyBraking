import numpy as np
import torch
from torch.nn.modules.loss import _Loss

class Specificity(_Loss):

    def __init__(self):
        super(Specificity, self).__init__()

    def forward(self, yp, y):

        yp_thresholded = (torch.sigmoid(yp) > 0.5) * 1.0
        tn = 1.0 * torch.sum((yp_thresholded + y) == 0)  # true negatives
        fp = 1.0 * torch.sum((torch.abs(yp_thresholded - 1) + y) == 0)  # false positives
        specificity = tn / (tn + fp)

        return specificity
