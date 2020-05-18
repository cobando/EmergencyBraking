import numpy as np
import torch
from torch.nn.modules.loss import _Loss

class Sensitivity(_Loss):

    def __init__(self):
        super(Sensitivity, self).__init__()

    def forward(self, yp, y):

        yp_thresholded = (torch.sigmoid(yp) > 0.5) * 1.0
        tp = 1.0 * torch.sum((yp_thresholded + y) == 2)  # true positives
        fn = 1.0 * torch.sum((torch.abs(y - 1) + yp_thresholded) == 0)
        sensitivity = tp / (tp + fn)

        return sensitivity
