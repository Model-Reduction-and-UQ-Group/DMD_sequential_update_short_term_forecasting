
import torch
import numpy as np
import scipy.io
#import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=1e-20):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def normalize(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def denormalize(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class MaxMinNormalizer(object):
    def __init__(self, x, eps=1e-20):
        super(MaxMinNormalizer, self).__init__()

        self.min = torch.min(x)
        self.max = torch.max(x)
        self.eps = eps

    def normalize(self, x):
        x = (x - self.min) / (self.max - self.min + self.eps)
        return x

    def denormalize(self, x):
        x = x * (self.max - self.min + self.eps) + self.min
        return x

    def cuda(self):
        self.min = self.min.cuda()
        self.max = self.max.cuda()

    def cpu(self):
        self.min = self.min.cpu()
        self.max = self.max.cpu()
