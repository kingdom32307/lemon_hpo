import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Transfer(nn.Module):
    """
    Parameters
    ----------
    batch_size: int
        batch size of image dataset
    lr: float
        The learning rate of inner weight parameter of CNN.
    momentum: float
        momentum coefficient for Stochastic Gradient Descent (SGD)
    weight_decay: float
        the coefficients of a regularization term for cross entropy
    nesterov: bool
        Whether using nesterov or not in SGD.
    rho: float
        Hyperparameter in SAM.
    epochs: int
        The number of training throughout one learning process.
    lr_step: list of float
        When to decrease the learning rate.
        The learning rate will decline at lr_step[k] * epochs epoch.
    lr_decay: float
        How much make learning rate decline at epochs * lr_step[k]
    n_cls: int
        The number of classes on a given task.
    """

    def __init__(self,
                 batch_size=32,
                 lr=1.0e-2,
                 momentum=0.8,
                 weight_decay=1.0e-3,
                 nesterov=True,
                 lr_decay=0.2,
                 lr_step=[0.3, 0.6, 0.8],
                 rho=0.05,
                 epochs=200,
                 opt="sam",
                 gcam=False,
                 mtra="resnet50",
                 n_cls=4
                 ):
        super(Transfer, self).__init__()

        # Hyperparameter Configuration for CNN.
        self.batch_size = int(batch_size)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.lr_step = [int(step * self.epochs) for step in lr_step]
        self.nesterov = nesterov
        self.lr_decay = lr_decay
        self.rho = rho
        self.opt = opt
        self.mtra = mtra
        self.gcam = gcam

