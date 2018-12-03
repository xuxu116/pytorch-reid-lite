#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao CHEN (chaochancs@gmail.com)
# Created On: 2017-08-11
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.

    """
    def __init__(self, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


