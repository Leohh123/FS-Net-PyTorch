import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import unpack_sequence


def classification_loss(app_prob, app_gt, class_weights=None):
    # print('app_prob', app_prob, app_prob.shape)
    # print('app_gt', app_gt, app_gt.shape)
    return F.nll_loss(app_prob.log(), app_gt, weight=class_weights)


def reconstruction_loss(value_prob, value_gt):
    # print('value_prob', value_prob, len(value_prob), value_prob[0].shape)
    # print('value_gt', value_gt, len(value_gt))
    value_prob_cat = torch.cat(value_prob)
    value_gt_cat = torch.cat(value_gt)
    # print(value_prob_cat.shape, value_gt_cat.shape)
    return F.nll_loss(value_prob_cat.log(), value_gt_cat)
