import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common import elementwise_apply


class ReconstructionLayer(nn.Module):
    def __init__(self, input_dim, n_class=5000):
        super().__init__()
        self.theta = nn.Parameter(torch.rand(input_dim, n_class) - 0.5)
        self.bias = nn.Parameter(torch.rand(n_class) - 0.5)
        self.softmax = nn.Softmax(dim=1)

    def get_prob(self, x):
        result = torch.exp(x @ self.theta + self.bias)
        prob = self.softmax(result)
        return prob

    def forward(self, decoder_output_ps):
        value_prob_ps = elementwise_apply(self.get_prob, decoder_output_ps)
        return value_prob_ps
