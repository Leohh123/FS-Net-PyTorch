import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLayer(nn.Module):
    def __init__(self, input_dim, n_class=19):
        super().__init__()
        self.theta = nn.Parameter(torch.rand(input_dim, n_class) - 0.5)
        self.bias = nn.Parameter(torch.rand(n_class) - 0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, compressed_feature):
        result = compressed_feature @ self.theta + self.bias
        app_prob = self.softmax(result)
        return app_prob
