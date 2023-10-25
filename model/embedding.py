import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, emb_dim=16):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, x: torch.LongTensor):
        bit_features = [((x >> i) & 1).float() for i in range(self.emb_dim)]
        embedded = torch.stack(bit_features, dim=1)
        return embedded
