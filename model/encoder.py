import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, input_dim=16, n_layer=2, hidden_size=128, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layer,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, embedded_ps):
        output, hn = self.gru(embedded_ps)
        # print('======== output ========')
        # print(output)
        # print('======== hidden ========')
        # print(hn)
        encoder_feature = hn.permute(1, 0, 2).flatten(start_dim=1)
        return encoder_feature
