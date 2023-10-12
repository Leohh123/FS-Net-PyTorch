import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, n_layer=2, hidden_size=128, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layer,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, encoder_feature, lengths):
        # print('encoder_feature', encoder_feature, encoder_feature.shape)
        inputs = [encoder_feature[i].repeat(length, 1)
                  for i, length in enumerate(lengths)]
        inputs_ps = pack_sequence(inputs)
        output_ps, hn = self.gru(inputs_ps)
        # print('======== output ========')
        # print(output)
        # print('======== hidden ========')
        # print(hn)
        # print('hn', hn, hn.shape)
        decoder_feature = hn.permute(1, 0, 2).flatten(start_dim=1)
        return decoder_feature, output_ps
