import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, unpack_sequence

from .embedding import EmbeddingLayer
from .encoder import EncoderLayer
from .decoder import DecoderLayer
from .dense import DenseLayer
from .classification import ClassificationLayer
from .reconstruction import ReconstructionLayer

from utils.common import elementwise_apply


class FSNet(nn.Module):
    def __init__(self, n_app=19, max_value=5000, emb_dim=16, n_layer=2, hidden_size=128, dropout=0.2):
        super().__init__()
        self.embedding = EmbeddingLayer(emb_dim=emb_dim)
        self.encoder = EncoderLayer(
            input_dim=emb_dim,
            n_layer=n_layer,
            hidden_size=hidden_size,
            dropout=dropout
        )
        self.decoder = DecoderLayer(
            input_dim=hidden_size*n_layer*2,
            n_layer=n_layer,
            hidden_size=hidden_size,
            dropout=dropout
        )
        self.dense = DenseLayer(
            feature_dim=hidden_size*n_layer*2,
            dropout=dropout
        )
        self.classify = ClassificationLayer(
            input_dim=hidden_size*n_layer*2,
            n_class=n_app
        )
        self.reconstruct = ReconstructionLayer(
            input_dim=hidden_size*2,
            n_class=max_value
        )

    def forward(self, input):
        # print('input', input)
        if not isinstance(input, list):
            input = [input]
        input_ps = pack_sequence(input)
        # print('input_ps', input_ps)
        lengths = [tensor.shape[0] for tensor in input]
        # print('lengths', lengths)
        embedded_ps = elementwise_apply(self.embedding, input_ps)
        encoder_feature = self.encoder(embedded_ps)
        decoder_feature, decoder_output_ps = self.decoder(
            encoder_feature, lengths)
        # print('decoder_feature', decoder_feature, decoder_feature.shape)
        # decoder_output = unpack_sequence(decoder_output_ps)
        # print('decoder_output', decoder_output, len(decoder_output), decoder_output[0].shape)
        compressed_feature = self.dense(encoder_feature, decoder_feature)
        app_prob = self.classify(compressed_feature)
        value_prob_ps = self.reconstruct(decoder_output_ps)
        value_prob = unpack_sequence(value_prob_ps)
        return app_prob, value_prob
