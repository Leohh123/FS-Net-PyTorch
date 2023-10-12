import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, feature_dim=512, dropout=0.2):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(
                in_features=feature_dim*4,
                out_features=feature_dim*2
            ),
            nn.SELU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=feature_dim*2,
                out_features=feature_dim
            ),
            nn.SELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, encoder_feature, decoder_feature):
        consistency = encoder_feature * decoder_feature
        difference = torch.abs(encoder_feature - decoder_feature)
        compound_feature = torch.cat([
            encoder_feature,
            decoder_feature,
            consistency,
            difference
        ], dim=1)
        compressed_feature = self.compress(compound_feature)
        return compressed_feature
