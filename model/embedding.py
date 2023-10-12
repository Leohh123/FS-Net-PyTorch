import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, emb_dim=16, max_row=5000):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=max_row,
            embedding_dim=emb_dim
        )

    def forward(self, x):
        return self.embedding(x)
