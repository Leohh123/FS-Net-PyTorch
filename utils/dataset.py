import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import json
import random


class DatasetFromJson(Dataset):
    def __init__(self, json_path):
        super().__init__()
        with open(json_path) as f:
            self.data = json.loads(f.read())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, seq = self.data[index]
        return {'label': label, 'seq': seq}


class DatasetForImbalance(Dataset):
    def __init__(self, action, metadata, json_path):
        super().__init__()
        self.action = action
        self.meta = metadata
        # train_count = np.array(self.meta[f'{self.action}_count'])
        # self.prob_of_app = 1 / train_count
        # self.prob_of_app = self.prob_of_app / self.prob_of_app.sum()
        # print('self.prob_of_app', self.prob_of_app)
        # print(len(self))
        with open(json_path) as f:
            self.seqs_of_app = json.loads(f.read())

    def __len__(self):
        return self.meta[f'n_{self.action}']

    def __getitem__(self, index):
        app_id = random.randint(0, self.meta['n_app'] - 1)
        seq = random.choice(self.seqs_of_app[app_id])
        return {'label': app_id, 'seq': seq}


def get_metadata(json_path):
    with open(json_path) as f:
        metadata = json.loads(f.read())
    return metadata


def collate_fn(batch):
    labels, seqs = [], []
    for sample in batch:
        labels.append(sample['label'])
        seqs.append(sample['seq'])
    seqs.sort(key=len, reverse=True)
    return {
        'label': torch.LongTensor(labels),
        'seq': [torch.LongTensor(seq) for seq in seqs],
    }
