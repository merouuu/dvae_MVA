#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal ECG Dataset Loader for DVAE/VRNN
Labels textuels -> classes entières automatiquement
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


###########################################################
# 1) DATALOADER PRINCIPAL
###########################################################

def build_dataloader(cfg):

    seed = cfg.getint('DataFrame', 'seed')
    np.random.seed(seed)
    data_path = cfg.get('User', 'data_path')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    seq_len = cfg.getint('DataFrame', 'sequence_len')
    use_random_seq = cfg.getboolean('DataFrame', 'use_random_seq')

    # Chargement
    data = np.load(data_path, allow_pickle=True)
    X = data["X_fast"]     # shape (N, 300)
    Y = data["y_fast"]     # strings !

    N = len(X)

    # Shuffle global
    if shuffle:
        perm = np.random.permutation(N)
        X = X[perm]
        Y = Y[perm]

    # Split 70 / 30
    split = int(0.7 * N)
    train_X, train_Y = X[:split], Y[:split]
    val_X,   val_Y   = X[split:], Y[split:]

    # Création datasets
    if use_random_seq:
        train_dataset = ECGDatasetRandom(train_X, train_Y, seq_len)
        val_dataset   = ECGDatasetRandom(val_X, val_Y, seq_len)
    else:
        train_dataset = ECGDatasetFull(train_X, train_Y, seq_len)
        val_dataset   = ECGDatasetFull(val_X, val_Y, seq_len)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)



###########################################################
# Utility : map texte -> entier
###########################################################

def build_label_map(Y):
    class2id = {}
    next_id = 0
    for y in Y:
        if y not in class2id:
            class2id[y] = next_id
            next_id += 1
    return class2id



###########################################################
# 2) Dataset Full
###########################################################

class ECGDatasetFull(Dataset):

    def __init__(self, X, Y, seq_len):
        self.seq_len = seq_len

        # mapping texte -> entier
        self.class2id = build_label_map(Y)

        self.data = []
        self.labels = []

        for x, y in zip(X, Y):
            x = x.astype(np.float32)

            # normalisation
            m = np.max(np.abs(x))
            if m > 0:
                x = x / m

            # segmentation
            n = len(x) // seq_len
            for i in range(n):
                seg = x[i*seq_len:(i+1)*seq_len]
                seg = seg[:, None]  # (seq_len, 1)

                self.data.append(seg)
                self.labels.append(self.class2id[y])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), torch.tensor(self.labels[idx]).long()



###########################################################
# 3) Dataset Random
###########################################################

class ECGDatasetRandom(Dataset):

    def __init__(self, X, Y, seq_len):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len

        # mapping texte -> entier
        self.class2id = build_label_map(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx]

        # normalisation
        m = np.max(np.abs(x))
        if m > 0:
            x = x / m

        # tirage aléatoire
        if len(x) > self.seq_len:
            start = np.random.randint(0, len(x) - self.seq_len)
            x = x[start:start+self.seq_len]
        else:
            x = np.pad(x, (0, self.seq_len - len(x)))

        x = x[:, None]  # (seq_len,1)

        return torch.tensor(x).float(), torch.tensor(self.class2id[y]).long()
