#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Dataset Loader compatible with DVAE/VRNN architecture
Author: ChatGPT (adapted from DVAE-speech)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


###########################################################
# 1) FONCTION PRINCIPALE POUR CRÉER LES DATALOADERS
###########################################################

def build_dataloader(cfg):

    # Lecture paramètres généraux
    data_path = cfg.get('User', 'data_path')  # chemin vers ecg_fast_data.npz
    batch_size = cfg.getint('DataFrame', 'batch_size')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')
    use_random_seq = cfg.getboolean('DataFrame', 'use_random_seq')

    # Chargement du fichier NPZ
    data = np.load(data_path, allow_pickle=True)
    X = data["X_fast"].tolist()
    Y = data["y_fast"].tolist()
    META = data["meta_fast"].tolist()

    assert len(X) == len(Y)

    N = len(X)

    # Shuffle global si demandé
    if shuffle:
        perm = np.random.permutation(N)
        X = [X[i] for i in perm]
        Y = [Y[i] for i in perm]
        META = [META[i] for i in perm]

    # Split 70/30
    split = int(0.7 * N)
    train_X, train_Y, train_META = X[:split], Y[:split], META[:split]
    val_X, val_Y, val_META = X[split:], Y[split:], META[split:]

    # Création dataset
    if use_random_seq:
        train_dataset = ECGDatasetRandom(train_X, train_Y, sequence_len)
        val_dataset = ECGDatasetRandom(val_X, val_Y, sequence_len)
    else:
        train_dataset = ECGDatasetFull(train_X, train_Y, sequence_len)
        val_dataset = ECGDatasetFull(val_X, val_Y, sequence_len)

    # Nombre d'échantillons utilisables
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader, train_num, val_num



###########################################################
# 2) VERSION FULL : découpe déterministe de l'ensemble ECG
###########################################################

class ECGDatasetFull(Dataset):

    def __init__(self, X_list, Y_list, seq_len):
        """
        Découpe toutes les séries ECG en segments non superposés
        """
        self.seq_len = seq_len
        self.data = []
        self.labels = []

        for x, y in zip(X_list, Y_list):
            x = np.array(x)

            # normalisation
            if np.max(np.abs(x)) > 0:
                x = x / np.max(np.abs(x))

            # segmenter en blocs de longueur seq_len
            n_segments = len(x) // seq_len

            for i in range(n_segments):
                seg = x[i * seq_len : (i + 1) * seq_len]
                self.data.append(seg.astype(np.float32))
                self.labels.append(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return torch.tensor(x).float(), torch.tensor(y).long()



###########################################################
# 3) VERSION RANDOM : extrait un segment aléatoire à chaque appel
###########################################################

class ECGDatasetRandom(Dataset):

    def __init__(self, X_list, Y_list, seq_len):
        self.seq_len = seq_len
        self.X = X_list
        self.Y = Y_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = np.array(self.X[index])
        y = self.Y[index]

        # normalisation
        if np.max(np.abs(x)) > 0:
            x = x / np.max(np.abs(x))

        # tirer un segment random si série assez longue
        if len(x) > self.seq_len:
            start = np.random.randint(0, len(x) - self.seq_len)
            x = x[start:start + self.seq_len]
        else:
            # pad si trop court
            pad = self.seq_len - len(x)
            x = np.pad(x, (0, pad))

        return torch.tensor(x).float(), torch.tensor(y).long()
