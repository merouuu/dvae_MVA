#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music Dataset Loader for DVAE/VRNN (Polyphonic MIDI)
Gère le parsing MIDI -> Piano Roll -> Tensor
"""

import os
import numpy as np
import torch
import pretty_midi
from torch.utils.data import Dataset, DataLoader

###########################################################
# 1) OUTIL DE CRÉATION DU DATASET (A lancer une fois)
###########################################################

def create_midi_dataset_file(source_folder, output_path, fs=16, min_len=32):
    """
    Scanne un dossier de .mid, convertit tout en une liste de matrices numpy,
    et sauvegarde le tout dans un fichier compressé .npz.
    
    Args:
        source_folder (str): Dossier contenant les fichiers .mid
        output_path (str): Fichier de sortie (ex: 'bach_dataset.npz')
        fs (int): Fréquence d'échantillonnage (frames par seconde)
        min_len (int): Longueur minimale en frames pour garder un morceau
    """
    print(f"--- Création du dataset depuis {source_folder} ---")
    
    all_piano_rolls = []
    file_list = [f for f in os.listdir(source_folder) if f.endswith('.mid') or f.endswith('.midi')]
    
    count = 0
    for f in file_list:
        full_path = os.path.join(source_folder, f)
        try:
            # 1. Chargement MIDI
            pm = pretty_midi.PrettyMIDI(full_path)
            
            # 2. Conversion Piano Roll (128, T)
            # fs=16 signifie 1 frame = 62.5ms (double-croche approx à 120bpm)
            pr = pm.get_piano_roll(fs=fs) 
            
            # 3. Transpose (T, 128)
            pr = pr.T 
            
            # 4. Crop des notes (Piano standard 88 touches : 21 à 108)
            # On garde 88 dimensions fixes
            pr = pr[:, 21:109]
            
            # 5. Normalisation "Soft" (Velocity 0-127 -> 0.0-1.0)
            # Pour VRNN Gaussien, on veut du continu.
            pr = pr.astype(np.float32) / 127.0
            
            # Filtre : on ne garde que si assez long
            if pr.shape[0] >= min_len:
                all_piano_rolls.append(pr)
                count += 1
                
        except Exception as e:
            print(f"Erreur sur {f}: {e}")
            
    print(f"Terminé : {count} morceaux traités.")
    
    # Sauvegarde en format 'object' car les longueurs sont variables
    np.savez_compressed(output_path, data=np.array(all_piano_rolls, dtype=object))
    print(f"Dataset sauvegardé sous : {output_path}")


###########################################################
# 2) DATALOADER PRINCIPAL
###########################################################

def build_music_dataloader(cfg):
    
    seed = cfg.getint('DataFrame', 'seed')
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data_path = cfg.get('User', 'data_path') # Doit pointer vers le .npz créé ci-dessus
    batch_size = cfg.getint('DataFrame', 'batch_size')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    
    # Paramètres spécifiques Musique
    seq_len = cfg.getint('DataFrame', 'sequence_len') # Ex: 64 ou 100
    
    # Chargement du fichier préparé
    # On s'attend à un dictionnaire avec une clé 'data' contenant une liste de tableaux
    raw_data = np.load(data_path, allow_pickle=True)['data']
    
    N = len(raw_data)
    
    # Shuffle global des morceaux
    if shuffle:
        np.random.shuffle(raw_data)
        
    # Split 80 / 20 (Classique pour la musique)
    split = int(0.8 * N)
    train_data = raw_data[:split]
    val_data   = raw_data[split:]
    
    # Création datasets
    # On utilise toujours le mode "Random Crop" pour la musique car les morceaux
    # sont très longs et on veut varier les contextes.
    train_dataset = MusicDataset(train_data, seq_len)
    val_dataset   = MusicDataset(val_data, seq_len)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=shuffle, num_workers=num_workers)
                              
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
                            
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


###########################################################
# 3) DATASET CLASSE
###########################################################

class MusicDataset(Dataset):
    def __init__(self, data_list, seq_len):
        """
        data_list: Liste de numpy arrays (T_variable, 88)
        seq_len: Longueur de la fenêtre d'entrainement (T_fixe)
        """
        self.data_list = data_list
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # Récupérer le morceau entier (T, 88)
        full_song = self.data_list[idx]
        total_len = full_song.shape[0]
        
        # 1. Random Crop (Extraction d'une séquence aléatoire)
        # Si le morceau est plus grand que seq_len, on coupe au hasard
        if total_len > self.seq_len:
            max_start = total_len - self.seq_len
            start = np.random.randint(0, max_start)
            segment = full_song[start : start + self.seq_len]
        
        # Si le morceau est trop court (rare si bien filtré), on pad avec des 0 (silence)
        else:
            padding = np.zeros((self.seq_len - total_len, 88), dtype=np.float32)
            segment = np.concatenate([full_song, padding], axis=0)
            
        # 2. Conversion Tensor
        x = torch.from_numpy(segment).float() 
        
        # Output : (seq_len, 88)
        # Note: Pas besoin de 'y' (labels) pour un VRNN génératif pur,
        # mais si ton code attend un tuple (x, y), on peut renvoyer (x, x) ou (x, 0)
        # Pour rester compatible avec ton code d'entrainement ECG qui attend (x, y):
        
        # Dummy label (0) car on fait de la génération non-supervisée (ou x est la cible)
        y = torch.tensor(0).long() 
        
        return x, y

# --- Zone de test ---
if __name__ == "__main__":
    # Exemple d'utilisation pour créer le fichier dataset.npz
    # create_midi_dataset_file("./midi_source/", "./data/bach_dataset.npz", fs=16)
    pass