import matplotlib.pyplot as plt
import torch
import numpy as np
from dvae.dataset.music_dataset import MusicDataset
from dvae.utils import myconf
import pretty_midi
import IPython.display as ipd
from pretty_midi import PrettyMIDI

def load_train_val_sequences_music(cfg_path):
    """
    Reconstruit les tenseurs X_train / X_val pour la musique.
    
    Note : Comme le MusicDataset utilise du 'Random Crop', cette fonction
    génère un 'snapshot' fixe (une séquence extraite par morceau).
    Si vous relancez la fonction, les séquences changeront (sauf si seed fixée).
    """
    
    # === 1. Charger la config ===
    # Assurez-vous d'avoir importé myconf ou définissez-le
    
    cfg = myconf()
    cfg.read(cfg_path)
    
    data_path = r"C:\code\dvae_final\DVAE\data\bach_data.npz"
    seq_len   = cfg.getint('DataFrame', 'sequence_len')
    shuffle   = cfg.getboolean('DataFrame', 'shuffle')
    seed      = cfg.getint('DataFrame', 'seed')
    
    # === 2. Charger les données Brutes (.npz) ===
    # Le fichier contient un dictionnaire avec la clé 'data'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")

    raw_data = np.load(data_path, allow_pickle=True)['data']
    N = len(raw_data)
    
    # === 3. Reproduire le Shuffle Global ===
    # On fixe la seed pour que la séparation Train/Val soit toujours la même
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Attention : raw_data est un array d'objets (liste de matrices), 
    # on doit le mélanger avec numpy
    if shuffle:
        np.random.shuffle(raw_data)
        
    # === 4. Split 80/20 (Comme dans build_music_dataloader) ===
    split = int(0.8 * N)
    train_data = raw_data[:split]
    val_data   = raw_data[split:]
    
    # === 5. Instancier les Datasets ===
    train_ds = MusicDataset(train_data, seq_len)
    val_ds   = MusicDataset(val_data, seq_len)
    
    # === 6. Construire les Tensors Finaux ===
    # On itère sur le dataset pour extraire une séquence (crop) pour chaque morceau.
    # MusicDataset retourne (x, y) où y est dummy (0).
    
    print(f"Extraction des séquences musicales (Train: {len(train_ds)}, Val: {len(val_ds)})...")
    
    # On utilise torch.stack pour empiler les tenseurs individuels en un gros batch
    # X_train aura la forme (N_train, seq_len, 88)
    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    y_train = torch.stack([train_ds[i][1] for i in range(len(train_ds))]) # Que des 0
    
    X_val = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    y_val = torch.stack([val_ds[i][1] for i in range(len(val_ds))]) # Que des 0
    
    print(f"Terminé. X_train shape: {X_train.shape}")
    
    return X_train, y_train, X_val, y_val

def continue_generation_vrnn(model, split, total_len):
    device = model.device
    
    # Attention : on détecte la taille du batch depuis les états cachés enregistrés
    # model.h_full est de forme (Seq, Layers, Batch, Dim) ou (Seq, Batch, Dim) selon l'implémentation
    # Ici on assume que le batch est dimension 1 dans h_full[t]
    batch = model.h_full[0].shape[1] 

    # --- 1. Récupérer les VRAIS états à t = split-1 ---
    # Ces variables existent parce qu'on a lancé model(x_context) juste avant !
    h_t = model.h_full[split-1].clone()  # shape (num_layers, batch, dim_RNN)
    c_t = model.c_full[split-1].clone()  # shape (num_layers, batch, dim_RNN)

    # last-layer hidden state
    h_t_last = h_t[-1].unsqueeze(0)      # shape (1, batch, dim_RNN)

    # Préparation du buffer de sortie
    gen_len = total_len - split
    
    # x_dim est 88 pour la musique
    y_gen = torch.zeros(gen_len, batch, model.x_dim, device=device)

    # --- 2. GÉNÉRATION LIBRE EXACTE ---
    for t in range(gen_len):
        # 2.1 Prior : p(z_t | h_t_last)
        mu_p, logvar_p = model.generation_z(h_t_last)
        z_t = model.reparameterization(mu_p, logvar_p)

        # 2.2 Decode : x_t = p(x | z_t, h_t_last)
        feat_z = model.feature_extractor_z(z_t)
        y_t = model.generation_x(feat_z, h_t_last)

        y_gen[t] = y_t

        # 2.3 Mettre à jour RNN
        feat_x = model.feature_extractor_x(y_t) # C'est le fameux phi_x !
        h_t, c_t = model.recurrence(feat_x, feat_z, h_t, c_t)

        # 2.4 Actualiser h_t_last
        h_t_last = h_t[-1].unsqueeze(0)

    return y_gen

def play_midi(midi_path):
    """
    Lit un fichier MIDI dans un Notebook Jupyter/Colab.
    """
    try:
        # 1. Charger le MIDI
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # 2. Synthétiser l'audio
        # fs = Fréquence d'échantillonnage (44100Hz est standard pour l'audio)
        # Essayer d'utiliser fluidsynth pour un vrai son de piano
        try:
            # Cherche un soundfont commun sur Linux/Colab
            audio_data = pm.fluidsynth(fs=44100, sf2_path='/users/share/sounds/sf2/FluidR3_GM.sf2')
            print("Synthesizer: FluidSynth (Piano réaliste)")
        except:
            # Fallback : ondes simples (bip-bip) si fluidsynth n'est pas installé
            print("Synthesizer: Simple Sine Wave (Son basique)")
            audio_data = pm.synthesize(fs=44100)

        # 3. Créer le widget audio
        return ipd.Audio(audio_data, rate=44100)
        
    except Exception as e:
        print(f"Erreur de lecture : {e}")
        return None

def plot_piano_roll(piano_roll, title):
    plt.figure(figsize=(10, 4))
    # On transpose pour avoir le temps en X et les notes en Y
    # aspect='auto' permet d'étirer l'image pour qu'elle soit lisible
    plt.imshow(piano_roll.T, origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=1)
    plt.colorbar(label="Vélocité / Probabilité")
    plt.xlabel("Temps (frames)")
    plt.ylabel("Pitch (Notes MIDI)")
    plt.title(title)
    plt.show()


# Fonction utilitaire pour lire le MIDI dans le notebook
def play_midi_file(midi_file):
    try:
        pm = PrettyMIDI(midi_file)
        # fs=16000 est suffisant pour une pré-écoute rapide
        audio_data = pm.synthesize(fs=16000) 
        return ipd.Audio(audio_data, rate=16000)
    except Exception as e:
        print(f"Erreur audio : {e}")

def plot_reconstruction(x, y):
    """
    x : input (seq_len, 1, 1)
    y : output from VRNN (seq_len, 1, 1)
    """

    # Remove batch+channel dims → (seq_len,)
    x_np = x.squeeze().cpu().numpy()
    y_np = y.squeeze().cpu().numpy()

    plt.figure(figsize=(10,4))
    plt.plot(x_np, label="Original", linewidth=2)
    plt.plot(y_np, label="Reconstruction", linewidth=2, alpha=0.7)
    plt.title("VRNN Reconstruction")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

def continue_generation_vrnn_final(model, split, total_len):
    device = model.device
    # On garde la détection dynamique du batch (plus robuste que batch=1)
    batch = model.h_full[0].shape[1] 
    
    # --- Initialisation (Récupération des états cachés) ---
    h_t = model.h_full[split-1].clone()
    c_t = model.c_full[split-1].clone()
    h_t_last = h_t[-1].unsqueeze(0)

    gen_len = total_len - split
    y_gen = torch.zeros(gen_len, batch, model.x_dim, device=device)

    for t in range(gen_len):
        # 1. Prior
        mu_p, logvar_p = model.generation_z(h_t_last)
        z_t = model.reparameterization(mu_p, logvar_p)

        # 2. Decode
        feat_z = model.feature_extractor_z(z_t)
        
        # Génération de la sortie (Log-Power Spectrogram)
        y_t = model.generation_x(feat_z, h_t_last) 
        
        # Stockage (on garde le Log-Power pour la visualisation/audio)
        y_gen[t] = y_t

        # 3. Recurrence avec LOG -> LINEAR
        # ADAPTATION ICI : Au lieu du seuillage binaire (>0.5), 
        # on applique exp() pour repasser en échelle linéaire 
        # car feature_extractor_x attend du Linear Power.
        y_t_linear = torch.exp(y_t)
        
        feat_x = model.feature_extractor_x(y_t_linear) 
        
        # Mise à jour des états RNN
        h_t, c_t = model.recurrence(feat_x, feat_z, h_t, c_t)
        h_t_last = h_t[-1].unsqueeze(0)

    return y_gen

def plot_continuation(x, y_resynth, y_gen, split=300):
    """
    x         : original input (seq_len, 1, 1)
    y_resynth : analysis-resynthesis output (seq_len, 1, 1)
    y_gen     : generated continuation (gen_len, 1, 1)
    split     : index where free generation begins
    """

    # Convert to numpy
    x_np = x.squeeze().cpu().numpy()               # (300,)
    y_resynth_np = y_resynth.squeeze().cpu().numpy()
    y_gen_np = y_gen.squeeze().cpu().numpy()

    # Concatenate reconstruction + continuation
    y_full = np.concatenate([y_resynth_np[:split], y_gen_np], axis=0)

    plt.figure(figsize=(12,4))
    plt.plot(x_np, label="Original (input)", linewidth=2)
    plt.plot(y_full, label="Reconstruction + Continuation", linewidth=2, alpha=0.8)

    # Vertical red line
    plt.axvline(split, color='red', linestyle='--', linewidth=2,
                label=f"Switch to free generation @ {split}")

    plt.title("VRNN Continuation Generation")
    plt.xlabel("Time step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_continuation_ss(x, y_resynth, y_gen, split=300):
    """
    For VRNN_ss:
    x         : (seq_len, 1, 1)
    y_resynth : VRNN_ss reconstruction (seq_len, 1, 1)
    y_gen     : continuation (gen_len, 1, 1)
    split     : time index where continuation starts
    """

    # Convert tensors → numpy
    x_np = x.squeeze().cpu().numpy()           # (seq_len,)
    y_resynth_np = y_resynth.squeeze().cpu().numpy()
    y_gen_np = y_gen.squeeze().cpu().numpy()   # (gen_len,)

    # Combine reconstruction + generation
    y_full = np.concatenate([y_resynth_np[:split], y_gen_np], axis=0)

    plt.figure(figsize=(14, 4))
    
    # Plot original and reconstructed+continued signal
    plt.plot(x_np, label="Original (input)", linewidth=2)
    plt.plot(y_full, label="VRNN_ss reconstruction + continuation",
             linewidth=2, alpha=0.8)

    # Red vertical line
    plt.axvline(
        x=split,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Start of free generation (t={split})"
    )

    plt.title("VRNN_ss Continuation Generation")
    plt.xlabel("Time step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_latent_heatmap(z, t_start=0, t_end=300):
    """
    z       : latent tensor from VRNN, shape (seq_len, batch, z_dim)
    t_start : début de la fenêtre temporelle
    t_end   : fin de la fenêtre temporelle
    """

    # on squeeze batch dimension → (seq_len, z_dim)
    z = z.squeeze().cpu().numpy()       # shape = (seq_len, z_dim)

    # Sélection d’une fenêtre temporelle
    z_seg = z[t_start:t_end].T          # transpose → (z_dim, time)

    plt.figure(figsize=(8, 4))
    plt.imshow(z_seg, aspect='auto', cmap='viridis',
               interpolation='nearest')

    plt.colorbar(label="Latent value")
    plt.xlabel("Time")
    plt.ylabel("Latent dimension")

    plt.title(f"VRNN latent trajectories  (t={t_start}→{t_end})")
    plt.tight_layout()
    plt.show()


def plot_hidden_heatmap(h, t_start=0, t_end=300):
    """
    h       : hidden tensor from VRNN, shape (seq_len, batch, dim_RNN)
    t_start : début de la fenêtre temporelle
    t_end   : fin de la fenêtre temporelle
    """

    # enlever dimension batch → (seq_len, dim_RNN)
    h = h.squeeze().cpu().numpy()        # (seq_len, dim_RNN)

    # sélection temporelle + transpose → (dim_RNN, time)
    h_seg = h[t_start:t_end].T

    plt.figure(figsize=(10, 4))
    plt.imshow(
        h_seg,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    plt.colorbar(label="Hidden state value")
    plt.xlabel("Time")
    plt.ylabel("Hidden dimension")
    plt.title(f"LSTM hidden trajectories h(t)   (t={t_start}→{t_end})")
    plt.tight_layout()
    plt.show()


def plot_latent_mu_heatmap(z_mean, t_start=0, t_end=300):
    """
    z_mean : tensor VRNN (seq_len, batch, z_dim)
    """

    # Enlever la dimension batch
    z_mu = z_mean.squeeze().cpu().numpy()    # (seq_len, z_dim)

    # Sélection de la fenêtre temporelle
    z_slice = z_mu[t_start:t_end].T          # → (z_dim, time)

    plt.figure(figsize=(8, 4))
    plt.imshow(
        z_slice,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    plt.colorbar(label="z_mean value")
    plt.xlabel("Time")
    plt.ylabel("Latent dimension")
    plt.title(f"VRNN μ(t) latent trajectories   t=[{t_start}:{t_end}]")
    plt.tight_layout()
    plt.show()


def plot_latent_logvar_heatmap(z_logvar, t_start=0, t_end=300):
    """
    z_logvar : tensor VRNN (seq_len, batch, z_dim)
    """

    # Remove batch dimension → (seq_len, z_dim)
    z_lv = z_logvar.squeeze().cpu().numpy()

    # Window selection
    z_slice = z_lv[t_start:t_end].T     # → (z_dim, time)

    plt.figure(figsize=(8, 4))
    plt.imshow(
        z_slice,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    plt.colorbar(label="log-variance  log σ²(t)")
    plt.xlabel("Time")
    plt.ylabel("Latent dimension")
    plt.title(f"VRNN posterior log-variance  t=[{t_start}:{t_end}]")
    plt.tight_layout()
    plt.show()



def plot_mean_hidden_states(model, X, device, max_samples=64):
    """
    Compute and plot the AVERAGE hidden h(t) over many samples.
    
    X : numpy array of shape (N, 300)
    max_samples : how many samples to average over
    """

    # pick min(N, max_samples)
    N = min(len(X), max_samples)

    print(f"Processing {N} samples...")

    # Use lists to accumulate
    h_list = []
    zmu_list = []

    model.eval()

    for i in range(N):
        x = X[i]
        x = torch.tensor(x).float().unsqueeze(1).unsqueeze(1).to(device)  # (300,1,1)

        with torch.no_grad():
            _ = model(x)  # forward pass

        # Retrieve tensors
        h = model.h.squeeze(1).cpu().numpy()            # (300, dim_RNN)
        zmu = model.z_mean.squeeze(1).cpu().numpy()     # (300, z_dim)

        h_list.append(h)
        zmu_list.append(zmu)

    # Convert to arrays: shape (N, 300, dim_RNN)
    H = np.stack(h_list)       # (N, 300, dim_RNN)
    Z = np.stack(zmu_list)     # (N, 300, z_dim)

    # Compute MEAN
    H_mean = H.mean(axis=0).T   # → (dim_RNN, time)
    Z_mean = Z.mean(axis=0).T   # → (z_dim, time)

    print("Shapes:")
    print("  H_mean:", H_mean.shape)
    print("  Z_mean:", Z_mean.shape)

    # ---------- Plot MEAN h(t) ----------
    plt.figure(figsize=(12,5))
    plt.imshow(H_mean, aspect='auto', cmap='viridis')
    plt.colorbar(label="Mean hidden state value")
    plt.xlabel("Time")
    plt.ylabel("Hidden dimension")
    plt.title(f"Mean LSTM hidden trajectories over {N} samples")
    plt.tight_layout()
    plt.show()

    # ---------- Plot MEAN z_mean(t) ----------
    plt.figure(figsize=(12,5))
    plt.imshow(Z_mean, aspect='auto', cmap='viridis')
    plt.colorbar(label="Mean z_mu value")
    plt.xlabel("Time")
    plt.ylabel("Latent dimension")
    plt.title(f"Mean latent μ(t) over {N} samples")
    plt.tight_layout()
    plt.show()

    return H_mean, Z_mean


def plot_mean_hidden_states(model, X, device, max_samples=64):
    """
    Compute and plot the AVERAGE hidden h(t) over many samples.
    
    X : numpy array of shape (N, 300)
    max_samples : how many samples to average over
    """

    # pick min(N, max_samples)
    N = min(len(X), max_samples)

    print(f"Processing {N} samples...")

    # Use lists to accumulate
    h_list = []
    zmu_list = []

    model.eval()

    for i in range(N):
        x = X[i]
        x = torch.tensor(x).float().unsqueeze(1).unsqueeze(1).to(device)  # (300,1,1)

        with torch.no_grad():
            _ = model(x)  # forward pass

        # Retrieve tensors
        h = model.h.squeeze(1).cpu().numpy()            # (300, dim_RNN)
        zmu = model.z_mean.squeeze(1).cpu().numpy()     # (300, z_dim)

        h_list.append(h)
        zmu_list.append(zmu)

    # Convert to arrays: shape (N, 300, dim_RNN)
    H = np.stack(h_list)       # (N, 300, dim_RNN)
    Z = np.stack(zmu_list)     # (N, 300, z_dim)

    # Compute MEAN
    H_mean = H.mean(axis=0).T   # → (dim_RNN, time)
    Z_mean = Z.mean(axis=0).T   # → (z_dim, time)

    print("Shapes:")
    print("  H_mean:", H_mean.shape)
    print("  Z_mean:", Z_mean.shape)

    # ---------- Plot MEAN h(t) ----------
    plt.figure(figsize=(12,5))
    plt.imshow(H_mean, aspect='auto', cmap='viridis')
    plt.colorbar(label="Mean hidden state value")
    plt.xlabel("Time")
    plt.ylabel("Hidden dimension")
    plt.title(f"Mean LSTM hidden trajectories over {N} samples")
    plt.tight_layout()
    plt.show()

    # ---------- Plot MEAN z_mean(t) ----------
    plt.figure(figsize=(12,5))
    plt.imshow(Z_mean, aspect='auto', cmap='viridis')
    plt.colorbar(label="Mean z_mu value")
    plt.xlabel("Time")
    plt.ylabel("Latent dimension")
    plt.title(f"Mean latent μ(t) over {N} samples")
    plt.tight_layout()
    plt.show()

    return H_mean, Z_mean


def continue_generation_vrnn(model, split, total_len):

    device = model.device
    batch = 1

    # ================================================================
    # 1. Récupérer les VRAIS états à t = split-1
    # ================================================================
    
    # hidden states (all layers)
    h_t = model.h_full[split-1].clone()  # shape (num_layers, batch, dim_RNN)
    c_t = model.c_full[split-1].clone()  # shape (num_layers, batch, dim_RNN)

    # last-layer hidden state = used by prior + decoder
    h_t_last = h_t[-1].unsqueeze(0)      # shape (1, batch, dim_RNN)

    # On part d'un x_t initial = output reconstruit au split-1
    # comme dans forward() où y[t] = model.generation_x(...)
    y_prev = model.forward_output[split-1].unsqueeze(0) \
             if hasattr(model, "forward_output") else None

    # Mais y_prev N'EST PAS utilisé directement : le VRNN génère x_t from scratch
    # Ce var est inutile pour VRNN (contrairement à un ARNN)
    
    # Préparation du buffer de sortie
    gen_len = total_len - split
    y_gen = torch.zeros(gen_len, batch, model.x_dim, device=device)

    # ================================================================
    # 2. GÉNÉRATION LIBRE EXACTE
    # ================================================================
    for t in range(gen_len):

        # ---- 2.1 Prior : p(z_t | h_t_last)
        mu_p, logvar_p = model.generation_z(h_t_last)
        z_t = model.reparameterization(mu_p, logvar_p)  # (1, batch, z_dim)

        # ---- 2.2 Decode : x_t = p(x | z_t, h_t_last)
        feat_z = model.feature_extractor_z(z_t)
        y_t = model.generation_x(feat_z, h_t_last)       # (1, batch, x_dim)

        y_gen[t] = y_t

        # ---- 2.3 Mettre à jour RNN comme dans forward()
        feat_x = model.feature_extractor_x(y_t)
        h_t, c_t = model.recurrence(feat_x, feat_z, h_t, c_t)

        # ---- 2.4 Actualiser h_t_last = dernier layer
        h_t_last = h_t[-1].unsqueeze(0)

    return y_gen






###################### SS VRNN model #########################

import os, sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device =", device)

# ============================================================
#  DENSE BLOCK
# ============================================================

class Dense_Block(nn.Module):
    def __init__(self, in_dim, dense_list, activation='tanh'):
        super().__init__()

        act = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        layers = []
        dim = in_dim
        for h in dense_list:
            layers.append(nn.Linear(dim, h))
            layers.append(act)
            dim = h

        self.block = nn.Sequential(*layers)
        self.out_dim = dim

    def forward(self, x):
        return self.block(x)

# ============================================================
#  VRNN (version patchée schedule sampling)
# ============================================================

class VRNN_ss(nn.Module):
    def __init__(
        self,
        x_dim=1,
        z_dim=32,
        activation='tanh',
        dense_x=[128, 128],
        dense_z=[128, 128],
        dense_hx_z=[128, 128],
        dense_hz_x=[128, 128],
        dense_h_z=[128, 128],
        dim_RNN=128,
        num_RNN=1,
        dropout_p=0.0,
        beta=1.0,
        device="cuda"
    ):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.beta = beta
        self.device = device

        # schedule sampling : probabilité de teacher forcing
        self.ss_prob = 1.0

        self.feature_extractor_x = Dense_Block(x_dim, dense_x, activation)
        self.feature_extractor_z = Dense_Block(z_dim, dense_z, activation)

        # inference q(z|x,h)
        self.inference_xh_to_z = Dense_Block(
            self.feature_extractor_x.out_dim + dim_RNN,
            dense_hx_z,
            activation
        )
        self.inference_mean = nn.Linear(self.inference_xh_to_z.out_dim, z_dim)
        self.inference_logvar = nn.Linear(self.inference_xh_to_z.out_dim, z_dim)

        # prior p(z|h)
        self.prior_h_to_z = Dense_Block(dim_RNN, dense_h_z, activation)
        self.prior_mean = nn.Linear(self.prior_h_to_z.out_dim, z_dim)
        self.prior_logvar = nn.Linear(self.prior_h_to_z.out_dim, z_dim)

        # decoder p(x|z,h)
        self.generator_hz_to_x = Dense_Block(
            self.feature_extractor_z.out_dim + dim_RNN,
            dense_hz_x,
            activation
        )
        self.generator_mean = nn.Linear(self.generator_hz_to_x.out_dim, x_dim)

        self.dim_RNN = dim_RNN
        self.num_RNN = num_RNN
        self.rnn = nn.LSTM(
            self.feature_extractor_x.out_dim + self.feature_extractor_z.out_dim,
            dim_RNN,
            num_layers=num_RNN,
            dropout=dropout_p,
            batch_first=False
        )

        self.to(device)

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        

    def inference_z(self, fx, h_prev):
        hx = torch.cat([fx, h_prev], dim=-1)
        hidden = self.inference_xh_to_z(hx)
        mu = self.inference_mean(hidden)
        logvar = self.inference_logvar(hidden)
        return mu, logvar

    def generation_z(self, h_prev):
        hidden = self.prior_h_to_z(h_prev)
        mu = self.prior_mean(hidden)
        logvar = self.prior_logvar(hidden)
        return mu, logvar

    def generation_x(self, fz, h_prev):
        hz = torch.cat([fz, h_prev], dim=-1)
        hidden = self.generator_hz_to_x(hz)
        return self.generator_mean(hidden)

    def forward(self, x, training=True):
        T, B, _ = x.size()
        h_t = torch.zeros(self.num_RNN, B, self.dim_RNN, device=self.device)
        c_t = torch.zeros(self.num_RNN, B, self.dim_RNN, device=self.device)

        h_list = []
        recon_list = []
        z_list = []
        mu_q_list = []
        logvar_q_list = []
        mu_p_list = []
        logvar_p_list = []

        y_prev = None

        for t in range(T):
            x_t = x[t]

            # =============== Schedule Sampling ==================
            if training and t > 0 and self.ss_prob < 1.0:
                if torch.rand(1, device=x.device).item() > self.ss_prob:
                    if y_prev is not None:
                        x_t = y_prev.detach()

            fx = self.feature_extractor_x(x_t)
            h_prev = h_t[-1]

            mu_q, logvar_q = self.inference_z(fx, h_prev)
            z_t = self.reparameterization(mu_q, logvar_q)
            fz = self.feature_extractor_z(z_t)

            mu_p, logvar_p = self.generation_z(h_prev)

            y_t = self.generation_x(fz, h_prev)
            y_prev = y_t

            rnn_in = torch.cat([fx, fz], dim=-1).unsqueeze(0)
            _, (h_t, c_t) = self.rnn(rnn_in, (h_t, c_t))
            h_list.append(h_t[-1].clone())


            recon_list.append(y_t)
            z_list.append(z_t)
            mu_q_list.append(mu_q)
            logvar_q_list.append(logvar_q)
            mu_p_list.append(mu_p)
            logvar_p_list.append(logvar_p)

        recon = torch.stack(recon_list)
        z = torch.stack(z_list)
        mu_q = torch.stack(mu_q_list)
        logvar_q = torch.stack(logvar_q_list)
        mu_p = torch.stack(mu_p_list)
        logvar_p = torch.stack(logvar_p_list)

        # ----- store internal states for continuation -----
        self.h = torch.stack(h_list)          # (T, B, dim_RNN)
        self.z = z                            # <----- ADDED
        self.z_mean = mu_q
        self.z_logvar = logvar_q
        self.z_mean_p = mu_p
        self.z_logvar_p = logvar_p

        return recon, z, mu_q, logvar_q, mu_p, logvar_p


    def encode(self, x):
        """
        Utilisé pour forecasting.
        """
        y, z, mu_q, logvar_q, _, _ = self.forward(x, training=False)
        return z, mu_q, logvar_q
    

    
def generate_from_context(model, context_seq, pred_steps=300, device="cuda", stochastic=True):
    """
    Génère une séquence ECG en partant d'un contexte donné.
    Utilise:
       - encode(x_1:T) -> z_1:T
       - replay z dans le RNN pour obtenir h_T
       - génération future avec p(z|h)
    """
    model.eval()

    # Convertir le contexte en tenseur VRNN (T,1,1)
    x = torch.tensor(context_seq, dtype=torch.float32, device=device)
    x = x.unsqueeze(1).unsqueeze(-1)  # (T,1,1)
    T = x.size(0)

    with torch.no_grad():

        # ===== 1) Encoder le contexte : z_1:T =====
        z_hist, mu_hist, logvar_hist = model.encode(x)   # (T,1,z_dim)

        # ===== 2) Rejouer le contexte dans le RNN =====
        num_layers, B, h_dim = model.num_RNN, 1, model.dim_RNN
        h_t = torch.zeros(num_layers, B, h_dim, device=device)
        c_t = torch.zeros(num_layers, B, h_dim, device=device)

        fx_hist = model.feature_extractor_x(x)        # (T,1,dim_fx)
        fz_hist = model.feature_extractor_z(z_hist)   # (T,1,dim_fz)

        for t in range(T):
            # fx_hist[t] : (1, dim_fx), fz_hist[t] : (1, dim_fz)
            rnn_input = torch.cat([fx_hist[t], fz_hist[t]], dim=-1).unsqueeze(0)  # (1,1,dim_total)
            _, (h_t, c_t) = model.rnn(rnn_input, (h_t, c_t))

        # Dernier point du contexte comme point de départ
        x_t = x[-1]   # (1,1)

        # ===== 3) Génération future =====
        generated = []

        for _ in range(pred_steps):

            # φ_x(x_t)  -> (1, dim_fx)
            fx = model.feature_extractor_x(x_t)

            # dernier état h_{t-1} : garder la même forme que dans forward, donc 2D (B, dim_h)
            h_prev = h_t[-1]                  # (1, h_dim)  <-- plus de unsqueeze(0) ici

            # prior p(z|h)  -> (1, z_dim)
            mu_p, logvar_p = model.generation_z(h_prev)
            z_t = model.reparameterization(mu_p, logvar_p) if stochastic else mu_p

            # φ_z(z_t) -> (1, dim_fz)
            fz = model.feature_extractor_z(z_t)

            # p(x|z,h) -> (1, x_dim)
            y_t = model.generation_x(fz, h_prev)     # (1,1) si x_dim=1

            # extraire la valeur scalaire
            generated.append(y_t.item())

            # update RNN : concaténer deux tenseurs 2D -> 2D, puis unsqueeze(0) pour la dimension temporelle
            rnn_input = torch.cat([fx, fz], dim=-1).unsqueeze(0)  # (1,1,dim_total)
            _, (h_t, c_t) = model.rnn(rnn_input, (h_t, c_t))

            # autoregressif : x_t ← y_t (toujours (1,1))
            x_t = y_t

    return np.array(generated, dtype=np.float32)
