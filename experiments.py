import matplotlib.pyplot as plt
import torch
import numpy as np
from dvae.dataset.ecg_dataset import build_dataloader, ECGDatasetFull
from dvae.utils import myconf

def load_train_val_sequences(cfg_path):
    """
    Reconstruit EXACTEMENT X_train / X_val tels qu'utilisés pendant l'entraînement DVAE.
    Reproduit :
        - le shuffle global
        - la division 70/30
        - la segmentation en séquences
    """

    # === Charger config.ini ===
    cfg = myconf()
    cfg.read(cfg_path)

    data_path = cfg.get("User", "data_path")
    seq_len   = cfg.getint("DataFrame", "sequence_len")
    shuffle   = cfg.getboolean("DataFrame", "shuffle")
    seed      = cfg.getint("DataFrame", "seed", fallback=0)

    # === Charger les données ECG ===
    data = np.load(data_path, allow_pickle=True)
    X = data["X_fast"]
    Y = data["y_fast"]

    N = len(X)

    # === Reproduire EXACTEMENT le shuffle global ===
    np.random.seed(seed)
    if shuffle:
        perm = np.random.permutation(N)
        X = X[perm]
        Y = Y[perm]

    # === Split 70/30 identique au LearningAlgorithm ===
    split = int(0.7 * N)
    train_X, train_Y = X[:split], Y[:split]
    val_X,   val_Y   = X[split:], Y[split:]

    # === Reproduire EXACTEMENT la segmentation DVAE ===
    train_set = ECGDatasetFull(train_X, train_Y, seq_len)
    val_set   = ECGDatasetFull(val_X, val_Y, seq_len)

    # === Construire les tensors finaux ===
    X_train = torch.stack([train_set[i][0] for i in range(len(train_set))])
    y_train = torch.tensor([train_set[i][1] for i in range(len(train_set))])

    X_val = torch.stack([val_set[i][0] for i in range(len(val_set))])
    y_val = torch.tensor([val_set[i][1] for i in range(len(val_set))])

    return X_train, y_train, X_val, y_val



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
