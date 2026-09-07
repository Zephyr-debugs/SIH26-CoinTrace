"""
Phase 5 - Autoencoder behavioral anomaly detector.

Trains a small feedforward autoencoder, unsupervised, directly on the
entity feature matrix (CPU-trainable in seconds at hackathon scale - no
GPU dependency, matching the proposal's "commodity hardware" constraint).
Anomaly score = reconstruction error, normalized to [0, 1].

Requires PyTorch (see requirements.txt). If torch isn't installed, this
module raises a clear ImportError with install instructions rather than
failing cryptically deep inside detection/fusion.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cointrace import config

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _Autoencoder(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, n_features: int, hidden_dim: int):
        super().__init__()
        bottleneck = max(2, hidden_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_autoencoder(
    feat_df: pd.DataFrame,
    hidden_dim: int = config.AUTOENCODER_HIDDEN_DIM,
    epochs: int = config.AUTOENCODER_EPOCHS,
    lr: float = config.AUTOENCODER_LR,
    random_state: int = config.RANDOM_SEED,
) -> pd.Series:
    """Returns a Series (index=entity_id) of autoencoder_score in [0,1],
    where higher = larger reconstruction error = more anomalous."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the autoencoder detector. "
            "Install it with: pip install torch "
            "(CPU-only wheel is fine - see README)."
        )

    torch.manual_seed(random_state)

    X = StandardScaler().fit_transform(feat_df.values).astype(np.float32)
    X_tensor = torch.from_numpy(X)

    model = _Autoencoder(n_features=X.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(X_tensor)
        loss = loss_fn(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        per_sample_error = torch.mean((reconstructed - X_tensor) ** 2, dim=1).numpy()

    normalized = (per_sample_error - per_sample_error.min()) / (
        per_sample_error.max() - per_sample_error.min() + 1e-9
    )
    return pd.Series(normalized, index=feat_df.index, name="autoencoder_score")
