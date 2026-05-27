"""Step 4 v01: CNN architectures.

Four per-survey models. Same backbone, different input-channel counts.
Small and fast — appropriate for 17 known SNe + ~5K augmented positives.

Input is normalised internally:
  1. Replace NaN with 0
  2. Subtract per-channel sigma-clipped median (computed at train time and
     stored as a buffer — we estimate the per-channel mean+std on the
     training set and freeze it as a normaliser).

Output: P(detected SN-like point source at centre).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.norm_mean = nn.Parameter(torch.zeros(in_ch, 1, 1), requires_grad=False)
        self.norm_std  = nn.Parameter(torch.ones(in_ch, 1, 1),  requires_grad=False)
        # 64×64 → 32×32 → 16×16 → 8×8
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 64)
        self.fc2   = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def set_norm(self, mean, std):
        """Set per-channel normalisation buffers (call after computing
        train-set stats). mean and std are 1-D tensors of length in_ch."""
        with torch.no_grad():
            self.norm_mean.copy_(mean.view(self.in_ch, 1, 1))
            self.norm_std.copy_( std.view(self.in_ch, 1, 1).clamp(min=1e-6))

    def forward(self, x):
        # Replace NaN with 0
        x = torch.nan_to_num(x, nan=0.0)
        x = (x - self.norm_mean) / self.norm_std
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze(-1)   # logits


def make_models():
    """Return dict of survey-name → SmallCNN with appropriate in_ch."""
    return {
        "hst":  SmallCNN(in_ch=1),
        "jwst": SmallCNN(in_ch=4),
        "vis":  SmallCNN(in_ch=1),
        "nisp": SmallCNN(in_ch=3),
    }


def best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
