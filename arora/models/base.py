from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NNbase(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2
    ):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, base_vec: Tensor):
        base_vec = base_vec.flatten(start_dim=1)
        assert not torch.any(torch.isnan(base_vec)), base_vec
        ret: Tensor = self.stack(base_vec)
        assert not torch.any(torch.isnan(ret)), ret
        return ret
