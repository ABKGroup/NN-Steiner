from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NNdp(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2
    ):
        super().__init__()
        self.stack = nn.Sequential(
            nn.BatchNorm1d(input_size),
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
        self.small = nn.Linear(input_size, output_size)

    def forward(self, dp_vec: Tensor):
        assert not torch.any(torch.isnan(dp_vec)), dp_vec
        ret: Tensor = self.stack(dp_vec)
        skip: Tensor = self.small(dp_vec)
        assert not torch.any(torch.isnan(ret)), ret
        return ret + skip
