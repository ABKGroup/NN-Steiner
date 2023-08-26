from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NNretrieve(nn.Module):
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
            nn.Sigmoid(),
        )

    def forward(self, retrieve_vec: Tensor):
        assert not torch.any(torch.isnan(retrieve_vec)), retrieve_vec
        ret: Tensor = self.stack(retrieve_vec)
        assert not torch.any(torch.isnan(ret)), ret
        return ret
