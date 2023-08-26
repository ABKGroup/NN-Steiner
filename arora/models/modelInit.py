from __future__ import annotations

from torch import nn


def init_xavier(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
