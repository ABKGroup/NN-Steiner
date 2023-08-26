from __future__ import annotations

import math

import torch
from torch import Tensor

from arora.utils import get_f1


def test_f1():
    predict: Tensor = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])
    golden: Tensor = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    f1: float = get_f1(predict, golden)
    assert math.isclose(2 / 6, f1)
