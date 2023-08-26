from __future__ import annotations

import torch

from .transform import Transform
from .transformMLP import TransformMLP


def get_transform(type: str, dtype: torch.dtype) -> Transform:
    if type == "mlp":
        return TransformMLP(dtype)
    else:
        assert False, f"invalid model type: {type}"
