from typing import Dict, List

import torch
from torch import Tensor


class QuadTreeDataset:
    def __init__(self, files: List[str]) -> None:
        # parameters
        self.files: List[str] = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        file: str = self.files[idx]
        with open(file, "rb") as f:
            ret: Dict[str, Tensor] = torch.load(f)
        return ret
