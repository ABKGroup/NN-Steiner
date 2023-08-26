from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from torch import Tensor
from tqdm import tqdm

from .dataSet import QuadTreeDataset


class DataLoader:
    def __init__(
        self,
        dataset: QuadTreeDataset,
        collate_fn: Callable[[List[Dict[str, Tensor]]], Dict[str, Tensor]],
        batch_size: int,
    ) -> None:
        self.dataset: QuadTreeDataset = dataset
        self.collate_fn: Callable[
            [List[Dict[str, Tensor]]], Dict[str, Tensor]
        ] = collate_fn
        self.batch_size: int = batch_size

    def load_data(self, rank: int, world_size: int) -> List[Dict[str, Tensor]]:
        assert rank < world_size, [rank, world_size]

        ret: List[Dict[str, Tensor]] = []

        (start_batch, end_batch) = self._get_batch_interval(rank, world_size)
        for batch_idx in tqdm(range(start_batch, end_batch)):
            ret.append(self.load_batch(batch_idx))
        return ret

    def load_batch(self, batch_idx: int) -> Dict[str, Tensor]:
        start_idx: int = batch_idx * self.batch_size
        end_idx: int = min(start_idx + self.batch_size, len(self.dataset))
        batch_list: List[Dict[str, Tensor]] = [
            self.dataset[i] for i in range(start_idx, end_idx)
        ]
        batch: Dict[str, Tensor] = self.collate_fn(batch_list)
        return batch

    def _get_batch_interval(self, rank: int, world_size: int) -> Tuple[int, int]:
        total_batches: int = int(np.ceil(len(self.dataset) / self.batch_size))
        min_batches: int = int(np.floor(total_batches / world_size))
        remainder_batches: int = total_batches % world_size

        start: int = min_batches * rank + min(rank, remainder_batches)
        end: int = start + min_batches + (remainder_batches > rank)

        return (start, end)
