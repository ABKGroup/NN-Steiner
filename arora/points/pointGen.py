from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class PointGen(ABC):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed: int = seed

    def get_points(self, num_points: int) -> List[Tuple[int, int]]:
        ret: List[Tuple[int, int]] = self._get_points(num_points, self.seed)

        self.seed += 1

        return ret

    @abstractmethod
    def _get_points(self, num_points: int, seed: int) -> List[Tuple[int, int]]:
        ...
