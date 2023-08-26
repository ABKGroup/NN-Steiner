from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from arora.quadtree import Cell, Point, QuadTree, QuadTreeData, Rect, Terminal

from .transform import Transform


class TransformMLP(Transform):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__(dtype)

    def get_bound_tens(self, qt: QuadTree) -> Tensor:
        ret_list: List[List[int]] = []
        cell2bound: List[List[int]] = self._cell2bound(qt)

        for cell_bound in cell2bound:
            bound: List[int] = []
            for portal_id in cell_bound:
                if portal_id == -1:
                    bound.append(0)
                else:
                    assert portal_id >= 0, portal_id
                    bound.append(1)
            ret_list.append(bound)
        return torch.tensor(ret_list, dtype=self.dtype)

    def get_terminal_tens(self, qt: QuadTree) -> Tensor:
        ret_list: List[Tensor] = []
        for cell in qt.cells:
            # inner cell
            if len(cell.children) > 0:
                terminal_tens: Tensor = torch.full(size=(qt.kb * 2,), fill_value=-1)
            else:
                terminal_vec = self._terminals2vec_mlp(cell, qt)
                terminal_tens = torch.tensor(terminal_vec)
            ret_list.append(terminal_tens)
        return torch.stack(ret_list).to(self.dtype)

    def get_null_portal_tens(self) -> Tensor:
        return torch.zeros(size=(1,), dtype=self.dtype)

    def get_golden(self, qt: QuadTreeData) -> Tensor:
        return torch.tensor([1 if p else 0 for p in qt.portal_used], dtype=self.dtype)

    def _terminals2vec_mlp(self, cell: Cell, qt: QuadTree) -> np.ndarray:
        """
        terminals: [points]
            point: [x, y] (normalized, could be null)
            shape: (kb * 2,)
        """
        terminals_id: List[int] = cell.terminals_id
        ret_list: List[float] = []
        for terminal_id in terminals_id:
            terminal: Terminal = qt.terminals[terminal_id]
            (x, y) = self._point_norm2cell(terminal, cell)
            ret_list.extend([x, y])
        for _ in range(len(terminals_id), qt.args["kb"]):
            # ret_list.append([0, 0])
            ret_list.extend([-1, -1])

        ret: np.ndarray = np.array(ret_list)
        assert ret.shape == (qt.args["kb"] * 2,), [
            ret.shape,
            qt.args["kb"] * 2,
        ]
        return ret

    def _point_norm2cell(self, point: Point, cell: Cell) -> Tuple[float, float]:
        """
        normalize every value with x as base
        """
        bbox: Rect = cell.bbox
        x_: float = (point.x - bbox.x_min).to_float() / bbox.width.to_float()
        y_: float = (point.y - bbox.y_min).to_float() / bbox.width.to_float()
        return (x_, y_)
