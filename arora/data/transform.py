from __future__ import annotations

import copy
from abc import ABC, abstractclassmethod
from typing import Dict, List

import torch
from torch import Tensor

from arora.quadtree import QuadTree, QuadTreeData, Side


class Transform(ABC):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()

        self.dtype: torch.dtype = dtype

    def transform(self, qt_data: QuadTreeData) -> Dict[str, Tensor]:
        ret: Dict[str, Tensor] = {
            "tree_struct": self.get_tree_struct(qt_data),
            "cell2bound": self.get_cell2bound(qt_data),
            "cell2cross": self.get_cell2cross(qt_data),
            "bound_tens": self.get_bound_tens(qt_data),
            "terminal_tens": self.get_terminal_tens(qt_data),
            "null_portal_tens": self.get_null_portal_tens(),
            "golden": self.get_golden(qt_data),
        }

        return ret

    def get_tree_struct(self, qt: QuadTree) -> Tensor:
        """
        ret: Tensor
            shape: (n_cell, 4)
        """
        ret_list: List[List[int]] = []
        for cell in qt.cells:
            if cell.is_leaf():
                ret_list.append([-1 for _ in range(4)])
            else:
                ret_list.append(copy.deepcopy(cell.children))
        return torch.tensor(ret_list, dtype=self.dtype)

    def get_cell2bound(self, qt: QuadTree) -> Tensor:
        """
        ret: Tensor
            shape: (n_cell, n_cell_p)
        """
        return torch.tensor(self._cell2bound(qt), dtype=self.dtype)

    def get_cell2cross(self, qt: QuadTree) -> Tensor:
        """
        ret: Tensor
            shape: (n_cell, n_cell_p)
        """
        return torch.tensor(self._cell2cross(qt), dtype=self.dtype)

    def _cell2bound(self, qt: QuadTree) -> List[List[int]]:
        ret: List[List[int]] = []
        for cell in qt.cells:
            east: Side = qt.sides[cell.boundary.east]
            south: Side = qt.sides[cell.boundary.south]
            west: Side = qt.sides[cell.boundary.west]
            north: Side = qt.sides[cell.boundary.north]

            cell_bound: List[int] = self._sides2portals(east, south, west, north)
            ret.append(cell_bound)
        return ret

    def _cell2cross(self, qt: QuadTree) -> List[List[int]]:
        ret: List[List[int]] = []
        for cell in qt.cells:
            if cell.cross is None:
                cell_cross: List[int] = [-1 for _ in range(qt.n_cell_p)]
            else:
                right: Side = qt.sides[cell.cross.right]
                bottom: Side = qt.sides[cell.cross.bottom]
                left: Side = qt.sides[cell.cross.left]
                top: Side = qt.sides[cell.cross.top]

                cell_cross: List[int] = self._sides2portals(right, bottom, left, top)
            ret.append(cell_cross)
        return ret

    def _sides2portals(
        self, east: Side, south: Side, west: Side, north: Side
    ) -> List[int]:
        ret: List[int] = []
        ret.extend(self._side2portals(east))
        ret.extend(self._side2portals(south))
        ret.extend(self._side2portals(west))
        ret.extend(self._side2portals(north))
        return ret

    def _side2portals(self, side: Side) -> List[int]:
        ret: List[int] = []
        for portal_id in side.portals_id:
            if portal_id is None:
                ret.append(-1)
            else:
                ret.append(portal_id)
        return ret

    @abstractclassmethod
    def get_bound_tens(self, qt: QuadTree) -> Tensor:
        ...

    @abstractclassmethod
    def get_terminal_tens(self, qt: QuadTree) -> Tensor:
        ...

    @abstractclassmethod
    def get_null_portal_tens(self) -> Tensor:
        ...

    @abstractclassmethod
    def get_golden(self, qt: QuadTreeData) -> Tensor:
        ...
