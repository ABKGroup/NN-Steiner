from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from arora.fix import fix


@dataclass
class SimpleCell:
    x_min: fix
    y_min: fix
    x_max: fix
    y_max: fix

    def divide(self) -> Tuple[SimpleCell, SimpleCell, SimpleCell, SimpleCell]:
        x_cent: fix = (self.x_min + self.x_max) // 2
        y_cent: fix = (self.y_min + self.y_max) // 2
        first: SimpleCell = SimpleCell(x_cent, y_cent, self.x_max, self.y_max)
        second: SimpleCell = SimpleCell(self.x_min, y_cent, x_cent, self.y_max)
        third: SimpleCell = SimpleCell(self.x_min, self.y_min, x_cent, y_cent)
        fourth: SimpleCell = SimpleCell(x_cent, self.y_min, self.x_max, y_cent)
        return (first, second, third, fourth)

    def contains(self, point: Tuple[int, int]) -> bool:
        (x, y) = point
        return (self.x_min <= fix(x) < self.x_max) and (
            self.y_min <= fix(y) < self.y_max
        )


class SimpleQuadTree:
    def __init__(self) -> None:
        self.cells: List[SimpleCell] = []
        self.is_leaf: List[bool] = []

    def init_root(self, x_range: int, y_range: int) -> None:
        self.cells.append(SimpleCell(fix(0), fix(0), fix(x_range), fix(y_range)))
        self.is_leaf.append(True)

    def assert_valid(self) -> None:
        assert len(self.cells) == len(self.is_leaf), [
            len(self.cells),
            len(self.is_leaf),
        ]

    def get_leaf_cells_idx(self) -> List[int]:
        self.assert_valid()
        return [i for i in range(len(self.cells)) if self.is_leaf[i]]

    def get_leaf_cells(self) -> List[SimpleCell]:
        self.assert_valid()
        return [self.cells[i] for i in range(len(self.cells)) if self.is_leaf[i]]

    def divide_cell(self, cell_idx: int) -> None:
        self.assert_valid()
        assert self.is_leaf[cell_idx]
        self.is_leaf[cell_idx] = False
        cells: List[SimpleCell] = list(self.cells[cell_idx].divide())
        self.cells.extend(cells)
        self.is_leaf.extend([True for _ in range(4)])

    @staticmethod
    def get_level_quadtree(x_range: int, y_range: int, level: int) -> SimpleQuadTree:
        qt: SimpleQuadTree = SimpleQuadTree()
        qt.init_root(x_range, y_range)
        for _ in range(level):
            leaf_cells_idx: List[int] = qt.get_leaf_cells_idx()
            for leaf_cell_idx in leaf_cells_idx:
                qt.divide_cell(leaf_cell_idx)
        return qt

    def is_isomorphic(self, other: SimpleQuadTree) -> bool:
        self.assert_valid()
        other.assert_valid()
        if len(self.cells) != len(other.cells):
            return False
        if len(self.is_leaf) != len(other.is_leaf):
            return False

        cell_dict: Dict[Tuple[int, int, int, int], bool] = {}
        self_leaf_cells: List[SimpleCell] = self.get_leaf_cells()
        other_leaf_cells: List[SimpleCell] = other.get_leaf_cells()
        if len(self_leaf_cells) != len(other_leaf_cells):
            return False
        for self_leaf_cell in self_leaf_cells:
            query: Tuple[int, int, int, int] = (
                self_leaf_cell.x_min.num,
                self_leaf_cell.y_min.num,
                self_leaf_cell.x_max.num,
                self_leaf_cell.y_max.num,
            )
            cell_dict[query] = False
        for other_leaf_cell in other_leaf_cells:
            query: Tuple[int, int, int, int] = (
                other_leaf_cell.x_min.num,
                other_leaf_cell.y_min.num,
                other_leaf_cell.x_max.num,
                other_leaf_cell.y_max.num,
            )
            if query not in cell_dict:
                return False
            if cell_dict[query]:
                return False
            cell_dict[query] = True
        return all(cell_dict.values())


def get_struct_quadtree(
    points: List[Tuple[int, int]], qt: SimpleQuadTree, kb: int
) -> List[Tuple[int, int]]:
    leaf_cells: List[SimpleCell] = qt.get_leaf_cells()

    # put points into leaf_cells
    leaf2points: List[List[Tuple[int, int]]] = [[] for _ in range(len(leaf_cells))]
    for leaf, leaf_points in zip(leaf_cells, leaf2points):
        # find all the points in the leaf cell
        new_points: List[Tuple[int, int]] = []
        for point in points:
            if leaf.contains(point):
                if point not in leaf_points:
                    leaf_points.append(point)
            else:
                new_points.append(point)
        points = new_points

        # take away points redundant points from the leaf cell
        while len(leaf_points) > kb:
            remove_id = np.random.choice(len(leaf_points))
            leaf_points.pop(remove_id)

    assert len(points) == 0, len(points)

    ret: List[Tuple[int, int]] = [
        point for leaf_points in leaf2points for point in leaf_points
    ]

    return ret


# def get_points_iso_quadtree(
#     x_range: int, y_range: int, level: int, kb: int, seed: int
# ) -> List[Tuple[int, int]]:
#     assert level >= 0, level
#     np.random.seed(seed)

#     level2cells = get_cells_from_level(x_range, y_range, level)

#     num_points: List[int] = []
#     choice: np.ndarray = np.arange(kb + 1)
#     # determine the number of terminals in each cell
#     for _ in range(len(level2cells[-2])):
#         numbers: List[int] = []
#         for _ in range(4):
#             num: int = int(np.random.choice(choice))
#             numbers.append(num)

#         # The number of terminals should be more than kb at the upper level
#         diff: int = (kb + 1) - sum(numbers)
#         for _ in range(diff):
#             # pick a number to add 1 terminal
#             pick: int = int(np.random.choice(4))
#             numbers[pick] += 1
#         assert sum(numbers) > kb, numbers
#         num_points.extend(numbers)

#     assert len(num_points) == len(level2cells[-1]), [
#         len(num_points),
#         len(level2cells[-1]),
#     ]
#     points: List[Tuple[int, int]] = []
#     for num, leaf_cell in zip(num_points, level2cells[-1]):
#         low_x: int = int(np.ceil(leaf_cell.x_min.to_float()))
#         low_y: int = int(np.ceil(leaf_cell.y_min.to_float()))
#         high_x: int = int(np.floor(leaf_cell.x_max.to_float()))
#         high_y: int = int(np.floor(leaf_cell.y_max.to_float()))
#         for _ in range(num):
#             while True:
#                 point_arr: np.ndarray = np.random.randint(
#                     low=[low_x, low_y], high=[high_x, high_y], size=(2,)
#                 )
#                 point: Tuple[int, int] = (int(point_arr[0]), int(point_arr[1]))
#                 if point not in points:
#                     points.append(point)
#                     break

#     return points
