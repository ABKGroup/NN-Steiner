from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from arora.fix import fix


@dataclass
class Cell:
    x_min: fix
    y_min: fix
    x_max: fix
    y_max: fix

    def divide(self) -> Tuple[Cell, Cell, Cell, Cell]:
        x_cent: fix = (self.x_min + self.x_max) // 2
        y_cent: fix = (self.y_min + self.y_max) // 2
        first: Cell = Cell(x_cent, y_cent, self.x_max, self.y_max)
        second: Cell = Cell(self.x_min, y_cent, x_cent, self.y_max)
        third: Cell = Cell(self.x_min, self.y_min, x_cent, y_cent)
        fourth: Cell = Cell(x_cent, self.y_min, self.x_max, y_cent)
        return (first, second, third, fourth)

    def contains(self, point: Tuple[int, int]) -> bool:
        (x, y) = point
        return (self.x_min <= fix(x) < self.x_max) and (
            self.y_min <= fix(y) < self.y_max
        )


def get_cells_from_level(x_range: int, y_range: int, level: int) -> List[List[Cell]]:
    root_cell: Cell = Cell(fix(0), fix(0), fix(x_range), fix(y_range))
    level2cells: List[List[Cell]] = [[root_cell]]
    for i in range(level):
        cells_level_i: List[Cell] = level2cells[i]
        cells_level_ip1: List[Cell] = []
        for cell in cells_level_i:
            cells_level_ip1.extend(list(cell.divide()))
        level2cells.append(cells_level_ip1)

    for i, cell_level in enumerate(level2cells):
        assert len(cell_level) == 4**i, len(cell_level)
    return level2cells


def get_level_quadtree(
    points: List[Tuple[int, int]], x_range: int, y_range: int, level: int, kb: int
) -> List[Tuple[int, int]]:
    cells: List[List[Cell]] = get_cells_from_level(x_range, y_range, level)
    leaf_cells: List[Cell] = cells[-1]

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
