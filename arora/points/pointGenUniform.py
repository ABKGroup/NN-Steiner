from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np

from .pointGen import PointGen


class PointGenUniform(PointGen):
    def __init__(self, x_range: int, y_range: int, seed: int) -> None:
        super().__init__(seed)
        self.x_range: int = x_range
        self.y_range: int = y_range

    def _get_points(self, num_points: int, seed: int) -> List[Tuple[int, int]]:
        np.random.seed(seed)

        # uniform points
        points: np.ndarray = np.random.randint(
            low=[0, 0], high=[self.x_range, self.y_range], size=(num_points, 2)
        )
        assert points.shape == (len(points), 2), f"Error points.shape: {points.shape}"

        # get rid of overlapped points
        point_list: List[Tuple[int, int]] = [tuple(point) for point in points]
        ret_set: Set[Tuple[int, int]] = set(point_list)

        # complement overlapped points
        while len(ret_set) < num_points:
            point: np.ndarray = np.random.randint(
                low=[0, 0], high=[self.x_range, self.y_range], size=(2,)
            )
            query: Tuple[int, int] = (point[0], point[1])
            if query not in ret_set:
                ret_set.add(query)

        ret = list(ret_set)

        return ret
