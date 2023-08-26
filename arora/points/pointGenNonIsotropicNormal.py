from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np

from .pointGen import PointGen


class PointGenNonIsotropicNormal(PointGen):
    def __init__(
        self, x_range: int, y_range: int, x_stdev: float, y_stdev: float, seed: int
    ) -> None:
        super().__init__(seed)
        self.x_range: int = x_range
        self.y_range: int = y_range
        self.x_mean: int = np.random.randint(low=0, high=x_range)
        self.y_mean: int = np.random.randint(low=0, high=y_range)
        self.x_stdev: float = x_stdev
        self.y_stdev: float = y_stdev
        self.stdev_prod: float = 0.5 * self.x_stdev * self.y_stdev
        self.mean = np.array([self.x_mean, self.y_mean])
        self.cov = np.array(
            [[self.x_stdev**2, self.stdev_prod], [self.stdev_prod, self.y_stdev**2]]
        )

    def _get_points(self, num_points: int, seed: int) -> List[Tuple[int, int]]:
        np.random.seed(seed)

        # non-isotropic normal points
        points: np.ndarray = np.random.multivariate_normal(
            mean=self.mean,
            cov=self.cov,
            size=num_points,
        )
        assert points.shape == (len(points), 2), f"Error points.shape: {points.shape}"

        # mod exceeding points
        point_list: List[Tuple[int, int]] = [
            (int(x) % self.x_range, int(y) % self.y_range) for (x, y) in points
        ]

        # get rid of overlapped points
        ret_set: Set[Tuple[int, int]] = set(point_list)

        # complement overlapped points
        while len(ret_set) < num_points:
            point: np.ndarray = np.random.multivariate_normal(
                mean=self.mean,
                cov=self.cov,
            )
            query: Tuple[int, int] = (
                int(point[0]) % self.x_range,
                int(point[1]) % self.y_range,
            )
            if query not in ret_set:
                ret_set.add(query)

        ret = list(ret_set)

        for ret_point in ret:
            (x, y) = ret_point
            assert 0 <= x < self.x_range and 0 <= y < self.y_range, (x, y)

        return ret
