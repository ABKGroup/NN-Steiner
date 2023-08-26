from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from .pointGen import PointGen


class PointGenMixNormal(PointGen):
    def __init__(
        self, x_range: int, y_range: int, x_stdev: float, y_stdev: float, seed: int
    ) -> None:
        super().__init__(seed)

        self.x_range: int = x_range
        self.y_range: int = y_range
        self.x_stdev: float = x_stdev
        self.y_stdev: float = y_stdev
        self.stdev_prod: float = self.x_stdev * self.y_stdev

        np.random.seed(seed)

        self.num_clusters: int = 10
        self.centers: np.ndarray = np.random.uniform(
            low=[0, 0], high=[self.x_range, self.y_range], size=(self.num_clusters, 2)
        )
        self.rhos: np.ndarray = np.random.uniform(low=-1.0, high=1.0, size=10)
        self.covs: np.ndarray = np.array(
            [
                np.array(
                    [
                        [x_stdev**2, float(rho) * self.stdev_prod],
                        [float(rho) * self.stdev_prod, y_stdev**2],
                    ]
                )
                for rho in self.rhos
            ]
        )
        self.weights: np.ndarray = np.random.dirichlet(np.ones(self.num_clusters))

        self.generator: GaussianMixture = GaussianMixture(
            n_components=self.num_clusters,
            covariance_type="full",
            random_state=seed,
        )
        self.generator.fit(np.random.random(size=(10, 2)))  # dummy

        self.generator.weights_ = self.weights
        self.generator.means_ = self.centers
        self.generator.covariances_ = self.covs

    def _get_points(self, num_points: int, seed: int) -> List[Tuple[int, int]]:
        # non-isotropic normal points
        points, _ = self.generator.sample(num_points)
        assert points.shape == (len(points), 2), f"Error points.shape: {points.shape}"

        # mod exceeding points
        point_list: List[Tuple[int, int]] = [
            (int(x) % self.x_range, int(y) % self.y_range) for (x, y) in points
        ]

        # get rid of overlapped points
        ret_set: Set[Tuple[int, int]] = set(point_list)

        # complement overlapped points
        while len(ret_set) < num_points:
            point_arr, _ = self.generator.sample(1)
            point: np.ndarray = point_arr[0]
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
