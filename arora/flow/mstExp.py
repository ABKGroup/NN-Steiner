from __future__ import annotations

import logging
import os
from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from arora.points import read_points_dir
from arora.quadtree import Terminal
from arora.solver import MST

from .utils import show_exp_result


def mst_exp(args: DictConfig) -> None:
    logger: logging.Logger = logging.getLogger("mst_exp")

    # get points
    point_dir: str = os.path.join(
        hydra.utils.get_original_cwd(), args["mst_exp"]["test_set"]
    )
    points_set: List[List[Tuple[int, int]]] = read_points_dir(point_dir)

    # get solver
    solver: MST = MST()

    final_lens: List[float] = []
    final_lens_norm: List[float] = []

    # solving
    logger.info("solving...")
    for point_set in tqdm(points_set):
        terminals: List[Terminal] = Terminal.terminals_from_point(point_set)
        rmst = solver.solve(terminals)

        rmst.check()

        # costs
        final_cost: float = rmst.length()
        final_cost_norm: float = rmst.normalized_length()

        final_lens.append(final_cost)
        final_lens_norm.append(final_cost_norm)

    exp_name: str = args["mst_exp"]["output"]
    show_exp_result(final_lens, final_lens_norm, f"{exp_name}-mst", logger)
