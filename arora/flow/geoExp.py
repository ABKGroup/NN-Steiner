from __future__ import annotations

import logging
import os
import time
from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from arora.points import read_points_dir
from arora.quadtree import SteinerTree

from .utils import get_golden_stt, show_exp_result


def geo_exp(args: DictConfig) -> None:
    logger: logging.Logger = logging.getLogger("geo_exp")

    # get points
    point_dir: str = os.path.join(
        hydra.utils.get_original_cwd(), args["geo_exp"]["test_set"]
    )
    points_set: List[List[Tuple[int, int]]] = read_points_dir(point_dir)

    start_time: float = time.time()
    # get golden
    stts: List[SteinerTree] = [
        get_golden_stt(points, args["geo_exp"]["fst"]) for points in tqdm(points_set)
    ]
    end_time: float = time.time()
    runtime: float = end_time - start_time

    lens: List[float] = []
    lens_norm: List[float] = []

    for stt in tqdm(stts):
        # costs
        golden_cost: float = stt.length()
        golden_cost_norm: float = stt.normalized_length()

        lens.append(golden_cost)
        lens_norm.append(golden_cost_norm)

    exp_name: str = args["geo_exp"]["output"]
    show_exp_result(lens, lens_norm, f"{exp_name}-golden", logger, runtime)
