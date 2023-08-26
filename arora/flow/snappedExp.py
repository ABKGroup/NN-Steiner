from __future__ import annotations

import logging
import os
from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from arora.points import read_points_dir
from arora.quadtree import QuadTreeData, SteinerTree
from arora.solver import NNSteiner

from .utils import get_quadtreedata, show_exp_result


def snapped_exp(args: DictConfig) -> None:
    logger: logging.Logger = logging.getLogger("snapped_exp")

    # get points
    point_dir: str = os.path.join(
        hydra.utils.get_original_cwd(), args["snapped_exp"]["test_set"]
    )
    points_set: List[List[Tuple[int, int]]] = read_points_dir(point_dir)

    # get golden
    data: List[QuadTreeData] = [
        get_quadtreedata(points, args["quadtree"], args["snapped_exp"]["fst"])
        for points in tqdm(points_set)
    ]

    snapped_lens: List[float] = []
    golden_lens: List[float] = []
    snapped_lens_norm: List[float] = []
    golden_lens_norm: List[float] = []

    for datum in tqdm(data):
        golden_stt: SteinerTree = datum.golden_stt
        (_, processed_stt) = NNSteiner.build_stt_data(
            datum, args["snapped_exp"]["k"], args["snapped_exp"]["fst"]
        )

        processed_stt.check()

        # costs
        snapped_cost: float = processed_stt.length()
        snapped_cost_norm: float = processed_stt.normalized_length()
        golden_cost: float = golden_stt.length()
        golden_cost_norm: float = golden_stt.normalized_length()

        snapped_lens.append(snapped_cost)
        snapped_lens_norm.append(snapped_cost_norm)
        golden_lens.append(golden_cost)
        golden_lens_norm.append(golden_cost_norm)

    exp_name: str = args["snapped_exp"]["output"]
    show_exp_result(snapped_lens, snapped_lens_norm, f"{exp_name}-snapped", logger)
    show_exp_result(golden_lens, golden_lens_norm, f"{exp_name}-golden", logger)
