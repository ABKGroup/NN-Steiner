from __future__ import annotations

import logging
import os
import time
from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from arora.data import Transform, get_transform
from arora.models import NNArora
from arora.points import read_points_dir
from arora.quadtree import QuadTree, SteinerTree
from arora.solver import NNSteiner
from arora.utils import get_dtype

from .utils import get_model, get_quadtree, show_exp_result


def nn_exp(args: DictConfig) -> None:
    logger: logging.Logger = logging.getLogger("nn_exp")

    # get points
    point_dir: str = os.path.join(
        hydra.utils.get_original_cwd(), args["nn_exp"]["test_set"]
    )
    points_set: List[List[Tuple[int, int]]] = read_points_dir(point_dir)

    # get model
    nn_arora: NNArora = get_model(args, "nn_exp")

    # get solver
    transform: Transform = get_transform(
        args["model"]["type"], get_dtype(args["model"]["precision"])
    )
    solver: NNSteiner = NNSteiner(nn_arora, transform, args["model"]["device"])

    # solving
    stts: List[SteinerTree] = []
    start_time: float = time.time()
    for point_set in tqdm(points_set):
        qt: QuadTree = get_quadtree(point_set, args["quadtree"])
        (_, _, final_stt) = solver.solve(qt, args["nn_exp"]["k"], args["nn_exp"]["fst"])
        stts.append(final_stt)
    # qt: QuadTree = get_quadtree(points_set[1], args["quadtree"])
    # (_, _, final_stt) = solver.solve(qt,args["nn_exp"]["k"], args["nn_exp"]["fst"])
    # stts.append(final_stt)
    end_time: float = time.time()
    runtime: float = end_time - start_time

    final_lens: List[float] = []
    final_lens_norm: List[float] = []

    for stt in stts:
        # costs
        stt.check()
        final_cost: float = stt.length()
        final_cost_norm: float = stt.normalized_length()

        final_lens.append(final_cost)
        final_lens_norm.append(final_cost_norm)

    exp_name: str = args["nn_exp"]["output"]
    show_exp_result(final_lens, final_lens_norm, f"{exp_name}-solved", logger, runtime)
