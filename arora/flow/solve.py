from __future__ import annotations

import logging
import os
from typing import List, Tuple

import hydra
from omegaconf import DictConfig

from arora.data import Transform, get_transform
from arora.models import NNArora
from arora.points import read_points
from arora.quadtree import QuadTree
from arora.solver import NNSteiner
from arora.utils import get_dtype

from .utils import get_model, get_quadtree, show_stt


def solve(args: DictConfig) -> None:
    logger = logging.getLogger("solve")

    # mkdir
    output_dir: str = "solve"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger.info("load testcase")

    # get points
    point_file: str = os.path.join(
        hydra.utils.get_original_cwd(), args["solve"]["testcase"]
    )
    points: List[Tuple[int, int]] = read_points(point_file)
    qt: QuadTree = get_quadtree(points, args["quadtree"])

    # get model
    nn_arora: NNArora = get_model(args, "solve")

    # init solver and get steiner tree
    transform: Transform = get_transform(
        args["model"]["type"], get_dtype(args["model"]["precision"])
    )
    solver: NNSteiner = NNSteiner(nn_arora, transform, args["model"]["device"])
    (_, predict_stt, final_stt) = solver.solve(
        qt, args["solve"]["k"], args["solve"]["fst"]
    )

    # cost
    plot: bool = args["solve"]["plot"]
    show_stt("predict", qt, predict_stt, plot, output_dir, logger)
    show_stt("final", qt, final_stt, plot, output_dir, logger)
