from __future__ import annotations

import copy
import logging
import os
from typing import List, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor

from arora.data import Transform, get_transform
from arora.models import NNArora
from arora.points import read_points
from arora.quadtree import QuadTreeData, SteinerTree
from arora.solver import NNSteiner
from arora.utils import get_dtype, get_f1, plot_conf_matrix

from .utils import get_model, get_quadtreedata, show_stt


def eval(args: DictConfig) -> None:
    logger = logging.getLogger("eval")

    # mkdir
    output_dir: str = "eval"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger.info("load testcase...")

    # get points
    point_file: str = os.path.join(
        hydra.utils.get_original_cwd(), args["eval"]["testcase"]
    )
    points: List[Tuple[int, int]] = read_points(point_file)

    logger.info(f"# points = {len(points)}")
    # getQuadTreedata
    data: QuadTreeData = get_quadtreedata(points, args["quadtree"], args["eval"]["fst"])

    # get golden and adapt and processed
    golden_stt: SteinerTree = data.golden_stt
    (adapted_stt, processed_stt) = NNSteiner.build_stt_data(
        data, args["eval"]["k"], args["eval"]["fst"]
    )

    logger.info("evaluation starts...")

    # get model
    nn_arora: NNArora = get_model(args, "eval")

    # init solver and get steiner tree
    transform: Transform = get_transform(
        args["model"]["type"], get_dtype(args["model"]["precision"])
    )
    solver: NNSteiner = NNSteiner(nn_arora, transform, args["model"]["device"])
    (predict_tens, predict_stt, final_stt) = solver.solve(
        data, args["eval"]["threshold"], args["eval"]["k"], args["eval"]["fst"]
    )

    # get acc
    golden_tens: Tensor = transform.get_golden(data).to(args["model"]["device"])
    acc: float = get_f1(predict_tens, golden_tens)
    logger.info(f"acc: {acc}")
    plot: bool = args["eval"]["plot"]
    if plot:
        plot_conf_matrix(golden_tens, predict_tens, "eval")

    # cost
    golden_cost: float = show_stt("golden", data, golden_stt, plot, output_dir, logger)
    show_stt("adapted", data, adapted_stt, plot, output_dir, logger, golden_cost)
    show_stt(
        "processed", data, processed_stt, plot, output_dir, logger, golden_cost, True
    )
    show_stt("predict", data, predict_stt, plot, output_dir, logger, golden_cost)
    show_stt("final", data, final_stt, plot, output_dir, logger, golden_cost, True)
