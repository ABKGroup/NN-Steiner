import logging
import os
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from arora.fix import fix
from arora.models import NNArora
from arora.quadtree import (
    QuadTree,
    QuadTreeData,
    Rect,
    SteinerTree,
    Terminal,
    plot_testcase,
)
from arora.solver import GeoSteiner


def get_quadtree(points: List[Tuple[int, int]], quadtree_args: Dict) -> QuadTree:
    # get terminals
    terminals: List[Terminal] = Terminal.terminals_from_point(points)

    # get bbox
    bbox: Rect = Rect(
        fix(0),
        fix(0),
        fix(quadtree_args["bbox"]["width"]),
        fix(quadtree_args["bbox"]["height"]),
    )

    # solve
    qt: QuadTree = QuadTree(terminals, bbox, quadtree_args)
    return qt

def get_quadtree_like(points: List[Tuple[int, int]], qt: QuadTree) -> QuadTree:
    # get terminals
    terminals: List[Terminal] = Terminal.terminals_from_point(points)

    return QuadTree.tree_like(qt, terminals)

def get_golden_stt(points: List[Tuple[int, int]], fst: int) -> SteinerTree:
    # get terminals
    terminals: List[Terminal] = Terminal.terminals_from_point(points)
    # solve
    geo: GeoSteiner = GeoSteiner()
    stt: SteinerTree = geo.solve(terminals, fst)

    return stt


def get_quadtreedata(
    points: List[Tuple[int, int]], quadtree_args: Dict, fst: int
) -> QuadTreeData:
    # get terminals
    terminals: List[Terminal] = Terminal.terminals_from_point(points)

    stt: SteinerTree = get_golden_stt(points, fst)

    # get bbox
    bbox: Rect = Rect(
        fix(0),
        fix(0),
        fix(quadtree_args["bbox"]["width"]),
        fix(quadtree_args["bbox"]["height"]),
    )
    datum: QuadTreeData = QuadTreeData(terminals, bbox, quadtree_args, stt)
    return datum


def show_stt(
    name: str,
    data: QuadTree,
    stt: SteinerTree,
    plot: bool,
    output_dir: str,
    logger: logging.Logger,
    base: float | None = None,
    check: bool = False,
) -> float:
    if check:
        stt.check()

    cost: float = stt.length()
    msg: str = f"{name}_cost: {cost}"
    if base is not None:
        msg = f"{msg}, ratio: {cost / base}"
    logger.info(msg)
    stt.dump(os.path.join(output_dir, f"{name}.txt"))
    if plot:
        plot_testcase(
            stt.terminals,
            os.path.join(output_dir, name),
            msg,
            data,
            stt,
            data.bbox,
        )
    return cost


def get_model(args: DictConfig, flow: str) -> NNArora:
    if args[flow]["model"] is None:
        model_path: str = "train/model/nnArora_best.pt"
    else:
        model_path: str = os.path.join(
            hydra.utils.get_original_cwd(), args[flow]["model"]
        )
    nn_arora: NNArora = NNArora(args["model"], args["quadtree"])
    state_dict: Dict[str, Tensor] = torch.load(model_path)
    state_dict = {
        key.replace("module.", ""): value for key, value in state_dict.items()
    }
    nn_arora.load_state_dict(state_dict)
    return nn_arora


def show_exp_result(
    lens: List[float],
    lens_norm: List[float],
    name: str,
    logger: logging.Logger,
    runtime: float | None = None,
) -> None:
    lens_arr: np.ndarray = np.array(lens)
    lens_norm_arr: np.ndarray = np.array(lens_norm)

    assert lens_arr.shape == lens_norm_arr.shape, [lens_arr.shape, lens_norm_arr.shape]

    avg: float = round(lens_arr.mean(), 6)
    avg_norm: float = round(lens_norm_arr.mean(), 6)

    save_name: str = os.path.join(hydra.utils.get_original_cwd(), name)
    with open(f"{save_name}.txt", "w") as f:
        for length in lens_arr:
            f.write(f"{str(length)}\n")
    with open(f"{save_name}-norm.txt", "w") as f_norm:
        for length_norm in lens_norm_arr:
            f_norm.write(f"{str(length_norm)}\n")

    if runtime is not None:
        logger.info(f"Runtime: {round(runtime, 3)} secs")
    logger.info(f"Average {name} stt wirelength: {avg}, norm: {avg_norm}")
