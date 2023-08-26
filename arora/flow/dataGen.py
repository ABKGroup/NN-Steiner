from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import os
import pickle
import statistics
from typing import Dict, List, Set, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

from arora.data import Transform, get_transform
from arora.points import PointGen, get_level_quadtree, get_pointGen, write_points
from arora.quadtree import QuadTree, QuadTreeData
from arora.utils import get_dtype

from .utils import get_quadtree, get_quadtreedata


def make_name(data_args: Dict) -> str:
    num_points: int = data_args["num_points"]
    x_range: int = data_args["x_range"]
    y_range: int = data_args["y_range"]
    level: int = data_args["level"]
    dist: str = data_args["dist"]

    if level == -1:
        return f"point{num_points}_{x_range}x{y_range}-{dist}"
    else:
        return f"level{level}_{x_range}x{y_range}-{dist}"


def dump_file(
    points: List[Tuple[int, int]],
    data_args: Dict,
    quadtree_args: Dict,
    file_name: str,
) -> None:
    output: str = data_args["output"]
    if output == "pickle":
        datum: QuadTreeData = get_quadtreedata(points, quadtree_args, data_args["fst"])

        # save datum
        pkl_name: str = f"{file_name}.pkl"
        with open(pkl_name, "wb") as f:
            pickle.dump(datum, f)
    elif output == "tensor":
        datum: QuadTreeData = get_quadtreedata(points, quadtree_args, data_args["fst"])
        tranform: Transform = get_transform(
            data_args["type"], get_dtype(data_args["precision"])
        )

        # transform
        tens: Dict[str, Tensor] = tranform.transform(datum)

        # save tensor
        pt_name: str = f"{file_name}.pt"
        with open(pt_name, "wb") as f:
            torch.save(tens, f)

    elif output == "pt":
        pt_name: str = f"{file_name}.txt"
        write_points(points, pt_name)
    else:
        assert False, f"invalid output type: {output}"


def check_no_overlap(points: List[Tuple[int, int]]) -> None:
    point_set: Set[Tuple[int, int]] = set()
    for point in points:
        point_set.add(point)
    assert len(point_set) == len(points)


def worker(idx: int, data_args: Dict, quadtree_args: Dict, output_dir: str) -> QuadTree:
    # point generation
    point_gen: PointGen = get_pointGen(
        data_args["dist"], data_args, idx + data_args["seed"]
    )
    points: List[Tuple[int, int]] = point_gen.get_points(data_args["num_points"])

    # make level quadtree
    if data_args["level"] != -1:
        points: List[Tuple[int, int]] = get_level_quadtree(
            points,
            data_args["x_range"],
            data_args["y_range"],
            data_args["level"],
            quadtree_args["kb"],
        )
        # check quadtree shape

    qt: QuadTree = get_quadtree(points, quadtree_args)
    if data_args["level"] != -1:
        qt.check_level(quadtree_args["level"])

    check_no_overlap(points)
    name: str = make_name(data_args)
    file_name: str = os.path.join(output_dir, f"{name}-{idx}")
    dump_file(points, data_args, quadtree_args, file_name)
    return qt


def get_dir_type(data_type: str) -> str:
    if data_type == "tensor":
        return "data"
    if data_type == "pickle":
        return "data"
    if data_type == "pt":
        return "points"
    else:
        assert False, f"invalid data type: {data_type}"


def output_files(
    data_args: Dict, quadtree_args: Dict, dir_name: str, logger: logging.Logger
) -> None:
    worker_list: List[Tuple[int, Dict, Dict, str]] = [
        (i, copy.deepcopy(data_args), copy.deepcopy(quadtree_args), dir_name)
        for i in range(data_args["batch"])
    ]

    with mp.Pool() as pool:
        qts: List[QuadTree] = pool.starmap(worker, tqdm(worker_list))

    # debug
    # qts: List[QuadTree] = []
    # for (idx, data_args, quadtree_args, output_dir) in tqdm(worker_list):
    #     qts.append(worker(idx, data_args, quadtree_args, output_dir))

    if data_args["level"] != -1:
        logging.info("checking...")
        for qt in tqdm(qts):
            assert QuadTree.isomorphic(qt, qts[0])

    points_num: List[int] = [len(qt.terminals) for qt in qts]
    mean: float = statistics.mean(points_num)
    stdev: float = statistics.stdev(points_num)

    logger.info(f"points summary: mean: {mean}, stdev: {stdev}")


def data_gen(args: DictConfig) -> None:
    logger = logging.getLogger("data_gen")

    # get parameters
    data_args: Dict = args["data_gen"]
    quadtree_args: Dict = args["quadtree"]

    # get output name
    dir_name: str = make_name(data_args)

    # make dir
    output: str = data_args["output"]
    batch: str = data_args["batch"]
    dir_type: str = get_dir_type(output)
    output_dir: str = os.path.join(
        hydra.utils.get_original_cwd(), f"{dir_type}/{dir_name}-{batch}-{output}"
    )
    # make directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger.info(f"points saved to {output_dir}")

    output_files(data_args, quadtree_args, output_dir, logger)
