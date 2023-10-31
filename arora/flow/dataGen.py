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
from arora.points import (
    PointGen,
    SimpleCell,
    SimpleQuadTree,
    get_pointGen,
    get_struct_quadtree,
    write_points,
)
from arora.quadtree import QuadTree, QuadTreeData
from arora.utils import get_dtype

from .utils import (
    get_quadtree,
    get_quadtree_like,
    get_quadtreedata,
    get_quadtreedata_like,
)


def make_name(data_args: Dict) -> str:
    num_points: int = data_args["num_points"]
    x_range: int = data_args["x_range"]
    y_range: int = data_args["y_range"]
    level: int = data_args["level"]
    dist: str = data_args["dist"]
    constraint: str = data_args["constraint"]
    num_trees: int = data_args["num_trees"]

    if constraint == "batch":
        return f"batch{num_trees}_{x_range}x{y_range}-{dist}"
    elif constraint == "level":
        return f"level{level}_{x_range}x{y_range}-{dist}"
    else:
        return f"point{num_points}_{x_range}x{y_range}-{dist}"


def simplify_quadtree(qt: QuadTree) -> SimpleQuadTree:
    simple_qt: SimpleQuadTree = SimpleQuadTree()
    for cell in qt.cells:
        simple_qt.cells.append(
            SimpleCell(
                cell.bbox.x_min, cell.bbox.y_min, cell.bbox.x_max, cell.bbox.y_max
            )
        )
        simple_qt.is_leaf.append(cell.is_leaf())
    return simple_qt


def dump_file(
    points: List[Tuple[int, int]],
    data_args: Dict,
    quadtree_args: Dict,
    file_name: str,
    qt: QuadTree | None = None,
) -> Dict[str, Tensor] | None:
    output: str = data_args["output"]
    if output == "pickle":
        if qt is not None:
            datum: QuadTreeData = get_quadtreedata_like(points, qt, data_args["fst"])
            assert QuadTree.is_isomorphic(qt, datum)
        else:
            datum: QuadTreeData = get_quadtreedata(
                points, quadtree_args, data_args["fst"]
            )

        # save datum
        pkl_name: str = f"{file_name}.pkl"
        with open(pkl_name, "wb") as f:
            pickle.dump(datum, f)
    elif output == "tensor":
        if qt is not None:
            datum: QuadTreeData = get_quadtreedata_like(points, qt, data_args["fst"])
            assert QuadTree.is_isomorphic(qt, datum)
        else:
            datum: QuadTreeData = get_quadtreedata(
                points, quadtree_args, data_args["fst"]
            )
        tranform: Transform = get_transform(
            data_args["type"], get_dtype(data_args["precision"])
        )

        # transform
        tens: Dict[str, Tensor] = tranform.transform(datum)

        # save tensor
        pt_name: str = f"{file_name}.pt"
        with open(pt_name, "wb") as f:
            torch.save(tens, f)

        return tens

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


def worker(
    idx: int,
    data_args: Dict,
    quadtree_args: Dict,
    output_dir: str,
    tree: QuadTree | None,
) -> Tuple[int, Dict[str, Tensor] | None]:
    # point generation
    point_gen: PointGen = get_pointGen(
        data_args["dist"], data_args, idx + data_args["seed"]
    )
    points: List[Tuple[int, int]] = point_gen.get_points(data_args["num_points"])

    # make struct quadtree
    if tree is not None:
        points: List[Tuple[int, int]] = get_struct_quadtree(
            points, simplify_quadtree(tree), quadtree_args["kb"]
        )

    check_no_overlap(points)
    name: str = make_name(data_args)
    file_name: str = os.path.join(output_dir, f"{name}-{idx}")

    if tree is not None:
        dumped: Dict[str, Tensor] | None = dump_file(
            points, data_args, quadtree_args, file_name, tree
        )
    else:
        dumped = dump_file(points, data_args, quadtree_args, file_name)
    return (len(points), dumped)


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
    # build struct trees
    if data_args["constraint"] is None:
        trees: List[QuadTree | None] = [None]
    elif data_args["constraint"] == "level":
        simpl_qt: SimpleQuadTree = SimpleQuadTree.get_level_quadtree(
            data_args["x_range"], data_args["y_range"], data_args["level"]
        )
        point_gen: PointGen = get_pointGen(
            data_args["dist"], data_args, data_args["seed"]
        )
        points: List[Tuple[int, int]] = get_struct_quadtree(
            point_gen.get_points(data_args["num_points"]), simpl_qt, data_args["kb"]
        )
        qt: QuadTree = get_quadtree(points, quadtree_args)
        trees: List[QuadTree | None] = [qt]
    elif data_args["constraint"] == "batch":
        num_trees: int = data_args["num_trees"]
        assert num_trees > 0, f"invalid num_trees: {num_trees}"
        assert data_args["batch"] % num_trees == 0, f"invalid num_trees: {num_trees}"

        qts: List[QuadTree] = []
        qts_count: List[int] = []

        def is_isormorphic(qt: QuadTree) -> bool:
            for qt_idx, qt_ in enumerate(qts):
                if QuadTree.is_isomorphic(qt, qt_):
                    qts_count[qt_idx] += 1
                    return True
            return False

        while len(qts) < num_trees:
            point_gen: PointGen = get_pointGen(
                data_args["dist"], data_args, len(qts) + data_args["seed"]
            )
            points: List[Tuple[int, int]] = point_gen.get_points(
                data_args["num_points"]
            )
            qt: QuadTree = get_quadtree(points, quadtree_args)
            if not is_isormorphic(qt):
                qts.append(qt)
                qts_count.append(1)
        logger.info(f"tree_count: {qts_count}, levels: {[qt.max_level for qt in qts]}")

        trees: List[QuadTree | None] = [qt for qt in qts]
    else:
        assert False, f"invalid constraint: {data_args['constraint']}"

    batch_size: int = data_args["batch"] // len(trees)
    worker_list: List[Tuple[int, Dict, Dict, str, QuadTree | None]] = [
        (
            i,
            copy.deepcopy(data_args),
            copy.deepcopy(quadtree_args),
            dir_name,
            trees[i // batch_size],
        )
        for i in tqdm(range(data_args["batch"]))
    ]

    with mp.Pool() as pool:
        rets: List[Tuple[int, Dict[str, Tensor]|None]] = pool.starmap(worker, tqdm(worker_list))

    # debug
    # rets: List[Tuple[QuadTree, Dict[str, Tensor] | None]] = []
    # for idx, data_args, quadtree_args, output_dir, tree in tqdm(worker_list):
        # rets.append(worker(idx, data_args, quadtree_args, output_dir, tree))

    if data_args["constraint"] is not None:
        logging.info("checking...")
        for qt_idx, (_, dumped) in tqdm(enumerate(rets)):
            qt_ref, dumped_ref = rets[(qt_idx // batch_size) * batch_size]
            assert dumped is not None and dumped_ref is not None
            assert torch.all(dumped["tree_struct"] == dumped_ref["tree_struct"]), [
                dumped["tree_struct"],
                dumped_ref["tree_struct"],
            ]

    points_num: List[int] = [num_points for num_points, _ in rets]
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
