from __future__ import annotations

import copy
import logging
import os
import time
from typing import List

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from arora.fix import fix
from arora.quadtree import QuadTree, Rect, SteinerTree
from arora.solver import NNSteiner
from arora.utils import get_files

from .utils import show_exp_result


def extract_num(file: str) -> int:
    assert len(file.split(".")) == 2
    return int(file.split(".")[0])


def refine(args: DictConfig) -> None:
    logger: logging.Logger = logging.getLogger("refine")

    # get files
    point_dir: str = os.path.join(
        hydra.utils.get_original_cwd(), args["refine"]["point_dir"]
    )
    tree_dir: str = os.path.join(
        hydra.utils.get_original_cwd(), args["refine"]["tree_dir"]
    )

    point_files: List[str] = get_files(point_dir)

    tree_files: List[str] = os.listdir(tree_dir)
    tree_files = sorted(tree_files, key=extract_num)
    tree_files = [os.path.join(tree_dir, file) for file in tree_files]

    assert len(point_files) == len(tree_files), [len(point_files), len(tree_files)]

    # get stts
    stts: List[SteinerTree] = []
    for point_file, tree_file in zip(point_files, tree_files):
        stt: SteinerTree = SteinerTree.read(point_file, tree_file)
        stt.check_no_cycle()
        stts.append(stt)

    # get quadtrees
    bbox: Rect = Rect(
        fix(0),
        fix(0),
        fix(args["quadtree"]["bbox"]["width"]),
        fix(args["quadtree"]["bbox"]["height"]),
    )
    qts: List[QuadTree] = [
        QuadTree(stt.terminals, copy.deepcopy(bbox), args["quadtree"]) for stt in stts
    ]

    # refine
    start_time: float = time.time()
    for stt, qt in tqdm(zip(stts, qts)):
        NNSteiner._refine_leaf_cells(stt, [], {}, qt, args["refine"]["fst"])
        stt.post_process(args["refine"]["k"], args["refine"]["fst"])
    end_time: float = time.time()
    runtime: float = end_time - start_time

    final_lens: List[float] = []
    final_lens_norm: List[float] = []

    for stt in stts:
        # costs
        final_cost: float = stt.length()
        final_lens.append(final_cost)

        final_cost_norm: float = stt.normalized_length()
        final_lens_norm.append(final_cost_norm)

    exp_name: str = args["refine"]["output"]
    show_exp_result(final_lens, final_lens_norm, f"{exp_name}-refined", logger, runtime)
