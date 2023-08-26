from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig

from arora.fix import fix
from arora.points import read_points
from arora.quadtree import (
    QuadTree,
    QuadTreeData,
    Rect,
    Terminal,
    plot_data,
    plot_testcase,
)
from arora.solver import GeoSteiner, NNSteiner


def plot_quadtree(terminals: List[Terminal], bbox: Rect, output_dir: str, args: Dict):
    qt: QuadTree = QuadTree(terminals, bbox, args)
    plot_testcase(terminals, os.path.join(output_dir, "quadtree"), qt=qt)


def plot_golden(
    terminals: List[Terminal], bbox: Rect, output_dir: str, args: Dict, fst: int
) -> float:
    geo: GeoSteiner = GeoSteiner()
    stt = geo.solve(terminals, fst)
    cost: float = stt.length()
    title: str = f"cost: {cost}"
    data: QuadTree = QuadTreeData(terminals, bbox, args, stt)
    plot_data(data, os.path.join(output_dir, "golden"), title, quadtree=False)
    plot_data(data, os.path.join(output_dir, "golden-tree"), title)
    return cost


def plot_adapted(
    terminals: List[Terminal], bbox: Rect, output_dir: str, args: Dict, fst: int
) -> float:
    geo: GeoSteiner = GeoSteiner()
    stt = geo.solve(terminals, fst)
    data: QuadTree = QuadTreeData(terminals, bbox, args, stt)
    (adapted_stt, processed_stt) = NNSteiner.build_stt_data(data, 1, fst)
    cost: float = adapted_stt.length()
    title: str = f"cost: {cost}"
    plot_testcase(
        terminals, os.path.join(output_dir, "adapted"), title, data, adapted_stt, bbox
    )

    processed_cost: float = processed_stt.length()
    processed_title: str = f"cost: {processed_cost}"
    plot_testcase(
        terminals,
        os.path.join(output_dir, "processed"),
        processed_title,
        data,
        processed_stt,
        bbox,
    )
    return processed_cost


def plot(args: DictConfig):
    logger = logging.getLogger("plot")

    # get terminals
    file: str = os.path.join(hydra.utils.get_original_cwd(), args["plot"]["data"])
    points: List[Tuple[int, int]] = read_points(file)
    terminals: List[Terminal] = Terminal.terminals_from_point(points)

    filename: str = os.path.basename(file)
    output_dir: str = "plot_" + filename.split(".")[0]
    logger.info(f"plots saved to {output_dir}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    bbox_args: Dict = args["quadtree"]["bbox"]
    if bbox_args is None:
        bbox = Rect.from_terminals(terminals)
    else:
        bbox = Rect(fix(0), fix(0), fix(bbox_args["width"]), fix(bbox_args["height"]))

    if "terminals" in args["plot"]["output"]:
        logger.info("plotting terminals...")
        plot_testcase(terminals, os.path.join(output_dir, "terminals"))
        logger.debug("terminal plot saved")
    if "quadtree" in args["plot"]["output"]:
        logger.info("plotting quadtree...")
        plot_quadtree(terminals, bbox, output_dir, args["quadtree"])
        logger.debug("quadtree plot saved")
    if "golden" in args["plot"]["output"]:
        logger.info("plotting golden...")
        cost = plot_golden(
            terminals, bbox, output_dir, args["quadtree"], args["plot"]["fst"]
        )
        logger.info(f"cost: {cost}")
        logger.debug("golden plot saved")
    if "adapted" in args["plot"]["output"]:
        logger.info("plotting adapted...")
        cost = plot_adapted(
            terminals, bbox, output_dir, args["quadtree"], args["plot"]["fst"]
        )
        logger.info(f"cost: {cost}")
        logger.debug("adapted plot saved")
