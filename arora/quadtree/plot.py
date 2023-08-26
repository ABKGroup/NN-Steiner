from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from arora.fix import fix

from .quadTree import Cell, LPoint, QuadTree, Side
from .quadTreeData import QuadTreeData
from .shape import Point, PurePoint, Rect, Terminal
from .steinerTree import Edge, SteinerTree

POINT_SIZE: float = 10


def shift_bbox2origin(bbox: Rect) -> Rect:
    return Rect(fix(0), fix(0), (bbox.x_max - bbox.x_min), (bbox.y_max - bbox.y_min))


def shift_point(point: Point, bbox: Rect) -> Point:
    x: fix = point.x - bbox.x_min
    y: fix = point.y - bbox.y_min
    return PurePoint(point.id, x, y)


def plot_point(
    ax: plt.Axes, point: Point, color: str = "blue", shape: str = "o"
) -> None:
    ax.plot(
        point.x.to_float(),
        point.y.to_float(),
        shape,
        color=color,
        label=point.typ,
        ms=POINT_SIZE,
    )


def plot_bbox(ax: plt.Axes, bbox: Rect, color: str = "orange") -> None:
    rect = Rectangle(
        [bbox.x_min.to_float(), bbox.y_min.to_float()],
        (bbox.x_max - bbox.x_min).to_float(),
        (bbox.y_max - bbox.y_min).to_float(),
        edgecolor=color,
        fill=False,
    )
    ax.add_patch(rect)


def plot_boundary(ax: plt.Axes, bbox: Rect, extend: bool = False) -> None:
    ax.set_xlim(bbox.x_min.to_float() - extend, bbox.x_max.to_float() + extend)
    ax.set_ylim(bbox.y_min.to_float() - extend, bbox.y_max.to_float() + extend)


def plot_terminals(ax: plt.Axes, terminals: List[Terminal]) -> None:
    for terminal in terminals:
        plot_point(ax, terminal)


def plot_quadtree(ax: plt.Axes, qt: QuadTree) -> None:
    for portal in qt.portals:
        plot_point(ax, portal, "green")

    for cell in qt.cells:
        plot_bbox(ax, cell.bbox)


def plot_edge(
    ax: plt.Axes, points: Tuple[Point, Point], bbox: Rect | None = None
) -> None:
    (src, tgt) = points

    assert (
        src.x == tgt.x or src.y == tgt.y
    ), f"src: ({src.x}, {src.y}) does not match tgt: ({tgt.x},{tgt.y})"
    x = np.array([src.x.to_float(), tgt.x.to_float()])
    y = np.array([src.y.to_float(), tgt.y.to_float()])

    if bbox is not None:
        x = np.clip(x, a_min=bbox.x_min.to_float(), a_max=bbox.x_max.to_float())
        y = np.clip(y, a_min=bbox.y_min.to_float(), a_max=bbox.y_max.to_float())

    ax.plot(x, y, color="black", ms=POINT_SIZE)


def plot_steiner_tree(
    ax: plt.Axes,
    stt: SteinerTree,
    ledge2lpoint: Dict[int, int] | None = None,
    lpoints: List[LPoint] | None = None,
) -> None:
    assert (ledge2lpoint is None and lpoints is None) or (
        ledge2lpoint is not None and lpoints is not None
    )
    for terminal in stt.terminals:
        plot_point(ax, terminal, "blue")

    for stp in stt.stps:
        plot_point(ax, stp, "red")

    if lpoints is not None:
        for lpoint in lpoints:
            plot_point(ax, lpoint, "purple")

    for edge in stt.edges:
        # TODO: fix lshape problem
        if edge.typ == "straight":
            plot_edge(ax, (stt.get_point(edge.src), stt.get_point(edge.tgt)))
        elif edge.typ == "lshape":
            src: Point = stt.get_point(edge.src)
            tgt: Point = stt.get_point(edge.tgt)
            if ledge2lpoint is not None:
                assert lpoints is not None
                mid_id: int = ledge2lpoint[edge.id]
                mid_point: Point = lpoints[mid_id]
            else:
                mid_point: Point = PurePoint(-1, src.x, tgt.y)
            plot_edge(ax, (src, mid_point))
            plot_edge(ax, (mid_point, tgt))


def plot_testcase(
    terminals: List[Terminal],
    output: str,
    title: str | None = None,
    qt: QuadTree | None = None,
    stt: SteinerTree | None = None,
    bbox: Rect | None = None,
) -> None:
    plt.clf()
    plt.close()
    fig, ax = plt.subplots(1)

    if bbox is None:
        bbox = Rect.from_terminals(terminals)
    assert bbox is not None
    plot_boundary(ax, bbox, extend=True)

    plot_terminals(ax, terminals)

    if qt is not None:
        plot_quadtree(ax, qt)

    if stt is not None:
        plot_steiner_tree(ax, stt)

    fig.set_size_inches(10, 10)

    if title is not None:
        fig.suptitle(title)
    fig.savefig(f"{output}.png", dpi=1080)

    fig.clf()


def plot_data(
    data: QuadTreeData, output: str, title: str | None = None, quadtree: bool = True
) -> None:
    fig, ax = plt.subplots(1)

    plot_boundary(ax, data.bbox, extend=True)

    plot_terminals(ax, data.terminals)

    if quadtree:
        plot_quadtree(ax, data)

    plot_steiner_tree(ax, data.stt)

    fig.set_size_inches(10, 10)
    if title is not None:
        fig.suptitle(title)
    fig.savefig(f"{output}.png", dpi=1080)

    fig.clf()


def plot_side_portals(ax: plt.Axes, side: Side, qt: QuadTree):
    assert (
        len(side.portals_id) == qt.args["m"] + 2
    ), f"number of side portals should not be {len(side.portals_id)}"
    if side.typ is not None:
        for i, portal_id in enumerate(side.portals_id):
            if portal_id is None:
                assert (
                    i != 0 and i != qt.args["m"] + 1
                ), "the first and the last portals in a side should not be None"
                if side.typ == "vertical":
                    point: Point = PurePoint(
                        0, side.pivot, side.span[0] + side.interval * i
                    )
                elif side.typ == "horizontal":
                    point: Point = PurePoint(
                        0, side.span[0] + side.interval * i, side.pivot
                    )
                else:
                    assert False, f"Error side type: {side.typ}"
                plot_point(ax, point, "green", "x")
            else:
                point: Point = qt.portals[portal_id]
                plot_point(ax, point, "green")


def plot_cell(cell: Cell, qt: QuadTree, output_dir: str) -> None:
    plt.close()
    fig, ax = plt.subplots(1)

    plot_boundary(ax, cell.bbox, True)

    # plot the lines
    if len(cell.children):
        assert len(cell.children) == 4, cell.children
        for child_id in cell.children:
            child_cell: Cell = qt.cells[child_id]
            plot_bbox(ax, child_cell.bbox)
    else:
        plot_bbox(ax, cell.bbox)

    # plot the terminals
    for terminal_id in cell.terminals_id:
        terminal: Terminal = qt.terminals[terminal_id]
        plot_point(ax, terminal)

    # plot the portals
    plot_side_portals(ax, qt.sides[cell.boundary.east], qt)
    plot_side_portals(ax, qt.sides[cell.boundary.south], qt)
    plot_side_portals(ax, qt.sides[cell.boundary.west], qt)
    plot_side_portals(ax, qt.sides[cell.boundary.north], qt)
    if cell.cross is not None:
        plot_side_portals(ax, qt.sides[cell.cross.right], qt)
        plot_side_portals(ax, qt.sides[cell.cross.bottom], qt)
        plot_side_portals(ax, qt.sides[cell.cross.left], qt)
        plot_side_portals(ax, qt.sides[cell.cross.top], qt)

    fig.savefig(os.path.join(output_dir, f"cell-{cell.id}.png"))


def plot_cells(qt: QuadTree, output_dir: str) -> None:
    print("plotting cells...")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for cell in qt.cells:
        plot_cell(cell, qt, output_dir)
    print("cell plots saved")
