from __future__ import annotations

from typing import List, Tuple

from arora.fix import fix
from arora.points import read_points
from arora.quadtree import (
    Cell,
    Portal,
    QuadTree,
    QuadTreeData,
    Rect,
    Side,
    SteinerTree,
    Terminal,
)
from arora.solver import GeoSteiner, NNSteiner


def get_qt() -> QuadTree:
    points: List[Tuple[int, int]] = read_points(
        "./points/point100_10000x10000-uniform-100-pt/point100_10000x10000-uniform-0.txt"
    )
    terminals: List[Terminal] = Terminal.terminals_from_point(points)
    bbox = Rect(fix(0), fix(0), fix(10000), fix(10000))
    qt: QuadTree = QuadTree(terminals, bbox, {"m": 3, "kb": 4, "level": -1})
    return qt


def get_data(qt: QuadTree) -> QuadTreeData:
    solver = GeoSteiner()
    golden_stt: SteinerTree = solver.solve(qt.terminals, 1)
    return QuadTreeData(qt.terminals, qt.bbox, qt.args, golden_stt)


def test_portals() -> None:
    qt: QuadTree = get_qt()
    for side in qt.sides:
        assert len(side.portals_id) == qt.args["m"] + 2
        assert len(side.children) == 2 or len(side.children) == 0
        # canvas boundary
        if side.typ is None:
            for portal_id in side.portals_id:
                assert portal_id is None
        # check every portal on the right position
        elif side.typ == "vertical":
            for i, portal_id in enumerate(side.portals_id):
                if portal_id is not None:
                    portal: Portal = qt.portals[portal_id]
                    assert portal.x == side.pivot, [portal.x, side.pivot]
                    assert (
                        False
                        or portal.y == side.span[0] + side.interval * i
                        or portal.y == side.span[1]
                    )
        elif side.typ == "horizontal":
            for i, portal_id in enumerate(side.portals_id):
                if portal_id is not None:
                    portal: Portal = qt.portals[portal_id]
                    assert portal.y == side.pivot, [portal.y, side.pivot]
                    assert (
                        False
                        or portal.x == side.span[0] + side.interval * i
                        or portal.x == side.span[1]
                    )
        if side.parent_id is not None:
            continue


def assert_cell_bbox(cell: Cell, qt: QuadTree) -> None:
    bbox: Rect = cell.bbox

    east: Side = qt.sides[cell.boundary.east]
    south: Side = qt.sides[cell.boundary.south]
    west: Side = qt.sides[cell.boundary.west]
    north: Side = qt.sides[cell.boundary.north]

    assert bbox.x_min == west.pivot == south.span[0] == north.span[0]
    assert bbox.y_min == south.pivot == east.span[0] == west.span[0]
    assert bbox.x_max == east.pivot == south.span[1] == north.span[1]
    assert bbox.y_max == north.pivot == east.span[1] == west.span[1]

    if cell.cross is not None:
        right: Side = qt.sides[cell.cross.right]
        bottom: Side = qt.sides[cell.cross.bottom]
        left: Side = qt.sides[cell.cross.left]
        top: Side = qt.sides[cell.cross.top]

        assert bbox.x_min == left.span[0]
        assert bbox.y_min == bottom.span[0]
        assert bbox.x_max == right.span[1]
        assert bbox.y_max == top.span[1]
        assert bbox.x_cent == bottom.pivot == top.pivot == right.span[0] == left.span[1]
        assert bbox.y_cent == right.pivot == left.pivot == bottom.span[1] == top.span[0]


def assert_cell_children(cell: Cell, qt: QuadTree) -> None:
    first: Cell = qt.cells[cell.children[0]]
    second: Cell = qt.cells[cell.children[1]]
    third: Cell = qt.cells[cell.children[2]]
    fourth: Cell = qt.cells[cell.children[3]]

    assert cell.cross is not None
    # first
    assert first.boundary.east in qt.sides[cell.boundary.east].children
    assert qt.sides[first.boundary.east].parent_id == cell.boundary.east

    assert first.boundary.south == cell.cross.right

    assert first.boundary.west == cell.cross.top

    assert first.boundary.north in qt.sides[cell.boundary.north].children
    assert qt.sides[first.boundary.north].parent_id == cell.boundary.north

    # second
    assert second.boundary.east == cell.cross.top

    assert second.boundary.south == cell.cross.left

    assert second.boundary.west in qt.sides[cell.boundary.west].children
    assert qt.sides[second.boundary.west].parent_id == cell.boundary.west

    assert second.boundary.north in qt.sides[cell.boundary.north].children
    assert qt.sides[second.boundary.north].parent_id == cell.boundary.north

    # third
    assert third.boundary.east == cell.cross.bottom

    assert third.boundary.south in qt.sides[cell.boundary.south].children
    assert qt.sides[third.boundary.south].parent_id == cell.boundary.south

    assert third.boundary.west in qt.sides[cell.boundary.west].children
    assert qt.sides[third.boundary.west].parent_id == cell.boundary.west

    assert third.boundary.north == cell.cross.left

    # fourth
    assert fourth.boundary.east in qt.sides[cell.boundary.east].children
    assert qt.sides[fourth.boundary.east].parent_id == cell.boundary.east

    assert fourth.boundary.south in qt.sides[cell.boundary.south].children
    assert qt.sides[fourth.boundary.south].parent_id == cell.boundary.south

    assert fourth.boundary.west == cell.cross.bottom

    assert fourth.boundary.north == cell.cross.right


def test_cells() -> None:
    qt: QuadTree = get_qt()
    num_terminals: int = 0
    for cell in qt.cells:
        if len(cell.children) == 0:
            num_terminals += len(cell.terminals_id)
        assert_cell_bbox(cell, qt)
        if len(cell.children) != 0:
            for child in cell.children:
                assert qt.cells[child].parent == cell.id, cell.children
            if cell.parent is not None:
                assert cell.id in qt.cells[cell.parent].children
            assert len(cell.children) == 4
            assert_cell_children(cell, qt)
    assert num_terminals == len(qt.terminals)


def assert_stt(stt: SteinerTree) -> None:
    assert len(stt.points) == len(stt.point2edges), [
        len(stt.points),
        len(stt.point2edges),
    ]
    assert len(stt.points) == len(stt.terminals) + len(stt.stps), [
        len(stt.points),
        len(stt.terminals),
        len(stt.stps),
    ]
    assert len(stt.edges) == (len(stt.points) - 1), [len(stt.edges), len(stt.points)]

    stt.check_connection()
    assert stt.is_connected


def test_postprocess() -> None:
    qt: QuadTree = get_qt()
    data: QuadTreeData = get_data(qt)
    (adapted_stt, processed_stt) = NNSteiner.build_stt_data(data, 10, 1)
    assert_stt(adapted_stt)

    # postprocess
    processed_stt.check_trim()
    assert_stt(processed_stt)

    assert processed_stt.length() <= adapted_stt.length(), [
        processed_stt.length(),
        adapted_stt.length(),
    ]
