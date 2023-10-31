from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from arora.fix import fix

from .shape import Point, PurePoint, Rect, Terminal


class Portal(Point):
    def __init__(self, id: int, x: fix, y: fix, side_id: int) -> None:
        super().__init__(id, x, y)
        self.side_id: int = side_id

    @property
    def typ(self) -> str:
        return "portal"


class LPoint(Point):
    def __init__(self, id: int, x: fix, y: fix) -> None:
        super().__init__(id, x, y)

    @property
    def typ(self) -> str:
        return "l-point"


@dataclass
class LShapeTag:
    ledge_id: int
    lpoint: LPoint


class Side:
    def __init__(
        self,
        id: int,
        cell_id: int,
        parent_id: int | None,
        typ: str | None,
        pivot: fix,
        span: Tuple[fix, fix],
        m: int,
    ) -> None:
        self.id: int = id
        self.cell_id: int = cell_id
        self.parent_id: int | None = parent_id
        self.typ: str | None = typ
        """\"horizontal\" or \"vertical\". None if it's the canvas boundary"""

        self.pivot: fix = copy.deepcopy(pivot)

        (start, end) = span
        assert start < end
        self.span: Tuple[fix, fix] = (copy.deepcopy(start), copy.deepcopy(end))

        self.children: List[int] = []

        self.m = m
        """Number of portals between the corners"""

        self.portals_id: List[int | None] = [None for _ in range(self.m + 2)]
        """The portals should be the corners and the (\"m\"-1) inside portals"""

    @property
    def len(self) -> fix:
        (start, end) = self.span
        assert end > start
        return end - start

    @property
    def interval(self) -> fix:
        return self.len // (self.m + 1)

    @property
    def cent(self) -> fix:
        return (self.span[0] + self.span[1]) // 2

    def intersect(self, point1: Point, point2: Point) -> PurePoint | None:
        assert (
            point1.x == point2.x or point1.y == point2.y
        ), f"point1: ({point1.x},{point1.y}) and point2: ({point2.x}, {point2.y}) do not match"
        if self.typ is None:
            return None
        if self.typ == "vertical":
            # check if parallel
            if point1.x == point2.x:
                return None
            # check if points on differnt side
            if (
                False
                or (point1.x > self.pivot and point2.x > self.pivot)
                or (point1.x < self.pivot and point2.x < self.pivot)
            ):
                return None
            # check if points with span
            if self.span[0] <= point1.y <= self.span[1]:
                return PurePoint(-1, self.pivot, point1.y)
        if self.typ == "horizontal":
            # check if parallel
            if point1.y == point2.y:
                return None
            # check if points on differnt side
            if (
                False
                or (point1.y > self.pivot and point2.y > self.pivot)
                or (point1.y < self.pivot and point2.y < self.pivot)
            ):
                return None
            # check if points with span
            if self.span[0] <= point1.x <= self.span[1]:
                return PurePoint(-1, point1.x, self.pivot)


@dataclass
class OuterSides:
    east: int
    south: int
    west: int
    north: int


@dataclass
class InnerSides:
    right: int
    bottom: int
    left: int
    top: int


@dataclass
class Cell:
    """
    A bounding box inside a quadtree.
    """

    id: int
    bbox: Rect
    parent: int | None
    """None is the cell is the root"""
    terminals_id: List[int]
    boundary: OuterSides
    cross: InnerSides | None = None
    """None if the cell is a leaf"""
    children: List[int] = field(default_factory=list)
    """Empty if the cell is a leaf"""

    def is_leaf(self) -> bool:
        no_child: bool = len(self.children) == 0
        no_cross: bool = self.cross is None
        assert no_child == no_cross, [no_child, no_cross]
        return no_child


class QuadTree:
    def __init__(
        self, terminals: List[Terminal], bbox: Rect, args: Dict[str, int]
    ) -> None:
        # init data members
        self.args = args
        """
        \"m\": number of portals between the two corners
        \"kb\": the maximum number of portals in a leaf cell
        """
        assert (
            False
            or self.args["m"] == 1
            or self.args["m"] == 3
            or self.args["m"] == 7
            or self.args["m"] == 15
            or self.args["m"] == 31
        ), "m should be power of 2"
        self.bbox = bbox
        self.cells: List[Cell] = []
        self.sides: List[Side] = []
        self.portals: List[Portal] = []
        self.terminals: List[Terminal] = terminals
        self.terminal2cell: List[int] = [0 for _ in range(len(terminals))]

        # check if the bbox is valid
        for terminal in terminals:
            assert (
                True
                and bbox.x_min <= terminal.x <= bbox.x_max
                and bbox.y_min <= terminal.y <= bbox.y_max
            ), bbox

        # parameters
        self.m: int = self.args["m"]
        self.kb: int = self.args["kb"]
        self.n_side_p: int = self.m + 2
        self.n_cell_p: int = self.n_side_p * 4
        self.level: int = self.args["level"]
        self.max_level: int = -1

        # build root cell
        # build sides for root cell
        east: Side = Side(
            0, 0, None, None, bbox.x_max, (bbox.y_min, bbox.y_max), self.args["m"]
        )
        south: Side = Side(
            0, 0, None, None, bbox.y_min, (bbox.x_min, bbox.x_max), self.args["m"]
        )
        west: Side = Side(
            0, 0, None, None, bbox.x_min, (bbox.y_min, bbox.y_max), self.args["m"]
        )
        north: Side = Side(
            0, 0, None, None, bbox.y_max, (bbox.x_min, bbox.x_max), self.args["m"]
        )
        self.sides.append(east)
        self.sides.append(south)
        self.sides.append(west)
        self.sides.append(north)
        root_cell: Cell = Cell(
            0, bbox, None, [i for i in range(len(terminals))], OuterSides(0, 1, 2, 3)
        )
        self.cells.append(root_cell)

        # construct the quadCells recursively
        self._divide_cell(0, 0)

        # # update the connection between portals and the nearest terminal
        # self._connect_portals()

    def _divide_cell(self, cell_id: int, level: int) -> None:
        cell: Cell = self.cells[cell_id]
        if (len(cell.terminals_id) <= self.args["kb"]) and (level >= self.level):
            return
        if self.max_level < level:
            self.max_level = level

        # build new sides in the parent cell
        right_side_id: int = len(self.sides)
        bottom_side_id: int = right_side_id + 1
        left_side_id: int = bottom_side_id + 1
        top_side_id: int = left_side_id + 1
        right_side: Side = Side(
            right_side_id,
            cell_id,
            None,
            "horizontal",
            cell.bbox.y_cent,
            (cell.bbox.x_cent, cell.bbox.x_max),
            self.args["m"],
        )
        self.sides.append(right_side)
        bottom_side: Side = Side(
            bottom_side_id,
            cell_id,
            None,
            "vertical",
            cell.bbox.x_cent,
            (cell.bbox.y_min, cell.bbox.y_cent),
            self.args["m"],
        )
        self.sides.append(bottom_side)
        left_side: Side = Side(
            left_side_id,
            cell_id,
            None,
            "horizontal",
            cell.bbox.y_cent,
            (cell.bbox.x_min, cell.bbox.x_cent),
            self.args["m"],
        )
        self.sides.append(left_side)
        top_side: Side = Side(
            top_side_id,
            cell_id,
            None,
            "vertical",
            cell.bbox.x_cent,
            (cell.bbox.y_cent, cell.bbox.y_max),
            self.args["m"],
        )
        self.sides.append(top_side)
        self._portal_new(right_side_id)
        self._portal_new(bottom_side_id)
        self._portal_new(left_side_id)
        self._portal_new(top_side_id)
        cell.cross = InnerSides(
            right_side_id, bottom_side_id, left_side_id, top_side_id
        )

        # build the children
        self._build_child(cell.id, top_side_id, right_side_id, "first", level + 1)
        self._build_child(cell.id, top_side_id, left_side_id, "second", level + 1)
        self._build_child(cell.id, bottom_side_id, left_side_id, "third", level + 1)
        self._build_child(cell.id, bottom_side_id, right_side_id, "fourth", level + 1)

    def _update_corner(self, p: Portal, side: Side) -> int:
        """return the id of the corner portal"""
        (portal_exist, portal_id) = self._portal_in_side(side)
        if not portal_exist:
            portal_id = len(self.portals)
            self.portals.append(p)
        else:
            origin_portal: Portal = self.portals[portal_id]
            assert p.x == origin_portal.x, p.y == origin_portal.y
        return portal_id

    def _portal_in_side(self, side: Side) -> Tuple[bool, int]:
        """if the portal is in the side, then return the portal id, else return 0"""
        portal_id: int | None = side.portals_id[(self.args["m"] + 1) // 2]
        if portal_id is not None:
            return (True, portal_id)
        return (False, 0)

    def _portal_new(self, side_id: int) -> None:
        side: Side = self.sides[side_id]

        # create the portals in between
        interval: fix = side.len // (self.args["m"] + 1)
        for i in range(0, self.args["m"] + 2):
            point: fix = side.span[0] + interval * i
            assert (
                point <= side.span[1]
            ), f"interval: {point} should be within ({side.span[0], side.span[1]},)"
            if side.typ == "horizontal":
                p: Portal = Portal(len(self.portals), point, side.pivot, side_id)
            elif side.typ == "vertical":
                p: Portal = Portal(len(self.portals), side.pivot, point, side_id)
            else:
                assert False, f'side.typ should not be "{side.typ}"'
            assert p.id <= len(self.portals)
            assert side.portals_id[i] is None, side.portals_id
            side.portals_id[i] = p.id
            self.portals.append(p)

    def _build_child(
        self,
        parent_cell_id: int,
        new_side_v_id: int,
        new_side_h_id: int,
        quad: str,
        level: int,
    ) -> None:
        parent_cell: Cell = self.cells[parent_cell_id]
        parent_bbox: Rect = parent_cell.bbox
        new_bbox: Rect = self._get_bbox_by_quad(parent_bbox, quad)
        new_cell_id: int = len(self.cells)

        # get all the terminals in it
        terminals_id_in_bbox: List[int] = []
        for terminal_id in parent_cell.terminals_id:
            terminal: Terminal = self.terminals[terminal_id]
            if new_bbox.contains(terminal):
                terminals_id_in_bbox.append(terminal_id)
                assert len(self.terminal2cell) == len(
                    self.terminals
                ), f"len(self.terminal2cell): {len(self.terminal2cell)} does not match with len(self.terminals): {len(self.terminals)}"
                self.terminal2cell[terminal_id] = new_cell_id

        # inherit the two old sides
        if quad == "first":
            east_side = self._inherit_side(
                new_cell_id, parent_cell.boundary.east, (new_bbox.y_min, new_bbox.y_max)
            )
            south_side = new_side_h_id
            west_side = new_side_v_id
            north_side = self._inherit_side(
                new_cell_id,
                parent_cell.boundary.north,
                (new_bbox.x_min, new_bbox.x_max),
            )
        elif quad == "second":
            east_side = new_side_v_id
            south_side = new_side_h_id
            west_side = self._inherit_side(
                new_cell_id, parent_cell.boundary.west, (new_bbox.y_min, new_bbox.y_max)
            )
            north_side = self._inherit_side(
                new_cell_id,
                parent_cell.boundary.north,
                (new_bbox.x_min, new_bbox.x_max),
            )
        elif quad == "third":
            east_side = new_side_v_id
            south_side = self._inherit_side(
                new_cell_id,
                parent_cell.boundary.south,
                (new_bbox.x_min, new_bbox.x_max),
            )
            west_side = self._inherit_side(
                new_cell_id, parent_cell.boundary.west, (new_bbox.y_min, new_bbox.y_max)
            )
            north_side = new_side_h_id
        elif quad == "fourth":
            east_side = self._inherit_side(
                new_cell_id, parent_cell.boundary.east, (new_bbox.y_min, new_bbox.y_max)
            )
            south_side = self._inherit_side(
                new_cell_id,
                parent_cell.boundary.south,
                (new_bbox.x_min, new_bbox.x_max),
            )
            west_side = new_side_v_id
            north_side = new_side_h_id
        else:
            assert False, f'quadrant should not be "{quad}"'

        child_id: int = len(self.cells)
        child: Cell = Cell(
            child_id,
            new_bbox,
            parent_cell_id,
            terminals_id_in_bbox,
            OuterSides(east_side, south_side, west_side, north_side),
            None,
        )
        parent_cell.children.append(child_id)
        self.cells.append(child)
        self._divide_cell(child_id, level)

    def _inherit_side(self, cell_id: int, p_side_id: int, span: Tuple[fix, fix]) -> int:
        """
        typ: str
        \"vertical\" or \"horizontal\"
        """
        p_side: Side = self.sides[p_side_id]
        assert len(p_side.portals_id) == self.args["m"] + 2

        # to see if the side already exists
        if len(p_side.children) > 0:
            assert len(p_side.children) <= 2, p_side.children
            for child_id in p_side.children:
                child: Side = self.sides[child_id]
                if child.span == span:
                    return child_id

        # if does not exist
        assert len(p_side.children) < 2, p_side.children
        child_id: int = len(self.sides)
        p_side.children.append(child_id)

        side: Side = Side(
            child_id,
            cell_id,
            p_side_id,
            p_side.typ,
            p_side.pivot,
            span,
            self.args["m"],
        )
        self.sides.append(side)

        if span == (p_side.span[0], p_side.cent):
            offset: int = 0
        elif span == (p_side.cent, p_side.span[1]):
            offset: int = (self.args["m"] + 1) // 2
        else:
            assert False, [p_side.span, span]

        if side.typ is not None:
            # inherit portals from parents
            for i in range(self.args["m"] + 2):
                p_portal_id = i // 2 + offset
                if i % 2 == 0:
                    side.portals_id[i] = p_side.portals_id[p_portal_id]
                elif i % 2 == 1:
                    side.portals_id[i] = None
                else:
                    assert False
        return child_id

    def _get_bbox_by_quad(self, parent: Rect, quad: str) -> Rect:
        if quad == "first":
            return Rect(parent.x_cent, parent.y_cent, parent.x_max, parent.y_max)
        elif quad == "second":
            return Rect(parent.x_min, parent.y_cent, parent.x_cent, parent.y_max)
        elif quad == "third":
            return Rect(parent.x_min, parent.y_min, parent.x_cent, parent.y_cent)
        elif quad == "fourth":
            return Rect(parent.x_cent, parent.y_min, parent.x_max, parent.y_cent)
        else:
            assert False, f'quadrant should not be "{quad}"'

    def check_level(self, level: int) -> None:
        # check if the quadtree is a full tree with level
        level_count: int = -1
        level_cells: List[int] = [0]
        while len(level_cells) > 0:
            level_count += 1
            next_level_cells: List[int] = []
            for cell_id in level_cells:
                cell: Cell = self.cells[cell_id]
                if level_count < level:
                    assert len(cell.children) == 4, cell
                next_level_cells.extend(cell.children)
            level_cells = next_level_cells
        assert level_count == level, [level_count, level]

    @staticmethod
    def is_isomorphic(qt_a: QuadTree, qt_b: QuadTree) -> bool:
        if len(qt_a.cells) != len(qt_b.cells):
            return False
        for cell_a, cell_b in zip(qt_a.cells, qt_b.cells):
            if not cell_a.children == cell_b.children:
                return False
        return True

    @staticmethod
    def arg_clustering(qts: List[QuadTree], max_size: int) -> List[List[int]]:
        # first step clustering best on num of cells
        hash_num_cell: Dict[int, List[int]] = {}
        for i, qt in enumerate(qts):
            num_cell: int = len(qt.cells)
            if num_cell not in hash_num_cell:
                hash_num_cell[num_cell] = [i]
            else:
                hash_num_cell[num_cell].append(i)

        ret: List[List[int]] = []
        for idx_list in hash_num_cell.values():
            ret.extend(QuadTree.clustering_Osquare(idx_list, qts, max_size))
        return ret

    def _clean_terminals(self) -> None:
        self.terminals = []
        self.terminal2cell = []
        for cell in self.cells:
            cell.terminals_id = []

    def _add_terminals(self, terminals: List[Terminal]) -> None:
        self.terminals.extend(terminals)
        self.terminal2cell.extend([-1 for _ in range(len(terminals))])

        def add_terminals2cell(cell_id: int, terms_id: List[int]) -> None:
            cell: Cell = self.cells[cell_id]
            for term_id in terms_id:
                terminal: Terminal = self.terminals[term_id]
                if cell.bbox.contains(terminal):
                    cell.terminals_id.append(term_id)
                    self.terminal2cell[term_id] = cell_id

            for child_id in cell.children:
                add_terminals2cell(child_id, cell.terminals_id)

            if cell.is_leaf():
                assert len(cell.terminals_id) <= self.kb, [
                    len(cell.terminals_id),
                    self.kb,
                ]

        add_terminals2cell(0, [i for i in range(len(terminals))])

        assert -1 not in self.terminal2cell, self.terminal2cell

    @staticmethod
    def clustering_Osquare(
        idx_list: List[int], qts: List[QuadTree], max_size: int
    ) -> List[List[int]]:
        ret: List[List[int]] = []

        for idx in idx_list:
            qt: QuadTree = qts[idx]
            for cluster in ret:
                qt_ref: QuadTree = qts[cluster[0]]
                if QuadTree.is_isomorphic(qt_ref, qt) and len(cluster) < max_size:
                    cluster.append(idx)
                    break
            else:
                ret.append([idx])

        return ret

    @staticmethod
    def clustering(qts: List[QuadTree], max_size: int) -> List[List[QuadTree]]:
        arg_cluster = QuadTree.arg_clustering(qts, max_size)

        ret: List[List[QuadTree]] = [
            [qts[idx] for idx in idx_cluster] for idx_cluster in arg_cluster
        ]

        return ret

    @staticmethod
    def tree_like(other: QuadTree, terminals: List[Terminal]) -> QuadTree:
        ret: QuadTree = copy.deepcopy(other)
        ret._clean_terminals()
        ret._add_terminals(terminals)
        return ret

    # def _connect_portals(self) -> None:
    #     print("connecting portals...")
    #     self.portal2terminal: List[int | None] = [
    #         None for _ in range(len(self.portals))
    #     ]

    #     def connect_portals_side(side_id: int, index: rtree.Index) -> None:
    #         side: Side = self.sides[side_id]
    #         for portal_id in side:
    #             if portal_id is None:
    #                 continue
    #             portal: Portal = self.portals[portal_id]
    #             if portal.side_id == side_id:
    #                 terminals = list(
    #                     index.nearest((portal.x.to_float(), portal.y.to_float()), 1)
    #                 )
    #                 assert (
    #                     self.portal2terminal[portal_id] is None
    #                 ), f"Error occurs in Cell[{side.cell_id}]'s side[{side_id}]: {self.portal2terminal[portal_id]}"
    #                 self.portal2terminal[portal_id] = terminals[0]

    #     for cell in self.cells:
    #         # update rtree for spatial query
    #         index: rtree.Index = rtree.index.Index()
    #         for (i, terminal_id) in enumerate(cell.terminals_id):
    #             terminal: Terminal = self.terminals[terminal_id]
    #             index.insert(i, (terminal.x.to_float(), terminal.y.to_float()))
