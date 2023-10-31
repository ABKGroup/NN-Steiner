from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import rtree

from arora.fix import fix

from .quadTree import Cell, Portal, QuadTree
from .shape import Point, PurePoint, Rect, Terminal
from .steinerTree import SteinerPoint, SteinerTree


class LPoint(Point):
    def __init__(self, id: int, x: fix, y: fix, edge_id: int) -> None:
        super().__init__(id, x, y)
        self.edge_id = edge_id

    @property
    def typ(self) -> str:
        return "l-point"


@dataclass
class DataCell:
    stps_id: List[int]
    lpoints_id: List[int]


class QuadTreeData(QuadTree):
    def __init__(
        self,
        terminals: List[Terminal],
        bbox: Rect,
        args: Dict[str, int],
        stt: SteinerTree,
    ) -> None:
        super().__init__(terminals, bbox, args)

        self._process_stt(stt)

    def _process_stt(self, stt):
        self.golden_stt = stt
        self.stt = copy.deepcopy(stt)

        self.data_cells: List[DataCell] = [
            DataCell([], []) for _ in range(len(self.cells))
        ]

        # put the steiner points into the cells
        self.data_cells[0].stps_id = [i for i in range(len(self.stt.stps))]
        self._update_stps(0, None)

        # put the lpoints into the cells
        self._set_lshapes()
        self.data_cells[0].lpoints_id = [i for i in range(len(self.lpoints))]
        self._update_lpoints(0, None)

        self._portal_snapping()

        self.feasible_portals: List[int] = [
            i for (i, used) in enumerate(self.portal_used) if used
        ]

    def _update_stps(self, cell_id: int, parent_id: int | None) -> None:
        cell: Cell = self.cells[cell_id]
        d_cell: DataCell = self.data_cells[cell_id]
        if parent_id is not None:
            assert parent_id == cell.parent, [parent_id, cell.parent]
            p_cell: Cell = self.cells[parent_id]
            p_d_cell: DataCell = self.data_cells[parent_id]
            for stp_id in p_d_cell.stps_id:
                stp: SteinerPoint = self.stt.stps[stp_id]
                if p_cell.bbox.contains(stp):
                    d_cell.stps_id.append(stp_id)

        for child_id in cell.children:
            self._update_stps(child_id, cell_id)

    def _update_lpoints(self, cell_id: int, parent_id: int | None) -> None:
        cell: Cell = self.cells[cell_id]
        d_cell: DataCell = self.data_cells[cell_id]
        if parent_id is not None:
            assert parent_id == cell.parent, [parent_id, cell.parent]
            p_cell: Cell = self.cells[parent_id]
            p_d_cell: DataCell = self.data_cells[parent_id]
            for lpoint_id in p_d_cell.lpoints_id:
                lpoint: LPoint = self.lpoints[lpoint_id]
                if p_cell.bbox.contains(lpoint):
                    d_cell.lpoints_id.append(lpoint_id)

        for child_id in cell.children:
            self._update_lpoints(child_id, cell_id)

    def _set_lshapes(self) -> None:
        self.ledge2lpoint: Dict[int, int] = {}
        self.lpoints: List[LPoint] = []

        # put all the portals in to the rtree
        index = rtree.index.Index()
        for i, portal in enumerate(self.portals):
            index.insert(i, (portal.x.to_float(), portal.y.to_float()))

        # for all the L shape edges
        for edge in self.stt.edges:
            if edge.typ == "lshape":
                lpoint_id: int = len(self.lpoints)
                self.ledge2lpoint[edge.id] = lpoint_id

                # create the query box, which is the bbox of the edge
                src: Point = self.stt.get_point(edge.src)
                tgt: Point = self.stt.get_point(edge.tgt)
                bbox = Rect(
                    min(src.x, tgt.x),
                    min(src.y, tgt.y),
                    max(src.x, tgt.x),
                    max(src.y, tgt.y),
                )

                # if there are points within the query box, then the
                # lshape can be any, else the lshape should depend on the
                # relationship with the nearest point
                inside = index.contains(bbox.to_nparray())
                assert inside is not None
                inside = list(inside)
                if inside:
                    self.lpoints.append(LPoint(lpoint_id, src.x, tgt.y, edge.id))
                else:
                    nearest = list(index.nearest(bbox.to_nparray(), num_results=1))
                    lpoint_1: LPoint = LPoint(lpoint_id, src.x, tgt.y, edge.id)
                    lpoint_2: LPoint = LPoint(lpoint_id, tgt.x, src.y, edge.id)
                    p_near: Portal = self.portals[nearest[0]]
                    dist_1 = Point.L1_dist(p_near, lpoint_1)
                    dist_2 = Point.L1_dist(p_near, lpoint_2)
                    if dist_2 < dist_1:
                        self.lpoints.append(lpoint_2)
                    else:
                        self.lpoints.append(lpoint_1)
            elif edge.typ == "straight":
                continue
            else:
                assert False, edge.typ

    def _portal_snapping(self) -> None:
        self.portal_used: List[bool] = [False for _ in range(len(self.portals))]
        terminal_set: Set[Tuple[int, int]] = set()
        for terminal in self.terminals:
            query: Tuple[int, int] = (terminal.x.reduce(), terminal.y.reduce())
            terminal_set.add(query)
        for edge in self.stt.edges:
            src: Point = self.stt.get_point(edge.src)
            tgt: Point = self.stt.get_point(edge.tgt)
            if edge.typ == "straight":
                self._update_portal(src, tgt, terminal_set)
            elif edge.typ == "lshape":
                lpoint: LPoint = self.lpoints[self.ledge2lpoint[edge.id]]
                assert (
                    False
                    or (src.x == lpoint.x and tgt.y == lpoint.y)
                    or (src.y == lpoint.y and tgt.x == lpoint.x)
                )
                self._update_portal(src, lpoint, terminal_set)
                self._update_portal(lpoint, tgt, terminal_set)

    def _update_portal(
        self, point1: Point, point2: Point, terminal_set: Set[Tuple[int, int]]
    ) -> None:
        for side in self.sides:
            inter_point: PurePoint | None = side.intersect(point1, point2)
            if inter_point is not None:
                dists: np.ndarray = np.array(
                    [
                        Point.L1_dist(self.portals[portal_id], inter_point)
                        if portal_id is not None
                        else np.inf
                        for portal_id in side.portals_id
                    ]
                )
                if np.all(dists == np.inf):
                    continue
                snapped_portal_id = side.portals_id[np.argmin(dists)]
                assert snapped_portal_id is not None, [side.portals_id, dists]

                # see if the portal overlap with terminal, if so, trim it
                portal: Portal = self.portals[snapped_portal_id]
                query: Tuple[int, int] = (portal.x.reduce(), portal.y.reduce())
                if query in terminal_set:
                    continue
                self.portal_used[snapped_portal_id] = True

    @staticmethod
    def arg_clustering(qts: List[QuadTreeData], max_size: int) -> List[List[int]]:
        return QuadTree.arg_clustering(qts, max_size)  # pyright: ignore

    @staticmethod
    def clustering(qts: List[QuadTreeData], max_size: int) -> List[List[QuadTreeData]]:
        return QuadTree.clustering(qts, max_size)  # pyright: ignore

    @staticmethod
    def tree_like(
        other: QuadTree, terminals: List[Terminal], stt: SteinerTree
    ) -> QuadTreeData:
        ret: QuadTreeData = QuadTreeData([], other.bbox, other.args, SteinerTree([]))
        ret.cells = copy.deepcopy(other.cells)
        ret.sides = copy.deepcopy(other.sides)
        ret.portals = copy.deepcopy(other.portals)
        ret._clean_terminals()
        ret._add_terminals(terminals)
        ret._process_stt(stt)
        return ret
