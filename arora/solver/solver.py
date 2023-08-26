from __future__ import annotations

import copy
from typing import Dict, List, Set, Tuple

import torch
from torch import Tensor

from arora.data import Transform
from arora.models import NNArora
from arora.quadtree import (
    Cell,
    Portal,
    QuadTree,
    QuadTreeData,
    Side,
    SteinerPoint,
    SteinerTree,
    Terminal,
)


class GeoSteiner:
    def __init__(self) -> None:
        pass

    def solve(self, terminals: List[Terminal], fst: int) -> SteinerTree:
        return SteinerTree.get_RSMT_GEO(terminals, fst)


class MST:
    def __init__(self) -> None:
        pass

    def solve(self, terminals: List[Terminal]) -> SteinerTree:
        return SteinerTree.get_mst(terminals, [])


class NNSteiner:
    def __init__(self, nn_arora: NNArora, transform: Transform, device: str) -> None:
        """
        device: the device for pytorch
        """
        super().__init__()

        self.device: str = device
        self.nn_arora: NNArora = nn_arora.to(device)
        self.transform: Transform = transform

    @torch.no_grad()
    def solve(
        self, qt: QuadTree, k: int, fst: int
    ) -> Tuple[Tensor, SteinerTree, SteinerTree]:
        inputs: Dict[str, Tensor] = {
            "tree_struct": self.transform.get_tree_struct(qt).to(self.device),
            "cell2bound": self.transform.get_cell2bound(qt).to(self.device),
            "cell2cross": self.transform.get_cell2cross(qt).to(self.device),
            "bound_tens": self.transform.get_bound_tens(qt)[None, :].to(self.device),
            "terminal_tens": self.transform.get_terminal_tens(qt)[None, :].to(
                self.device
            ),
            "null_portal_tens": self.transform.get_null_portal_tens()[None, :].to(
                self.device
            ),
        }

        self.nn_arora.eval()
        predict_tens: Tensor = self.nn_arora(**inputs)
        predict_tens = predict_tens.squeeze()

        # portal retrieval
        feasible_portals: List[int] = self._feasible_portals(0.5, predict_tens)

        # ablation study
        # feasible_portals: List[int] = []
        # predict_tens: Tensor = torch.ones(1)

        (predict_stt, final_stt) = self.build_stt(feasible_portals, qt, k, fst)
        return (predict_tens, predict_stt, final_stt)

    @staticmethod
    def build_stt_data(
        data: QuadTreeData, k: int, fst: int
    ) -> Tuple[SteinerTree, SteinerTree]:
        return NNSteiner.build_stt(data.feasible_portals, data, k, fst)

    @staticmethod
    def build_stt(
        feasible_portals: List[int], qt: QuadTree, k: int, fst: int
    ) -> Tuple[SteinerTree, SteinerTree]:
        (stps, portal2point) = NNSteiner._points_from_portals(feasible_portals, qt)
        predict_stt: SteinerTree = SteinerTree.get_mst(qt.terminals, stps)
        stt = copy.deepcopy(predict_stt)

        # post processing
        NNSteiner._refine_leaf_cells(stt, feasible_portals, portal2point, qt, fst)
        stt.post_process(k, fst)
        return (predict_stt, stt)

    @staticmethod
    def _feasible_portals(t: float, predict_tens: Tensor) -> List[int]:
        ret: List[int] = []
        for i, prob in enumerate(predict_tens):
            if prob >= t:
                ret.append(i)
        return ret

    @staticmethod
    def _points_from_portals(
        feasible_portals: List[int], qt: QuadTree
    ) -> Tuple[List[SteinerPoint], Dict[int, int]]:
        stps: List[SteinerPoint] = []
        portal2point: Dict[int, int] = {}

        # check overlap
        idx: Dict[Tuple[int, int], int] = {}
        for i, terminal in enumerate(qt.terminals):
            query: Tuple[int, int] = (terminal.x.reduce(), terminal.y.reduce())
            assert query not in idx, query
            idx[query] = i

        stp_count: int = 0
        for p_id in feasible_portals:
            portal: Portal = qt.portals[p_id]
            query: Tuple[int, int] = (portal.x.reduce(), portal.y.reduce())
            if query in idx:
                portal2point[p_id] = idx[query]
            else:
                stp_id: int = stp_count + len(qt.terminals)
                stp: SteinerPoint = SteinerPoint(stp_id, portal.x, portal.y)
                stps.append(stp)
                portal2point[p_id] = stp_id
                idx[query] = stp_id
                stp_count += 1

        return (stps, portal2point)

    @staticmethod
    def _refine_leaf_cells(
        stt: SteinerTree,
        feasible_portals: List[int],
        portal2point: Dict[int, int],
        qt: QuadTree,
        fst: int,
    ) -> SteinerTree:
        portals_set: Set[int] = set(feasible_portals)
        for cell in qt.cells:
            if cell.is_leaf():
                NNSteiner._refine_cell(cell, portals_set, portal2point, stt, qt, fst)
        return stt

    @staticmethod
    def _refine_cell(
        cell: Cell,
        portals_set: Set[int],
        portal2point: Dict[int, int],
        stt: SteinerTree,
        qt: QuadTree,
        fst: int,
    ) -> None:
        # gather all points, including terminals and portals
        cell_points: Set[int] = set()
        for t_id in cell.terminals_id:
            cell_points.add(t_id)
        cell_points.update(
            NNSteiner._get_side_points(
                cell.boundary.east, portals_set, portal2point, qt
            )
        )
        cell_points.update(
            NNSteiner._get_side_points(
                cell.boundary.south, portals_set, portal2point, qt
            )
        )
        cell_points.update(
            NNSteiner._get_side_points(
                cell.boundary.west, portals_set, portal2point, qt
            )
        )
        cell_points.update(
            NNSteiner._get_side_points(
                cell.boundary.north, portals_set, portal2point, qt
            )
        )

        # solve the connected subsets separately
        subsets: List[List[int]] = NNSteiner._find_connected_subsets(cell_points, stt)

        stt.refine_subsets_GEO(subsets, fst)

    @staticmethod
    def _get_side_points(
        side_id: int,
        portals_set: Set[int],
        portal2point: Dict[int, int],
        qt: QuadTree,
    ) -> List[int]:
        ret: List[int] = []
        side: Side = qt.sides[side_id]
        for portal_id in side.portals_id:
            if portal_id is not None:
                if portal_id in portals_set:
                    ret.append(portal2point[portal_id])
        return ret

    @staticmethod
    def _find_connected_subsets(
        points_id: Set[int], stt: SteinerTree
    ) -> List[List[int]]:
        """
        return list of subsets.
        each subset has a point set and an edge set
        """
        visited_points: Set[int] = set()

        def dfs(point_id: int, subset: List[int]) -> None:
            # mark visited point
            visited_points.add(point_id)
            subset.append(point_id)
            # find all neighbors
            for neighbor in stt.get_neighbors(point_id):
                if (neighbor in points_id) and (neighbor not in visited_points):
                    dfs(neighbor, subset)

        ret: List[List[int]] = []
        for point in points_id:
            if point not in visited_points:
                subset: List[int] = []
                dfs(point, subset)
                ret.append(subset)
        return ret

    def _stt_is_better(self, new_stt: SteinerTree, stt: SteinerTree | None) -> bool:
        new_cost = new_stt.length()
        if stt is None:
            return True
        cost = stt.length()
        return new_cost < cost

    # def _ite_retrieve(self) -> SteinerTree:
    #     # FIXME: t0 and delta_t should be hyperparameter
    #     t0: float = 0.5
    #     delta_t: float = 0.05

    #     stt: SteinerTree | None = None
    #     for i in range(int(t0 / delta_t)):
    #         t = t0 - delta_t * i
    #         new_stt = self.portal_retrieve(t)
    #         if self._stt_is_better(new_stt, stt):
    #             stt = new_stt
    #         else:
    #             break
    #     assert stt is not None
    #     return stt
