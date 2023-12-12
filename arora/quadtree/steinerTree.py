from __future__ import annotations

import copy
import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from arora.fix import fix
from arora.utils import L1_dist, points_int2float

from .lib import find_RSMT_GEO
from .rmst import find_RMST
from .shape import Point, Terminal


def find_RSMT_GEO_(
    points: List[Tuple[int, int]], fst: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    (stps_raw, edges_raw) = find_RSMT_GEO(points, fst)
    return (stps_raw, edges_raw)


def find_RMST_(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    point_set: Set[Tuple[int, int]] = set(points)
    assert len(point_set) == len(points), [len(point_set), len(points)]
    return find_RMST(points)


class SteinerPoint(Point):
    def __init__(self, id: int, x: fix, y: fix) -> None:
        super().__init__(id, x, y)
        self.valid: bool = True

    @property
    def typ(self) -> str:
        return "steiner-point"


@dataclass
class Edge:
    id: int
    src: int
    tgt: int
    typ: str
    """\"straight\" or \"lshape\""""
    valid: bool = True

    @staticmethod
    def create(edge_id: int, src: Point, tgt: Point) -> Edge:
        assert src.id != tgt.id, [src.id, tgt.id]
        if Point.is_lshape(src, tgt):
            edge_typ: str = "lshape"
        else:
            edge_typ: str = "straight"
        return Edge(
            edge_id,
            src.id,
            tgt.id,
            edge_typ,
        )


@dataclass
class SteinerTree:
    def __init__(
        self,
        terminals: List[Terminal],
        stps: List[SteinerPoint] | None = None,
        edges: List[Edge] | None = None,
    ) -> None:
        """
        The Steiner tree is maintained by lazy update.
        """
        # set up data structure
        self.points: List[Point] = []
        self.edges: List[Edge] = []
        self.point2edges: List[List[int]] = []

        # parameters
        self.num_terminals: int = len(terminals)
        self.cleaned: bool = True

        # init data structure
        self.points.extend(terminals)
        self.point2edges.extend([] for _ in range(self.num_terminals))
        if stps is not None:
            self.points.extend(stps)
            self.point2edges.extend([] for _ in range(len(stps)))
        if edges is not None:
            for edge in edges:
                assert edge.src != edge.tgt, [edge.src, edge.tgt]
                self.insert_edge(edge)

    @property
    def terminals(self) -> List[Terminal]:
        ret: List[Terminal] = []
        for i in range(self.num_terminals):
            point: Point = self.get_point(i)
            assert isinstance(point, Terminal)
            ret.append(point)
        return ret

    @property
    def stps(self) -> List[SteinerPoint]:
        ret: List[SteinerPoint] = []
        for i in range(self.num_terminals, len(self.points)):
            point: Point = self.get_point(i)
            assert isinstance(point, SteinerPoint)
            if point.valid:
                ret.append(point)
        return ret

    def get_point(self, idx: int) -> Point:
        ret: Point = self.points[idx]
        assert ret.id == idx, [ret.id, idx]
        return ret

    def get_edge(self, idx: int) -> Edge:
        ret: Edge = self.edges[idx]
        assert ret.id == idx, [ret.id, idx]
        return ret

    def get_another_point(self, edge_id: int, point_id: int) -> int:
        edge: Edge = self.edges[edge_id]
        if edge.src == point_id:
            assert edge.tgt != point_id, [edge.tgt, point_id]
            return edge.tgt
        elif edge.tgt == point_id:
            assert edge.src != point_id, [edge.src, point_id]
            return edge.src
        else:
            assert False, [edge.src, edge.tgt, point_id]

    def get_neighbors(self, point_id: int) -> List[int]:
        ret: List[int] = []
        for edge_id in self.point2edges[point_id]:
            ret.append(self.get_another_point(edge_id, point_id))
        return ret

    def length(self) -> float:
        self.cleanup()
        sum: float = 0.0
        for edge in self.edges:
            src: Point = self.get_point(edge.src)
            tgt: Point = self.get_point(edge.tgt)
            dist: float = Point.L1_dist(src, tgt)
            sum += dist
        return sum

    def normalized_length(self) -> float:
        self.cleanup()
        points: List[Tuple[int, int]] = [
            (point.x.to_int(), point.y.to_int()) for point in self.points
        ]
        float_points: List[Tuple[float, float]] = points_int2float(points)
        ret: float = 0.0
        for edge in self.edges:
            src: Tuple[float, float] = float_points[edge.src]
            tgt: Tuple[float, float] = float_points[edge.tgt]
            ret += L1_dist(src, tgt)
        return ret

    def insert_point(self, point: Point) -> None:
        assert point.id == len(self.points), [point.id, len(self.points)]
        self.points.append(point)
        self.point2edges.append([])

    def insert_edge(
        self, edge: Edge, point1: Point | None = None, point2: Point | None = None
    ) -> None:
        # insert points
        if point1 is not None:
            self.insert_point(point1)
        if point2 is not None:
            self.insert_point(point2)

        # insert edge
        assert edge.id == len(self.edges)
        self.edges.append(edge)

        # update point2edges
        src_id: int = edge.src
        self.point2edges[src_id].append(edge.id)
        tgt_id: int = edge.tgt
        self.point2edges[tgt_id].append(edge.id)

    def delete_point(self, point_id: int) -> List[int]:
        # this operation needs clean up
        self.cleaned = False

        assert point_id < len(self.points), [point_id, len(self.points)]
        # delete point
        stp: Point = self.get_point(point_id)
        assert isinstance(stp, SteinerPoint)
        assert stp.valid
        stp.valid = False

        # delete corresponding edges
        edges_id: List[int] = copy.deepcopy(self.point2edges[point_id])
        for edge_id in edges_id:
            self.delete_edge(edge_id)

        assert len(self.point2edges[point_id]) == 0, self.point2edges[point_id]
        return edges_id

    def delete_edge(self, edge_id: int) -> None:
        # this operation needs clean up
        self.cleaned = False

        edge: Edge = self.edges[edge_id]
        assert edge.valid
        edge.valid = False

        # delete corresponding point2edges
        src_id: int = edge.src
        self.point2edges[src_id].remove(edge_id)
        tgt_id: int = edge.tgt
        self.point2edges[tgt_id].remove(edge_id)

    def cleanup(self) -> None:
        # set the status to cleaned
        if self.cleaned:
            return
        self.cleaned = True

        # set up new data structure
        new_points: List[Point] = []
        new_edges: List[Edge] = []

        # init mapping
        old2new_point: Dict[int, int] = {}

        # clean points
        point_count: int = 0
        for point_id in range(len(self.points)):
            point: Point = self.get_point(point_id)
            if point_id < self.num_terminals:
                assert isinstance(point, Terminal)
                assert point_id == point_count, [point_id, point_count]
                new_points.append(point)
            else:
                assert isinstance(point, SteinerPoint)
                if not point.valid:
                    assert len(self.point2edges[point_id]) == 0, self.point2edges[
                        point_id
                    ]
                    continue
                new_points.append(SteinerPoint(point_count, point.x, point.y))
            old2new_point[point_id] = point_count
            point_count += 1
        assert len(new_points) == point_count, [len(new_points), point_count]

        # clean edges, check if there is redundant edge
        edge_count: int = 0
        idx: Set[Tuple[int, int]] = set()
        for edge_id in range(len(self.edges)):
            edge: Edge = self.get_edge(edge_id)
            query1: Tuple[int, int] = (edge.src, edge.tgt)
            query2: Tuple[int, int] = (edge.tgt, edge.src)
            if not edge.valid or (query1 in idx) or (query2 in idx):
                continue
            new_src_id: int = old2new_point[edge.src]
            new_tgt_id: int = old2new_point[edge.tgt]
            new_edges.append(Edge(edge_count, new_src_id, new_tgt_id, edge.typ))
            edge_count += 1
            idx.add(query1)
        assert len(new_edges) == edge_count, [len(new_edges), edge_count]

        # update point2edges
        new_point2edges: List[List[int]] = [[] for _ in range(len(new_points))]
        assert len(new_points) == len(new_edges) + 1, [len(new_points), len(new_edges)]
        for edge in new_edges:
            new_point2edges[edge.src].append(edge.id)
            new_point2edges[edge.tgt].append(edge.id)
            pass

        self.points = new_points
        self.edges = new_edges
        self.point2edges = new_point2edges

    @staticmethod
    def new_edges(
        edges_raw: List[Tuple[int, int]],
        terminals: List[Terminal],
        stps: List[SteinerPoint],
    ):
        edges: List[Edge] = []

        for i, edge_raw in enumerate(edges_raw):
            idx: Tuple[int, int] = edge_raw
            (src_idx, tgt_idx) = idx

            assert src_idx != tgt_idx, [src_idx, tgt_idx]
            # update the edges
            edges.append(SteinerTree.create_edge(i, src_idx, tgt_idx, terminals, stps))
        return edges

    @staticmethod
    def create_edge(
        edge_idx: int,
        src_idx: int,
        tgt_idx: int,
        terminals: List[Terminal],
        stps: List[SteinerPoint],
    ) -> Edge:
        ntps: int = len(terminals)
        if src_idx >= ntps and tgt_idx >= ntps:
            return Edge.create(edge_idx, stps[src_idx - ntps], stps[tgt_idx - ntps])
        elif src_idx >= ntps and tgt_idx < ntps:
            return Edge.create(edge_idx, stps[src_idx - ntps], terminals[tgt_idx])
        elif src_idx < ntps and tgt_idx >= ntps:
            return Edge.create(edge_idx, terminals[src_idx], stps[tgt_idx - ntps])
        elif src_idx < ntps and tgt_idx < ntps:
            return Edge.create(edge_idx, terminals[src_idx], terminals[tgt_idx])
        else:
            assert False, [src_idx, tgt_idx]

    @staticmethod
    def get_RSMT_GEO(terminals: List[Terminal], fst: int) -> SteinerTree:
        ter_list: List[Tuple[int, int]] = [
            (terminal.x.to_int(), terminal.y.to_int()) for terminal in terminals
        ]

        (stps_raw, edges_raw) = find_RSMT_GEO_(ter_list, fst)

        stps: List[SteinerPoint] = [
            SteinerPoint(i + len(terminals), fix(int(x)), fix(int(y)))
            for (i, (x, y)) in enumerate(stps_raw)
        ]

        edges = SteinerTree.new_edges(edges_raw, terminals, stps)

        return SteinerTree(terminals, stps, edges)

    @staticmethod
    def get_mst(terminals: List[Terminal], stps: List[SteinerPoint]) -> SteinerTree:
        """
        use the points to construct minimum spanning tree
        """
        # gather the points
        num_points = len(terminals) + len(stps)
        points: List[Tuple[int, int]] = []
        for terminal in terminals:
            query: Tuple[int, int] = (
                terminal.x.reduce(),
                terminal.y.reduce(),
            )
            points.append(query)
        for stp in stps:
            query: Tuple[int, int] = (stp.x.reduce(), stp.y.reduce())
            assert query not in points, query
            points.append(query)

        mst: List[Tuple[int, int]] = find_RMST_(points)
        assert len(mst) == num_points - 1, mst

        edges = SteinerTree.new_edges(mst, terminals, stps)

        return SteinerTree(terminals, stps, edges)

    def refine_subset_GEO(self, points_id: List[int], fst: int) -> None:
        # identify the interconnections within subset
        edges_id = self._find_inner_connections(points_id)

        # remove the conncections within subset
        for edge_id in edges_id:
            self.delete_edge(edge_id)

        # get RSMT result from geosteinr
        geo_points: List[Point] = [self.points[point_id] for point_id in points_id]
        ter_list: List[Tuple[int, int]] = [
            (point.x.reduce(), point.y.reduce()) for point in geo_points
        ]
        (stps_raw, edges_raw) = find_RSMT_GEO_(ter_list, fst)

        # create new stps and have a mapping between geo and points
        geo2point: List[int] = copy.deepcopy(points_id)
        for x_reduce, y_reduce in stps_raw:
            new_stp_id: int = len(self.points)
            new_stp: SteinerPoint = SteinerPoint(
                new_stp_id, fix.from_reduce(x_reduce), fix.from_reduce(y_reduce)
            )
            self.insert_point(new_stp)
            geo2point.append(new_stp_id)

        # create new edges
        for src_raw, tgt_raw in edges_raw:
            src_id: int = geo2point[src_raw]
            tgt_id: int = geo2point[tgt_raw]
            assert src_id != tgt_id, [src_id, tgt_id]
            new_edge: Edge = Edge.create(
                len(self.edges), self.get_point(src_id), self.get_point(tgt_id)
            )
            self.insert_edge(new_edge)
        return

    def refine_subsets_GEO_single(self, subsets: List[List[int]], fst: int) -> None:
        for subset in subsets:
            self.refine_subset_GEO(subset, fst)

    def refine_subsets_GEO_mp(self, subsets: List[List[int]], fst: int) -> None:
        geo_point_sets: List[List[Tuple[int, int]]] = []
        for subset in subsets:
            # identify the interconnections within subset
            edges_id = self._find_inner_connections(subset)

            # remove the conncections within subset
            for edge_id in edges_id:
                self.delete_edge(edge_id)

            # get RSMT result from geosteinr
            geo_points: List[Point] = [self.points[point_id] for point_id in subset]
            ter_list: List[Tuple[int, int]] = [
                (point.x.reduce(), point.y.reduce()) for point in geo_points
            ]
            geo_point_sets.append(ter_list)

        inputs: List[Tuple[List[Tuple[int, int]], int]] = [
            (geo_points, fst) for geo_points in geo_point_sets
        ]
        with mp.Pool() as pool:
            graphs_raw: List[
                Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]
            ] = pool.starmap(find_RSMT_GEO_, inputs)

        # create new stps and have a mapping between geo and points
        assert len(subsets) == len(graphs_raw), [len(subsets), len(graphs_raw)]
        for subset, graph_raw in zip(subsets, graphs_raw):
            (stps_raw, edges_raw) = graph_raw
            geo2point: List[int] = copy.deepcopy(subset)
            for x_reduce, y_reduce in stps_raw:
                new_stp_id: int = len(self.points)
                new_stp: SteinerPoint = SteinerPoint(
                    new_stp_id, fix.from_reduce(x_reduce), fix.from_reduce(y_reduce)
                )
                self.insert_point(new_stp)
                geo2point.append(new_stp_id)

            # create new edges
            for src_raw, tgt_raw in edges_raw:
                src_id: int = geo2point[src_raw]
                tgt_id: int = geo2point[tgt_raw]
                assert src_id != tgt_id, [src_id, tgt_id]
                new_edge: Edge = Edge.create(
                    len(self.edges), self.get_point(src_id), self.get_point(tgt_id)
                )
                self.insert_edge(new_edge)

    def refine_subsets_GEO(
        self, subsets: List[List[int]], fst: int, use_mp: bool = False
    ) -> None:
        if use_mp:
            self.refine_subsets_GEO_mp(subsets, fst)
        else:
            self.refine_subsets_GEO_single(subsets, fst)

    def _find_inner_connections(self, points_id: List[int]) -> List[int]:
        ret_set: Set[int] = set()
        for point_id in points_id:
            edges_id: List[int] = self.point2edges[point_id]
            for edge_id in edges_id:
                another: int = self.get_another_point(edge_id, point_id)
                # if the neighbor
                if another in points_id:
                    ret_set.add(edge_id)
        return list(ret_set)

    def post_process(self, k: int, fst: int) -> None:
        """
        do post processing of the steiner tree
        1. trim redundant steiner points
        2. rounding
        3. refine from leaves in the tree
        """
        self.trim()
        self.round()
        # self.refine_leaves(k, fst)
        self.trim()

    def trim(self) -> None:
        """
        1. trim degree-1 steiner points
        2. trim degree-2 steiner points
        """
        self.trim_d1()
        self.trim_d2()
        self.cleanup()

    def trim_d1(self) -> None:
        """
        trim degree-1 steiner points
        """
        while True:
            # identify one by one
            updated: bool = False
            redunt_stps: List[int] = []
            for stp in self.stps:
                assert stp.valid
                if len(self.point2edges[stp.id]) == 1:
                    updated = True
                    redunt_stps.append(stp.id)
            for redunt_stp_id in redunt_stps:
                self.delete_point(redunt_stp_id)
            if not updated:
                break

    def trim_d2(self) -> None:
        for stp in self.stps:
            assert stp.valid
            edges_id: List[int] = self.point2edges[stp.id]
            if len(edges_id) == 2:
                # get new connection
                src_id: int = self.get_another_point(edges_id[0], stp.id)
                tgt_id: int = self.get_another_point(edges_id[1], stp.id)
                new_edge: Edge = Edge.create(
                    len(self.edges), self.get_point(src_id), self.get_point(tgt_id)
                )
                self.insert_edge(new_edge)

                # delete point
                self.delete_point(stp.id)

    def round(self) -> None:
        # there may be overlapping points after rounding, so have to trim overlaps
        idx: Dict[Tuple[int, int], int] = {}
        for i, terminal in enumerate(self.terminals):
            idx[(terminal.x.to_int(), terminal.y.to_int())] = i

        for stp_id in range(self.num_terminals, len(self.points)):
            old_stp: Point = self.get_point(stp_id)
            query: Tuple[int, int] = (
                old_stp.x.round().to_int(),
                old_stp.y.round().to_int(),
            )
            if query in idx:
                overlap_point: int = idx[query]
                delete_edges: List[int] = self.delete_point(stp_id)
                # reconnect the edges of the deleted point
                for edge_id in delete_edges:
                    another: int = self.get_another_point(edge_id, stp_id)
                    if another != overlap_point:
                        self.insert_edge(
                            Edge.create(
                                len(self.edges),
                                self.get_point(another),
                                self.get_point(overlap_point),
                            )
                        )
            else:
                new_stp: SteinerPoint = SteinerPoint(
                    stp_id, old_stp.x.round(), old_stp.y.round()
                )
                self.points[stp_id] = new_stp
                idx[query] = stp_id
        self.cleanup()

    def refine_leaves(self, k: int, fst: int) -> None:
        """
        should apply cleanup before calling this method
        k is the size of the subset
        """
        if k == 1:
            return

        unused_points: Set[int] = set([i for i in range(len(self.points))])

        def bfs(root_point_id: int) -> List[int]:
            # also removes the points from unused_points
            ret: List[int] = []
            bfs_queue: queue.Queue = queue.Queue()
            bfs_queue.put(root_point_id)
            num_terminals: int = 0
            while (num_terminals < k) and (not bfs_queue.empty()):
                point_id: int = bfs_queue.get()
                ret.append(point_id)
                point: Point = self.get_point(point_id)
                if isinstance(point, Terminal):
                    num_terminals += 1
                unused_points.remove(point_id)
                for neighbor in self.get_neighbors(point_id):
                    if neighbor in unused_points:
                        bfs_queue.put(neighbor)
            return ret

        def is_leaf(point_id: int) -> bool:
            # consider point removal in the tree
            connection: int = 0
            for neighbor in self.get_neighbors(point_id):
                if neighbor in unused_points:
                    connection += 1
            return connection <= 1

        def is_inner_stp(point_id: int, subset: Set[int]) -> bool:
            point: Point = self.get_point(point_id)
            if isinstance(point, SteinerPoint):
                for edged_id in self.point2edges[point_id]:
                    if not self.get_another_point(edged_id, point_id) in subset:
                        return False
                return True
            else:
                return False

        # get subsets
        subsets: List[List[int]] = []
        while len(unused_points) > 0:
            for point_id in unused_points:
                if is_leaf(point_id):
                    subsets.append(bfs(point_id))
                    break
        assert len(unused_points) == 0, len(unused_points)

        # remove all steiner points in the subsets
        cleaned_subsets: List[List[int]] = []
        for subset in subsets:
            cleaned_subset: List[int] = []
            for point_id in subset:
                if is_inner_stp(point_id, set(subset)):
                    self.delete_point(point_id)
                else:
                    cleaned_subset.append(point_id)
            cleaned_subsets.append(cleaned_subset)

        self.refine_subsets_GEO(cleaned_subsets, fst)

        self.cleanup()

        return

    def check(self) -> None:
        self.check_no_cycle()
        self.check_trim()
        self.check_connection()

    def check_trim(self) -> None:
        for stp in self.stps:
            assert stp.valid
            assert len(self.point2edges[stp.id]) >= 3, len(self.point2edges[stp.id])

    def check_connection(self) -> None:
        shake_count: int = 0
        for edges in self.point2edges:
            shake_count += len(edges)
        assert (shake_count / 2) == len(self.edges), [shake_count, len(self.edges)]
        for point_id, point in enumerate(self.points):
            assert point_id == point.id, [point_id, point.id]
            for edge_id in self.point2edges[point_id]:
                self.get_another_point(edge_id, point_id)
        for edge_id, edge in enumerate(self.edges):
            assert edge_id == edge.id, f"Error with edge[{edge_id}].id == {edge.id}"
            self.get_another_point(edge_id, edge.src)
            self.get_another_point(edge_id, edge.tgt)

    def check_no_cycle(self) -> None:
        assert self.is_connected()
        assert len(self.points) == len(self.edges) + 1, [
            len(self.points),
            len(self.edges),
        ]

    def is_connected(self) -> bool:
        point_visited: List[bool] = [False for _ in range(len(self.points))]
        queue: List[int] = [0]  # start with any vertex in the graph
        while queue:
            vertex = queue.pop(0)
            assert point_visited[vertex] == False
            point_visited[vertex] = True
            for neighbor in self.get_neighbors(vertex):
                if not point_visited[neighbor]:
                    queue.append(neighbor)
        return all(point_visited)

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("steiner points\n")
            for stp in self.stps:
                f.write(f"{stp.x.to_int()} {stp.y.to_int()}\n")
            f.write("edges\n")
            for edge in self.edges:
                f.write(f"{edge.src} {edge.tgt}\n")

    @staticmethod
    def read(point_file: str, tree_file: str) -> SteinerTree:
        terminals: List[Terminal] = []

        # read points
        index: Set[Tuple[int, int]] = set()
        with open(point_file, "r") as pf:
            terminal_count: int = 0
            while line := pf.readline().split():
                assert len(line) == 2, line
                x: int = int(line[0])
                y: int = int(line[1])
                query: Tuple[int, int] = (x, y)
                if query not in index:
                    terminals.append(Terminal(terminal_count, fix(x), fix(y)))
                    terminal_count += 1
                    index.add(query)
        ret: SteinerTree = SteinerTree(terminals)

        # read tree
        with open(tree_file, "r") as tf:
            stp_str: str = tf.readline().replace("\n", "")
            assert stp_str == "steiner points", stp_str

            stp_count: int = 0
            while (stp_line := tf.readline().replace("\n", "")) != "edges":
                nums: List[str] = stp_line.split()
                assert len(nums) == 2, stp_line
                x: int = int(nums[0])
                y: int = int(nums[1])
                ret.insert_point(
                    SteinerPoint(stp_count + len(terminals), fix(x), fix(y))
                )
                stp_count += 1

            edge_count: int = 0
            while edge_line := tf.readline().split():
                assert len(edge_line) == 2
                src: int = int(edge_line[0])
                tgt: int = int(edge_line[1])
                ret.insert_edge(
                    Edge.create(edge_count, ret.get_point(src), ret.get_point(tgt))
                )
                edge_count += 1

        return ret
