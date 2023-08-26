from __future__ import annotations
from typing import List, Tuple
import sys

def read_points(file: str) -> List[Tuple[int, int]]:
    ret: List[Tuple[int, int]] = []
    with open(file, "r") as f:
        while(line:= f.readline().split()):
            assert len(line) == 2, line
            x: int = int(line[0])
            y: int = int(line[1])
            query: Tuple[int, int] = (x, y)
            if query not in ret:
                ret.append((x, y))
    return ret

def read_result(file: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    stps: List[Tuple[int, int]] = []
    edges: List[Tuple[int, int]] = []
    with open(file, "r") as f:
        # steiner points
        stp_str: str = f.readline().replace("\n", "")
        assert stp_str == "steiner points", stp_str
        while (stp_line:= f.readline().replace("\n", "")) != "edges":
            nums: List[str] = stp_line.split()
            assert len(nums) == 2, stp_line
            x: int = int(nums[0])
            y: int = int(nums[1])
            stps.append((x, y))
        while (edge_line:= f.readline().split()):
            assert len(edge_line) == 2
            src: int = int(edge_line[0])
            tgt: int = int(edge_line[1])
            edges.append((src, tgt))
    return (stps, edges)

def is_connected(num_points: int, edges: List[Tuple[int, int]]) -> bool:
    adj_mat: List[List[int]] = [[] for _ in range(num_points)]
    for edge in edges:
         (src, tgt) = edge
         adj_mat[src].append(tgt)
         adj_mat[tgt].append(src)

    visited: List[bool] = [False for _ in range(num_points)]
    def dfs(idx: int) -> None:
        visited[idx] = True
        adj: List[int] = adj_mat[idx]
        for neighbor in adj:
            if not visited[neighbor]:
                dfs(neighbor)

    dfs(0)

    return all(visited)    

def L1_dist(src: Tuple[int, int], tgt: Tuple[int, int]) -> int:
    src_x, src_y = src
    tgt_x, tgt_y = tgt
    return abs(src_x - tgt_x) + abs(src_y - tgt_y)

def get_length(points: List[Tuple[int, int]], edges: List[Tuple[int, int]]) -> int:
    ret: int = 0
    for edge in edges:
        src_id: int = edge[0]
        tgt_id: int = edge[1]
        src: Tuple[int, int] = points[src_id]
        tgt: Tuple[int, int] = points[tgt_id]
        ret += L1_dist(src, tgt)
    return ret

def main() -> None:
    points_file: str = sys.argv[1]
    result_file: str = sys.argv[2]
    terminals: List[Tuple[int, int]] = read_points(points_file)
    (stps, edges) = read_result(result_file)

    points: List[Tuple[int, int]] = terminals + stps

    # all the points are connected
    assert is_connected(len(points),edges)

    # no cycle
    assert len(points) == len(edges) + 1, [len(points), len(edges)]

    length: int = get_length(points, edges)
    print(f"check passed with wirelenth: {length}")

if __name__ == "__main__":
    main()