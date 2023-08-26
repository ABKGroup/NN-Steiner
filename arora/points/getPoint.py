from __future__ import annotations

import multiprocessing as mp
import sys
from typing import List, Tuple

from arora.utils import get_files


def read_points(file_name: str) -> List[Tuple[int, int]]:
    ret: List[Tuple[int, int]] = []
    with open(file_name, "r") as f:
        while num_strs := f.readline().split():
            x: int = int(num_strs[0])
            y: int = int(num_strs[1])
            # avoid duplicated points
            query: Tuple[int, int] = (x, y)
            if query not in ret:
                ret.append(query)
    return ret


def read_points_dir(dir_name: str) -> List[List[Tuple[int, int]]]:
    point_files: List[str] = get_files(dir_name)
    with mp.Pool() as pool:
        ret: List[List[Tuple[int, int]]] = pool.map(read_points, point_files)
    return ret


if __name__ == "__main__":
    print(read_points(sys.argv[1]))


def write_points(points: List[Tuple[int, int]], file_name: str) -> None:
    with open(file_name, "w") as wfile:
        for point in points:
            point_str = f"{point[0]} {point[1]}\n"
            wfile.write(point_str)
