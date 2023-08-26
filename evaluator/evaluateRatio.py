from __future__ import annotations
import numpy as np
from typing import List
import sys

def read_lengths(file: str) -> np.ndarray:
    ret_list: List[float] = []
    with open(file, "r") as f:
        while(line := f.readline().replace("\n", "")):
            num_str: str = line
            ret_list.append(float(num_str))
    return np.array(ret_list)

def main() -> None:
    comp_file: str = sys.argv[1]
    base_file: str = sys.argv[2]

    comp_arr: np.ndarray = read_lengths(comp_file)
    base_arr: np.ndarray = read_lengths(base_file)

    assert comp_arr.shape == base_arr.shape, [comp_arr.shape, base_arr.shape]
    percentage: float = ((comp_arr / base_arr).mean() - 1) * 100
    print(f"average error percentage of {len(comp_arr)} cases: {round(percentage, 4)}")

if __name__ == "__main__":
    main()