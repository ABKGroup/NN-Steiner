from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor


def get_acc(predict_tens: Tensor, golden_tens: Tensor) -> float:
    correct = torch.sum(golden_tens == predict_tens.round())
    acc = float(correct / torch.numel(golden_tens))
    assert acc <= 1
    return acc


def get_true_positive(predict_tens: Tensor, golden_tens: Tensor) -> int:
    return int(torch.sum(predict_tens * golden_tens))


def get_false_positive(predict_tens: Tensor, golden_tens: Tensor) -> int:
    return int(torch.sum(predict_tens * (~golden_tens)))


def get_false_negative(predict_tens: Tensor, golden_tens: Tensor) -> int:
    return int(torch.sum((~predict_tens) * golden_tens))


def get_f1(predict_tens: Tensor, golden_tens: Tensor) -> float:
    predict: Tensor = torch.round(predict_tens).to(dtype=torch.bool)
    golden: Tensor = golden_tens.to(dtype=torch.bool)

    true_positive: int = get_true_positive(predict, golden)
    false_positive: int = get_false_positive(predict, golden)
    false_negative: int = get_false_negative(predict, golden)
    return (2 * true_positive) / (2 * true_positive + false_positive + false_negative)


def get_dtype(precision: int) -> torch.dtype:
    if precision == 32:
        return torch.float32
    if precision == 64:
        return torch.float64
    else:
        assert False, f"invalid precision: {precision}"


def plot_conf_matrix(golden: Tensor, predict: Tensor, path: str) -> None:
    # Generate the nparray
    golden_tens: Tensor = golden.to("cpu")
    predict_tens: Tensor = predict.to("cpu")
    golden_arr: np.ndarray = np.array(golden_tens)
    predicted_arr: np.ndarray = np.array(predict_tens).round()

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(golden_arr, predicted_arr)
    conf_matrix = conf_matrix / conf_matrix.sum()

    # Visualize the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    plt.title("Confusion Matrix")
    plt.imshow(conf_matrix, cmap="Blues")
    plt.xticks([0, 1], ["Predicted Not Taken", "Predicted Taken"])
    plt.yticks([0, 1], ["True Not Taken", "True Taken"])
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{conf_matrix[i, j] * 100}%",
                ha="center",
                va="center",
                color="black",
            )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig(f"{path}/conf_mat.png")
    plt.clf()


def get_files(dir: str) -> List[str]:
    files: List[str] = os.listdir(dir)
    files = sorted(files)
    files = [os.path.join(dir, file) for file in files]

    return files


def points_int2float(points: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
    x_list: List[int] = [point[0] for point in points]
    y_list: List[int] = [point[1] for point in points]
    x_max: int = max(x_list)
    y_max: int = max(y_list)
    base: int = max(x_max, y_max) + 1
    ret: List[Tuple[float, float]] = [
        (point[0] / base, point[1] / base) for point in points
    ]
    return ret


def L1_dist(src: Tuple[float, float], tgt: Tuple[float, float]) -> float:
    src_x, src_y = src
    tgt_x, tgt_y = tgt
    return abs(src_x - tgt_x) + abs(src_y - tgt_y)


def avg(nums: List[float]) -> float:
    return sum(nums) / len(nums)
