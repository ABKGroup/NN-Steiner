from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from arora.fix import fix

POINT_SIZE: float = 5


class Point(ABC):
    def __init__(self, id: int, x: fix, y: fix) -> None:
        self.id: int = id
        self.x: fix = copy.deepcopy(x)
        self.y: fix = copy.deepcopy(y)

    @property
    @abstractmethod
    def typ(self) -> str:
        raise NotImplemented

    @staticmethod
    def L1_dist(point1: Point, point2: Point) -> float:
        return abs(point1.x.to_float() - point2.x.to_float()) + abs(
            point1.y.to_float() - point2.y.to_float()
        )

    @staticmethod
    def L2_dist(point1: Point, point2: Point) -> float:
        return (point1.x.to_float() - point2.x.to_float()) ** 2 + (
            point1.y.to_float() - point2.y.to_float()
        ) ** 2

    @staticmethod
    def comp(point1: Point, point2: Point) -> int:
        if point1.x == point2.x:
            if point1.y > point2.y:
                return 1
            elif point1.y < point2.y:
                return -1
            else:
                return 0
        elif point1.y == point2.y:
            if point1.x > point2.x:
                return 1
            elif point1.x < point2.x:
                return -1
            else:
                return 0
        else:
            assert (
                False
            ), f"point: ({point1.x}, {point1.y} and point: ({point2.x}, {point2.y})) have mismatch"

    @staticmethod
    def is_lshape(point1: Point, point2: Point):
        return point1.x != point2.x and point1.y != point2.y

    def shift(self, dis: Tuple[fix, fix]) -> None:
        (x, y) = dis
        self.x = self.x - x
        self.y = self.y - y


class PurePoint(Point):
    def __init__(self, id: int, x: fix, y: fix) -> None:
        super().__init__(id, x, y)

    @property
    def typ(self) -> str:
        return "pure point"


class Terminal(Point):
    def __init__(self, id: int, x: fix, y: fix) -> None:
        super().__init__(id, x, y)

    @property
    def typ(self) -> str:
        return "terminal"

    @staticmethod
    def from_point(id: int, point: Tuple[int, int]) -> Terminal:
        (x, y) = point
        return Terminal(id, fix(x), fix(y))

    @staticmethod
    def terminals_from_point(points: List[Tuple[int, int]]) -> List[Terminal]:
        return [Terminal.from_point(i, point) for (i, point) in enumerate(points)]


class Rect:
    def __init__(self, x_min: fix, y_min: fix, x_max: fix, y_max: fix) -> None:
        self.x_min: fix = copy.deepcopy(x_min)
        self.y_min: fix = copy.deepcopy(y_min)
        self.x_max: fix = copy.deepcopy(x_max)
        self.y_max: fix = copy.deepcopy(y_max)

    @property
    def x_cent(self) -> fix:
        return (self.x_min + self.x_max) // 2

    @property
    def y_cent(self) -> fix:
        return (self.y_min + self.y_max) // 2

    @property
    def width(self) -> fix:
        assert self.x_max > self.x_min, f"x_max: {self.x_max} <= x_min: {self.x_min}"
        return self.x_max - self.x_min

    @property
    def height(self) -> fix:
        assert self.y_max > self.y_min, f"y_max: {self.y_max} <= y_min: {self.y_min}"
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width.to_float() * self.height.to_float()

    def contains(self, point: Point) -> bool:
        return self.x_min <= point.x < self.x_max and self.y_min <= point.y < self.y_max

    def shift(self, dis: Tuple[fix, fix]) -> None:
        (x, y) = dis
        self.x_min = self.x_min - x
        self.x_max = self.x_max - x
        self.y_min = self.y_min - y
        self.y_max = self.y_max - y

    @staticmethod
    def from_terminals(terminals: List[Terminal]) -> Rect:
        x_max: fix = fix(0)
        y_max: fix = fix(0)
        for terminal in terminals:
            if terminal.x > x_max:
                x_max = terminal.x
            if terminal.y > y_max:
                y_max = terminal.y
        return Rect(fix(0), fix(0), x_max + fix(1), y_max + fix(1))

    def to_nparray(self) -> np.ndarray:
        return np.array(
            [
                self.x_min.to_float(),
                self.y_min.to_float(),
                self.x_max.to_float(),
                self.y_max.to_float(),
            ]
        )
