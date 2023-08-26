from __future__ import annotations

POINT_NUM: int = 32
INT_OFFSET: int = 16
ONE = 1 << POINT_NUM


class fix:
    def __init__(self, num: int = 0) -> None:
        self.num: int = num << POINT_NUM

    def __eq__(self, other: fix) -> bool:
        return self.num == other.num

    def __ne__(self, other: fix) -> bool:
        return self.num != other.num

    def __lt__(self, other: fix) -> bool:
        return self.num < other.num

    def __gt__(self, other: fix) -> bool:
        return self.num > other.num

    def __le__(self, other: fix) -> bool:
        return self.num <= other.num

    def __ge__(self, other: fix) -> bool:
        return self.num >= other.num

    def __add__(self, other: fix):
        ret: fix = fix()
        ret.num = self.num + other.num
        return ret

    def __sub__(self, other: fix):
        assert self.num >= other.num
        ret: fix = fix()
        ret.num = self.num - other.num
        return ret

    def __mul__(self, other: int):
        assert isinstance(other, int), f"should get int type but get {type(other)}"
        ret: fix = fix()
        ret.num = self.num * other
        return ret

    def __floordiv__(self, other: int):
        assert isinstance(other, int), f"should get int type but get {type(other)}"
        ret: fix = fix()
        ret.num = self.num // other
        return ret

    def __str__(self) -> str:
        return str(self.to_float())

    def __repr__(self) -> str:
        return str(self)

    def to_float(self) -> float:
        return self.num / ONE

    def to_int(self) -> int:
        return int(self.num / ONE)

    def round(self) -> fix:
        ret: fix = fix()
        ret.num = (self.num >> POINT_NUM) << POINT_NUM
        if (self.num & (1 << (POINT_NUM - 1))) != 0:
            ret.num += ONE
        return ret

    def reduce(self) -> int:
        return self.num >> INT_OFFSET

    @staticmethod
    def from_reduce(reduced: int) -> fix:
        ret: fix = fix()
        ret.num = reduced << INT_OFFSET
        return ret


if __name__ == "__main__":
    a = fix(10)
    b = fix(20)
    print(a + b)
