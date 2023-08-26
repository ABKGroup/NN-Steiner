from __future__ import annotations

from typing import Dict

from .pointGen import PointGen
from .pointGenMixNormal import PointGenMixNormal
from .pointGenNonIsotropicNormal import PointGenNonIsotropicNormal
from .pointGenNormal import PointGenNormal
from .pointGenUniform import PointGenUniform


def get_pointGen(dist: str, args: Dict, seed: int) -> PointGen:
    if dist == "uniform":
        return PointGenUniform(args["x_range"], args["y_range"], seed)
    elif dist == "normal":
        return PointGenNormal(
            args["x_range"], args["y_range"], args["x_stdev"], args["y_stdev"], seed
        )
    elif dist == "non-isotropic":
        return PointGenNonIsotropicNormal(
            args["x_range"], args["y_range"], args["x_stdev"], args["y_stdev"], seed
        )
    elif dist == "mix-normal":
        return PointGenMixNormal(
            args["x_range"], args["y_range"], args["x_stdev"], args["y_stdev"], seed
        )
    else:
        assert False, f"invalid point_gen distribution: {dist}"
