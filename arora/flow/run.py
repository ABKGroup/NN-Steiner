from __future__ import annotations

from omegaconf import DictConfig

from .dataGen import data_gen
from .eval import eval
from .geoExp import geo_exp
from .mstExp import mst_exp
from .nnExp import nn_exp
from .plot import plot
from .refine import refine
from .snappedExp import snapped_exp
from .solve import solve
from .train import train


def run(args: DictConfig) -> None:
    if "data_gen" in args["flow"]:
        data_gen(args)
    if "train" in args["flow"]:
        train(args)
    if "eval" in args["flow"]:
        eval(args)
    if "plot" in args["flow"]:
        plot(args)
    if "solve" in args["flow"]:
        solve(args)
    if "geo_exp" in args["flow"]:
        geo_exp(args)
    if "snapped_exp" in args["flow"]:
        snapped_exp(args)
    if "nn_exp" in args["flow"]:
        nn_exp(args)
    if "mst_exp" in args["flow"]:
        mst_exp(args)
    if "refine" in args["flow"]:
        refine(args)
