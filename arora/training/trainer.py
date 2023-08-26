from __future__ import annotations

import copy
import os
from typing import Dict

import torch
import torch.multiprocessing as mp
from torch import Tensor, nn

from arora.data import DataLoader
from arora.utils import get_f1

from .trainerProcess import TrainerProcess


class Trainer:
    def __init__(
        self,
        args: Dict,
    ) -> None:
        super().__init__()
        self.args: Dict = args

        if not os.path.exists("train"):
            os.mkdir("train")

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        # setup
        print("setting up")
        n_gpus: int = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size: int = n_gpus

        print("finish setting up")

        # multiprocessing
        mp.spawn(  # pyright: ignore
            self.worker,
            args=(world_size, model, train_loader, val_loader, self.args),
            nprocs=world_size,
            join=True,
        )

    def worker(
        self,
        rank: int,
        world_size: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: Dict,
    ) -> None:
        if rank == 0:
            print("start multiprocessing")
        TrainerProcess(rank, world_size, model, train_loader, val_loader, args)

    def test(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        test_batch: Dict[str, Tensor] = test_loader.load_batch(0)
        # get inputs
        inputs: Dict[str, Tensor] = copy.copy(test_batch)
        for key, val in inputs.items():
            inputs[key] = val.to("cuda:0")  # pyright: ignore
        golden_tens: Tensor = copy.copy(inputs["golden"])
        del inputs["golden"]

        # forward
        model.eval()
        model = model.to("cuda:0")
        predict_tens: Tensor = model(**inputs)

        assert torch.all(0 <= predict_tens) and torch.all(
            predict_tens <= 1
        ), predict_tens
        assert torch.all(0 <= golden_tens) and torch.all(predict_tens <= 1), golden_tens

        # get metrics
        acc: float = get_f1(predict_tens, golden_tens)

        assert acc <= 1, acc

        log: Dict = {"acc": acc}
        return log
