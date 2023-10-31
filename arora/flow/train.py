from __future__ import annotations

import logging
import os
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor

from arora.data import DataLoader, QuadTreeDataset, collate_fn
from arora.models import NNArora, init_xavier
from arora.training import Trainer
from arora.utils import get_files


def train(args: DictConfig) -> None:
    logger = logging.getLogger("train")

    logger.info("training start")
    # seed
    torch.manual_seed(args["train"]["seed"])

    # mkdir
    if not os.path.exists("train"):
        os.mkdir("train")

    # get training data (load data, cluter, put into data loader)
    train_dir = os.path.join(hydra.utils.get_original_cwd(), args["train"]["train_set"])
    train_files = get_files(train_dir, True)
    train_set: QuadTreeDataset = QuadTreeDataset(train_files)
    train_loader: DataLoader = DataLoader(
        train_set, collate_fn, args["train"]["batch_size"]
    )

    # get validation data
    val_dir = os.path.join(hydra.utils.get_original_cwd(), args["train"]["val_set"])
    val_files = get_files(val_dir, True)
    val_set: QuadTreeDataset = QuadTreeDataset(val_files)
    val_loader: DataLoader = DataLoader(
        val_set, collate_fn, args["train"]["val_batch_size"]
    )

    # get test data
    test_dir = os.path.join(hydra.utils.get_original_cwd(), args["train"]["test_set"])
    test_files = get_files(test_dir, True)
    test_set: QuadTreeDataset = QuadTreeDataset(test_files)
    test_loader: DataLoader = DataLoader(
        test_set, collate_fn, args["train"]["test_batch_size"]
    )

    # training
    # TODO: set training args
    nn_arora: NNArora = NNArora(args["model"], args["quadtree"])
    train_args: Dict = dict(args["train"])
    if train_args["checkpoint"] is not None:
        ckpt_path: str = os.path.join(
            hydra.utils.get_original_cwd(), train_args["checkpoint"]
        )
        state_dict: Dict[str, Tensor] = torch.load(ckpt_path)
        state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }
        nn_arora.load_state_dict(state_dict)
    else:
        nn_arora.apply(init_xavier)
    trainer: Trainer = Trainer(train_args)
    trainer.fit(nn_arora, train_loader, val_loader)

    # testing
    # load trained model
    trained_nn_arora: NNArora = NNArora(args["model"], args["quadtree"])
    model_path: str = "train/model/nnArora_best.pt"
    state_dict: Dict[str, Tensor] = torch.load(model_path)
    state_dict = {
        key.replace("module.", ""): value for key, value in state_dict.items()
    }
    trained_nn_arora.load_state_dict(state_dict)

    # test
    test_dict: Dict = trainer.test(trained_nn_arora, test_loader)
    test_acc: float = test_dict["acc"]
    logger.info(f"test acc: {test_acc}")
