from __future__ import annotations

import copy
import multiprocessing as mp
import os
from typing import Dict, Iterable, List

import torch
import torch.distributed as dist
from torch import Tensor, nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from arora.data import DataLoader
from arora.utils import get_f1


def load_loop(file: str) -> Dict[str, Tensor]:
    assert isinstance(file, str), file
    assert len(file) > 0, file
    with open(file, "rb") as f:
        ret: Dict[str, Tensor] = torch.load(f)
    return ret


class TrainerProcess:
    def __init__(
        self,
        rank: int,
        world_size: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args: Dict,
    ) -> None:
        # parameters
        self.rank: int = rank
        self.world_size: int = world_size
        self.epochs: int = args["epochs"]
        self.eval: int = args["eval"]
        self.max_no_update: int = args["max_no_update"]
        self.val_save: bool = args["val_save"]
        self.portal_weight: int = args["portal_weight"]
        self.best_acc: float = 0.0
        self.no_update: int = 0

        # setup
        self.setup(rank, self.world_size)
        model = model.to(rank)
        self.ddp_model: DDP = DDP(model, device_ids=[rank])
        self.writer: SummaryWriter = SummaryWriter("train/")
        self.optimizer: optim.Optimizer = optim.Adam(
            self.ddp_model.parameters(), lr=args["lr"]
        )

        # get_data
        if self.rank == 0:
            print("loading local training data...")
        train_data: List[Dict[str, Tensor]] = train_loader.load_data(rank, world_size)
        dist.barrier()
        if self.rank == 0:
            print("loading local valildation data...")
        val_data: List[Dict[str, Tensor]] = val_loader.load_data(rank, world_size)
        dist.barrier()

        if self.rank == 0:
            print("start trainig")
        # first evaluation
        self.validation_step(val_data, 0)
        # iteration
        iter = tqdm(range(self.epochs)) if self.rank == 0 else range(self.epochs)
        if args["profile"]:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(".log/"),
                with_stack=True,
            ) as prof:
                self.training_iter(train_data, val_data, iter, prof)
            if self.rank == 0:
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        else:
            self.training_iter(train_data, val_data, iter)

        self.cleanup()

    def training_iter(
        self,
        train_data: List[Dict[str, Tensor]],
        val_data: List[Dict[str, Tensor]],
        iter: Iterable,
        prof: torch.profiler.profile | None = None,
    ):
        for epoch in iter:
            self.training_step(train_data, epoch)
            if epoch % self.eval == (self.eval - 1):
                val_log: Dict = self.validation_step(val_data, epoch + 1)
                self.checkpoint(val_log)
                if self.earlystop():
                    break
            if prof is not None:
                prof.step()

    def setup(self, rank: int, world_size: int):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def load_data(self, files: List[List[str]]) -> List[Dict[str, Tensor]]:
        local_files: List[str] = files[self.rank]
        ret: List[Dict[str, Tensor]] = []

        with mp.Pool() as pool:
            ret: List[Dict[str, Tensor]] = pool.map(load_loop, tqdm(local_files))

        return ret

    def training_step(
        self,
        batches: List[Dict[str, Tensor]],
        epoch: int,
    ) -> Dict:
        accs: List[float] = []
        losses: List[float] = []
        for batch in batches:
            # get inputs
            inputs: Dict[str, Tensor] = copy.copy(batch)
            for key, val in inputs.items():
                inputs[key] = val.to(self.rank)  # pyright: ignore
            golden_tens: Tensor = copy.copy(inputs["golden"])
            del inputs["golden"]

            # forward
            self.ddp_model.train()
            predict_tens: Tensor = self.ddp_model(**inputs)

            assert torch.all(0 <= predict_tens) and torch.all(
                predict_tens <= 1
            ), predict_tens
            assert torch.all(0 <= golden_tens) and torch.all(
                golden_tens <= 1
            ), golden_tens

            # update
            loss: Tensor = nn.functional.binary_cross_entropy(
                input=predict_tens,
                target=golden_tens,
                weight=golden_tens * self.portal_weight + 1,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get metrics
            accs.append(get_f1(predict_tens, golden_tens))
            losses.append(float(loss))

        reduced_acc: float = self.all_reduce(sum(accs), len(accs))
        reduced_loss: float = self.all_reduce(sum(losses), len(losses))
        assert reduced_acc <= 1, reduced_acc
        if self.rank == 0:
            self.writer.add_scalar("train/f1", reduced_acc, epoch)
            self.writer.add_scalar("train/loss", reduced_loss, epoch)

        log: Dict = {
            "acc": reduced_acc,
            "loss": reduced_loss,
        }
        return log

    @torch.no_grad()
    def validation_step(
        self,
        batches: List[Dict[str, Tensor]],
        epoch: int,
    ) -> Dict:
        if self.rank == 0:
            if self.val_save:
                if not os.path.exists("train/model"):
                    os.mkdir("train/model")
                torch.save(self.ddp_model.state_dict(), "train/model/nnArora_best.pt")

        accs: List[float] = []
        for batch in batches:
            # get inputs
            inputs: Dict[str, Tensor] = copy.copy(batch)
            for key, val in inputs.items():
                inputs[key] = val.to(self.rank)  # pyright: ignore
            golden_tens: Tensor = copy.copy(inputs["golden"])
            del inputs["golden"]

            # forward
            self.ddp_model.eval()
            predict_tens: Tensor = self.ddp_model(**inputs)

            assert torch.all(0 <= predict_tens) and torch.all(
                predict_tens <= 1
            ), predict_tens
            assert torch.all(0 <= golden_tens) and torch.all(
                predict_tens <= 1
            ), golden_tens

            # get metrics
            accs.append(get_f1(predict_tens, golden_tens))

        reduced_acc: float = self.all_reduce(sum(accs), len(accs))
        assert reduced_acc <= 1, reduced_acc
        if self.rank == 0:
            self.writer.add_scalar("val/f1", reduced_acc, epoch)

        log: Dict = {"acc": reduced_acc}
        return log

    def checkpoint(self, val_log: Dict) -> None:
        if self.rank == 0:
            if not os.path.exists("train/model"):
                os.mkdir("train/model")

            acc: float = val_log["acc"]
            if acc > self.best_acc:
                self.best_acc = acc
                torch.save(self.ddp_model.state_dict(), "train/model/nnArora.pt")
                print(f"model saved with acc: {acc}")
                self.no_update = 0
            else:
                self.no_update += 1

    def earlystop(self) -> bool:
        if self.rank == 0:
            send_tens: Tensor = torch.tensor(self.no_update >= self.max_no_update).to(
                self.rank  # pyright: ignore
            )
            dist.broadcast(send_tens, 0)
            return bool(send_tens)
        else:
            rec_tens: Tensor = torch.zeros(size=(), dtype=torch.bool).to(
                self.rank  # pyright: ignore
            )
            dist.broadcast(rec_tens, 0)
            return bool(rec_tens)

    def all_reduce(self, scalar: int | float, weight: int | float) -> float:
        total: Tensor = torch.tensor([scalar, weight], dtype=torch.float32)
        total = total.to(self.rank)  # pyright: ignore
        dist.all_reduce(total)
        scalar_sum, weight_sum = total.tolist()
        return scalar_sum / weight_sum
