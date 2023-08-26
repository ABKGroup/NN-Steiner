from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from .base import NNbase
from .dp import NNdp
from .retrieve import NNretrieve
from .top import NNtop


@dataclass
class ModelSize:
    input_size: int
    output_size: int


@dataclass
class ModelConf:
    base: ModelSize
    dp: ModelSize
    top: ModelSize
    retrieve: ModelSize


class NNArora(nn.Module):
    def __init__(
        self,
        model_args: Dict,
        qt_args: Dict,
    ) -> None:
        super().__init__()
        # parameters
        model_conf = NNArora.get_model_conf(model_args, qt_args)
        hidden_size: int = model_args["hidden_size"]
        dropout: float = model_args["dropout"]

        # models
        model_type: str = model_args["type"]
        if model_args["type"] == "mlp":
            self.init_mlp(model_conf, hidden_size, dropout)
        else:
            assert False, f"invalid model type: {model_type}"

    def init_mlp(self, model_conf: ModelConf, hidden_size: int, dropout: float):
        self.nn_base = NNbase(
            model_conf.base.input_size,
            hidden_size,
            model_conf.base.output_size,
            dropout,
        )
        self.nn_dp = NNdp(
            model_conf.dp.input_size,
            hidden_size,
            model_conf.dp.output_size,
            dropout,
        )
        self.nn_top = NNtop(
            model_conf.top.input_size,
            hidden_size,
            model_conf.top.output_size,
            dropout,
        )
        self.nn_retrieve = NNretrieve(
            model_conf.retrieve.input_size,
            hidden_size,
            model_conf.retrieve.output_size,
            dropout,
        )

    @staticmethod
    def get_model_conf(model_args: Dict, quadtree_args: Dict) -> ModelConf:
        # TODO: set model size
        model_type: str = model_args["type"]
        if model_args["type"] == "mlp":
            m: int = quadtree_args["m"]
            kb: int = quadtree_args["kb"]
            n_side_portal: int = m + 2
            n_cell_portal: int = n_side_portal * 4
            emb_size: int = model_args["emb_size"]
            dp_size: int = emb_size * n_cell_portal

            # base: [terminals, boundary, boundary]
            base = ModelSize(kb * 2 + n_cell_portal, dp_size)
            # dp: [dp_vec * 4, boundary]
            dp = ModelSize(dp_size * 4 + n_cell_portal, dp_size)
            # top: [dp_vec * 4, top_vc]
            top = ModelSize(dp_size * 5, n_cell_portal)
            # retrieve: [dp_vec * 4, boundary]
            retrieve = ModelSize(dp_size * 4 + n_cell_portal, n_cell_portal)

            return ModelConf(base, dp, top, retrieve)
        else:
            assert False, f"invalid model type: {model_type}"

    def forward(
        self,
        tree_struct: Tensor,
        cell2bound: Tensor,
        cell2cross: Tensor,
        bound_tens: Tensor,
        terminal_tens: Tensor,
        null_portal_tens: Tensor,
    ) -> Tensor:
        # check tensor
        self.init_tensor(
            tree_struct,
            cell2bound,
            cell2cross,
            bound_tens,
            terminal_tens,
            null_portal_tens,
        )

        # bottom up
        forward_embs: Tensor = self._bottom_up()

        # top down
        self._top_down(forward_embs)

        # return the predicted portals
        ret_list: List[Tensor] = []

        for i in range(len(self.portal_probs)):
            assert i in self.portal_probs
            ret_list.append(self.portal_probs[i])

        ret: Tensor = torch.cat(ret_list, dim=1)
        return ret

    def init_tensor(
        self,
        tree_struct: Tensor,
        cell2bound: Tensor,
        cell2cross: Tensor,
        bound_tens: Tensor,
        terminal_tens: Tensor,
        null_portal_tens: Tensor,
    ):
        """
        tree_struct: Tensor
            shape: (n_cell, 4)
        cell2bound: Tensor
            shape: (n_cell, n_cell_p)
        cell2cross: Tensor
            shape: (n_cell, n_cell_p)
        bound_tens: Tensor
            shape: (batch, n_cell, n_cell_p)
        terminal_tens: Tensor
            shape: (batch, n_cell, kb * 2)
        null_portal_tens: Tensor
            shape: (batch, 1)
        """
        # check batch size
        assert len(bound_tens) == len(terminal_tens) == len(null_portal_tens), [
            len(bound_tens),
            len(terminal_tens),
            len(null_portal_tens),
        ]

        # check n_cell
        assert (
            len(tree_struct)
            == len(cell2bound)
            == len(cell2cross)
            == bound_tens.shape[1]
            == terminal_tens.shape[1]
        ), [
            len(tree_struct),
            len(cell2bound),
            len(cell2cross),
            bound_tens.shape[1],
            terminal_tens.shape[1],
        ]
        self.n_cell: int = len(tree_struct)

        # check n_cell_p
        assert cell2bound.shape[-1] == cell2cross.shape[-1] == bound_tens.shape[-1], [
            cell2bound.shape[-1],
            cell2cross.shape[-1],
            bound_tens.shape[-1],
        ]
        self.n_cell_p: int = cell2bound.shape[-1]

        # assignment
        # structure
        self.tree_struct: Tensor = tree_struct
        self.cell2bound: Tensor = cell2bound
        self.cell2cross: Tensor = cell2cross

        # input
        self.bound_tens: Tensor = bound_tens
        self.terminal_tens: Tensor = terminal_tens
        self.null_portal_tens: Tensor = null_portal_tens

        # intermediate: shape: (batch, n_child_vec)
        self.child_tens: Dict[int, Tensor] = {}

        # output
        self.portal_probs: Dict[int, Tensor] = {}

    def _bottom_up(self) -> Tensor:
        # recursively apply NNs to the corresponding cell
        return self._apply_cell_forward(0)

    def _top_down(self, forward_embs: Tensor) -> None:
        # NNtop
        probs_top: Tensor = self._apply_top(forward_embs)
        self._update_portal_probs(0, probs_top)

        for child_id in self.tree_struct[0]:
            self._apply_cell_backward(int(child_id))

    def _apply_cell_forward(self, cell_id: int) -> Tensor:
        """
        return the output of the cell NN
        """
        children: Tensor = self.tree_struct[cell_id]
        assert len(children) == 4, children

        # NNbase
        if torch.all(children == -1):
            return self._apply_base(cell_id)
        # NNdp
        else:
            child_cell_embs: List[Tensor] = []
            for child_id in children:
                child_cell_embs.append(self._apply_cell_forward(int(child_id)))
            return self._apply_dp(cell_id, child_cell_embs)

    def _apply_cell_backward(self, cell_id: int) -> None:
        children: Tensor = self.tree_struct[cell_id]
        # leaf cell
        if torch.all(children == -1):
            return

        cell_probs: Tensor = self._apply_retrieve(cell_id)
        assert cell_probs.shape[-1] == self.n_cell_p, cell_probs.shape
        self._update_portal_probs(cell_id, cell_probs)

        for child_id in children:
            self._apply_cell_backward(int(child_id))

    def _apply_base(self, cell_id: int) -> Tensor:
        # decide shape
        bound_tens: Tensor = self.bound_tens[:, cell_id]
        terminal_tens: Tensor = self.terminal_tens[:, cell_id]

        base_in: Tensor = torch.cat([bound_tens, terminal_tens], dim=1)

        return self.nn_base(base_in)

    def _apply_dp(self, cell_id: int, child_cell_embs: List[Tensor]) -> Tensor:
        assert len(child_cell_embs) == 4

        # decide shape
        child_tens: Tensor = torch.cat(child_cell_embs, dim=1)
        bound_tens: Tensor = self.bound_tens[:, cell_id]

        # save child_tens
        assert cell_id not in self.child_tens
        self.child_tens[cell_id] = child_tens

        dp_in: Tensor = torch.cat([child_tens, bound_tens], dim=1)

        return self.nn_dp(dp_in)

    def _apply_top(self, forward_embs: Tensor) -> Tensor:
        # decide shape
        assert 0 in self.child_tens
        child_tens: Tensor = self.child_tens[0]

        top_in: Tensor = torch.cat([child_tens, forward_embs], dim=1)

        return self.nn_top(top_in)

    def _apply_retrieve(self, cell_id: int) -> Tensor:
        children: Tensor = self.tree_struct[cell_id]
        # leaf cell
        assert torch.all(children != -1), children

        # decide shape
        assert cell_id in self.child_tens
        child_tens: Tensor = self.child_tens[cell_id]
        bound_tens: Tensor = self._get_bound_tensor(cell_id)

        retrieve_in: Tensor = torch.cat([child_tens, bound_tens], dim=1)

        return self.nn_retrieve(retrieve_in)

    def _get_bound_tensor(self, cell_id: int) -> Tensor:
        ret_list: List[Tensor] = []
        portals_id: Tensor = self.cell2bound[cell_id]
        for portal_id in portals_id:
            p_id: int = int(portal_id)
            if p_id == -1:
                ret_list.append(self.null_portal_tens)
            else:
                prob: Tensor | None = self.portal_probs[p_id]
                assert prob is not None
                ret_list.append(prob)

        assert len(ret_list) == self.n_cell_p
        ret: Tensor = torch.cat(ret_list, dim=1)
        return ret

    def _update_portal_probs(self, cell_id: int, cell_probs: Tensor) -> None:
        """
        update the portal probs of cross
        cell_probs.shape: (batch, 4 * num_side_portal)
        """
        assert cell_probs.shape[-1] == self.n_cell_p

        portals_id: Tensor = self.cell2cross[cell_id]

        for i, portal_id in enumerate(portals_id):
            p_id: int = int(portal_id)
            assert p_id >= 0, p_id
            self.portal_probs[p_id] = cell_probs[:, i].reshape(shape=(-1, 1))
