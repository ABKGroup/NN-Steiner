from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor


def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    batch:
        tree_struct: Tensor
            shape: (n_cell, 4)
        cell2bound: Tensor
            shape: (n_cell, n_cell_p)
        cell2cross: Tensor
            shape: (n_cell, n_cell_p)
        bound_tens: Tensor
            shape: (n_cell, n_cell_p)
        terminal_tens: Tensor
            shape: (n_cell, kb * 2)
        null_portal_tens: Tensor
            shape: (1)
        golden: Tensor
            shape: (n_cell_p)
    return:
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
        golden: Tensor
            shape: (batch, n_cell_p)
    """
    ret: Dict[str, Tensor] = {}
    # tree_struct
    assert torch.all(batch[0]["tree_struct"] == batch[-1]["tree_struct"]), [
        batch[0]["tree_struct"].size(),
        batch[-1]["tree_struct"].size(),
    ]
    ret["tree_struct"] = batch[0]["tree_struct"]

    # cell2bound
    assert torch.all(batch[0]["cell2bound"] == batch[-1]["cell2bound"]), [
        batch[0]["cell2bound"].size(),
        batch[-1]["cell2bound"].size(),
    ]
    ret["cell2bound"] = batch[0]["cell2bound"]

    # cell2cross
    assert torch.all(batch[0]["cell2cross"] == batch[-1]["cell2cross"]), [
        batch[0]["cell2cross"].size(),
        batch[-1]["cell2cross"].size(),
    ]
    ret["cell2cross"] = batch[0]["cell2cross"]

    # bound_tens
    bound_tens_list: List[Tensor] = [ele["bound_tens"] for ele in batch]
    bound_tens: Tensor = torch.stack(bound_tens_list)
    ret["bound_tens"] = bound_tens

    # terminal_tens
    terminal_tens_list: List[Tensor] = [ele["terminal_tens"] for ele in batch]
    terminal_tens: Tensor = torch.stack(terminal_tens_list)
    ret["terminal_tens"] = terminal_tens

    # null_portal_tens
    null_portal_tens_list: List[Tensor] = [ele["null_portal_tens"] for ele in batch]
    null_portal_tens: Tensor = torch.stack(null_portal_tens_list)
    ret["null_portal_tens"] = null_portal_tens

    # golden
    golden_list: List[Tensor] = [ele["golden"] for ele in batch]
    golden: Tensor = torch.stack(golden_list)
    ret["golden"] = golden

    return ret
