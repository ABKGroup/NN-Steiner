from __future__ import annotations

from typing import Dict

import hydra
import torch
from omegaconf import DictConfig

from arora.flow import run


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
