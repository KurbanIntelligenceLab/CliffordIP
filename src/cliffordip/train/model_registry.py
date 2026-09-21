"""Model registry."""

import torch.nn as nn
from omegaconf import DictConfig

from cliffordip.train.registry import Registry

MODELS: Registry[nn.Module] = Registry("model", entry_point_group="cliffordip.models")


def register_model(name: str):
    """Decorator registering a ``(cfg) -> nn.Module`` factory under ``name``."""
    return MODELS.register(name)


def build_model(cfg: DictConfig) -> nn.Module:
    """Instantiate the model named by ``cfg.model.name``."""
    return MODELS.get(cfg.model.name)(cfg)


def list_models() -> list[str]:
    return MODELS.names()
