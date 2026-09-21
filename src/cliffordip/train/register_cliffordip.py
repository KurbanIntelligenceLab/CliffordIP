"""Register the CliffordIP model in the model registry."""

from omegaconf import DictConfig, OmegaConf

from cliffordip.train.model_registry import register_model
from cliffordip.wrapper import CliffordIPWrapper

NAMES = ("cliffordip", "cliffordip_3m", "cliffordip_10m")


def build_cliffordip(cfg: DictConfig) -> CliffordIPWrapper:
    """Build the model from ``cfg.model``."""
    return CliffordIPWrapper.from_mapping(OmegaConf.to_container(cfg.model, resolve=True))


for _name in NAMES:
    register_model(_name)(build_cliffordip)
