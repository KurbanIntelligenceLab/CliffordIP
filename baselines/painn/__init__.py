"""PaiNN model (data_wrapper interface)."""

from omegaconf import DictConfig

from cliffordip.train.model_registry import register_model
from .wrapper import PaiNNWrapper


@register_model("painn")
def build_painn(cfg: DictConfig) -> PaiNNWrapper:
    m = cfg.model
    return PaiNNWrapper(
        hidden_channels=m.get("hidden_channels", 128),
        num_layers=m.get("num_layers", 4),
        num_rbf=m.get("num_rbf", 64),
        cutoff=m.cutoff,
        max_neighbors=m.get("max_neighbors", 50),
        num_elements=m.get("num_elements", 100),
    )
