"""SchNet model (PyG-native)."""

from omegaconf import DictConfig
from torch_geometric.nn.models import SchNet

from cliffordip.train.model_registry import register_model


@register_model("schnet")
def build_schnet(cfg: DictConfig) -> SchNet:
    m = cfg.model
    return SchNet(
        hidden_channels=m.hidden_channels,
        num_filters=m.num_filters,
        num_interactions=m.num_interactions,
        num_gaussians=m.num_gaussians,
        cutoff=m.cutoff,
    )
