"""DimeNet++ model (PyG-native)."""

from omegaconf import DictConfig
from torch_geometric.nn.models import DimeNetPlusPlus

from cliffordip.train.model_registry import register_model


@register_model("dimenetpp")
def build_dimenetpp(cfg: DictConfig) -> DimeNetPlusPlus:
    m = cfg.model
    return DimeNetPlusPlus(
        hidden_channels=m.hidden_channels,
        out_channels=m.get("out_channels", 1),
        num_blocks=m.num_blocks,
        int_emb_size=m.get("int_emb_size", 64),
        basis_emb_size=m.basis_emb_size,
        out_emb_channels=m.get("out_emb_channels", 256),
        num_spherical=m.num_spherical,
        num_radial=m.num_radial,
        cutoff=m.cutoff,
        envelope_exponent=m.get("envelope_exponent", 5),
    )
