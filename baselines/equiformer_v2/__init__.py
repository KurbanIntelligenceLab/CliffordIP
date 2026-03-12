"""EquiformerV2 model (vendored from atomicarchitects/equiformer_v2)."""

from omegaconf import DictConfig

from cliffordip.train.model_registry import register_model
from .wrapper import EquiformerV2Wrapper


@register_model("equiformer_v2")
def build_equiformer_v2(cfg: DictConfig) -> EquiformerV2Wrapper:
    m = cfg.model
    task = cfg.dataset.get("task_type", "scalar")
    regress_forces = task in ("energy_forces", "s2ef")
    # Compact defaults targeting ~1M parameters for faster training
    return EquiformerV2Wrapper(
        num_layers=m.get("num_layers", 2),
        sphere_channels=m.get("sphere_channels", 30),
        attn_hidden_channels=m.get("attn_hidden_channels", 20),
        num_heads=m.get("num_heads", 4),
        attn_alpha_channels=m.get("attn_alpha_channels", 16),
        attn_value_channels=m.get("attn_value_channels", 8),
        ffn_hidden_channels=m.get("ffn_hidden_channels", 64),
        lmax_list=m.get("lmax_list", [4]),
        mmax_list=m.get("mmax_list", [2]),
        cutoff=m.get("cutoff", 12.0),
        max_neighbors=m.get("max_neighbors", 50),
        num_elements=m.get("n_atom_types", 100),
        regress_forces=regress_forces,
    )
