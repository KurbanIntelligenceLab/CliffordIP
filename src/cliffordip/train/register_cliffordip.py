"""Register Clifford model variants in the model registry."""

from omegaconf import DictConfig

from cliffordip.train.model_registry import register_model
from cliffordip.wrapper import CliffordIPWrapper


def _build(cfg: DictConfig) -> CliffordIPWrapper:
    m = cfg.model
    return CliffordIPWrapper(
        n_atom_types=m.get("n_atom_types", 100),
        n_channels=m.n_channels,
        n_interactions=m.n_interactions,
        n_rbf=m.get("n_rbf", 20),
        cutoff=m.cutoff,
        n_hidden_output=m.get("n_hidden_output", 64),
        max_neighbors=m.get("max_neighbors", 50),
        direct_forces=m.get("direct_forces", True),
        use_attention=m.get("use_attention", True),
        use_self_interaction=m.get("use_self_interaction", True),
        max_body_order=m.get("max_body_order", 3),
        use_l2=m.get("use_l2", True),
        use_multiscale=m.get("use_multiscale", True),
        use_gp_readout=m.get("use_gp_readout", True),
        n_heads=m.get("n_heads", 4),
        use_dens=m.get("use_dens", False),
        use_compile=m.get("use_compile", False),
    )


@register_model("cliffordip")
def build_cliffordip(cfg: DictConfig) -> CliffordIPWrapper:
    return _build(cfg)


@register_model("cliffordip_3m")
def build_cliffordip_3m(cfg: DictConfig) -> CliffordIPWrapper:
    return _build(cfg)


@register_model("cliffordip_10m")
def build_cliffordip_10m(cfg: DictConfig) -> CliffordIPWrapper:
    return _build(cfg)
