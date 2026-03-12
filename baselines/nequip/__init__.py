"""NequIP model (data_wrapper interface)."""

from omegaconf import DictConfig

from cliffordip.train.model_registry import register_model
from .wrapper import NequIPWrapper


@register_model("nequip")
def build_nequip(cfg: DictConfig) -> NequIPWrapper:
    m = cfg.model
    task = cfg.dataset.get("task_type", "scalar")
    do_derivatives = task in ("energy_forces", "s2ef")
    return NequIPWrapper(
        r_max=m.get("r_max", 6.0),
        num_layers=m.get("num_layers", 4),
        l_max=m.get("l_max", 1),
        parity=m.get("parity", True),
        num_features=m.get("num_features", 32),
        type_embed_num_features=m.get("type_embed_num_features"),
        radial_mlp_depth=m.get("radial_mlp_depth", 2),
        radial_mlp_width=m.get("radial_mlp_width", 64),
        num_bessels=m.get("num_bessels", 8),
        polynomial_cutoff_p=m.get("polynomial_cutoff_p", 6),
        max_num_elements=m.get("max_num_elements", 90),
        do_derivatives=do_derivatives,
    )
