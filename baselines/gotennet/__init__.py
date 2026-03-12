"""GotenNet model (data_wrapper interface)."""

from omegaconf import DictConfig

from cliffordip.train.model_registry import register_model
from .wrapper import GotenNetEnergyModel


@register_model("gotennet")
def build_gotennet(cfg: DictConfig) -> GotenNetEnergyModel:
    m = cfg.model
    return GotenNetEnergyModel(
        n_atom_basis=m.n_atom_basis,
        n_interactions=m.n_interactions,
        cutoff=m.cutoff,
        max_num_neighbors=m.max_num_neighbors,
        cutoff_fn_name=m.get("cutoff_fn_name", "cosine"),
        pooling=m.get("pooling", "auto"),
        task_type=cfg.dataset.get("task_type", "scalar"),
    )
