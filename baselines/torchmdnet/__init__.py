"""TorchMD-Net model (data_wrapper interface)."""

from omegaconf import DictConfig

from cliffordip.train.model_registry import register_model
from .wrapper import TorchMDNetWrapper


@register_model("torchmdnet")
def build_torchmdnet(cfg: DictConfig) -> TorchMDNetWrapper:
    m = cfg.model
    return TorchMDNetWrapper(
        model_arch=m.get("model_arch", "equivariant-transformer"),
        embedding_dimension=m.get("embedding_dimension", 128),
        num_layers=m.get("num_layers", 6),
        num_rbf=m.get("num_rbf", 64),
        rbf_type=m.get("rbf_type", "expnorm"),
        trainable_rbf=m.get("trainable_rbf", False),
        activation=m.get("activation", "silu"),
        cutoff_lower=m.get("cutoff_lower", 0.0),
        cutoff_upper=m.get("cutoff_upper", 6.0),
        max_z=m.get("max_z", 100),
        max_num_neighbors=m.get("max_num_neighbors", 50),
        num_heads=m.get("num_heads", 8),
        distance_influence=m.get("distance_influence", "both"),
        neighbor_embedding=m.get("neighbor_embedding", True),
        attn_activation=m.get("attn_activation", "silu"),
        output_model=m.get("output_model", "Scalar"),
        reduce_op=m.get("reduce_op", "add"),
        vector_cutoff=m.get("vector_cutoff", False),
        equivariance_invariance_group=m.get("equivariance_invariance_group", "O(3)"),
        static_shapes=m.get("static_shapes", False),
    )
