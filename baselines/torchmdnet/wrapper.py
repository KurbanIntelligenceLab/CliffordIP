"""
TorchMD-Net wrapper for standalone use with PyG Data objects.

Wraps TorchMD-Net models (equivariant-transformer, tensornet, etc.) so they can be used
with our standard PyG DataLoader-based training infrastructure. Trains from scratch.

Requires: torchmd-net (pip install torchmd-net)
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data


class TorchMDNetWrapper(nn.Module):
    """
    Wrapper around TorchMD-Net that accepts PyG Data objects.

    Forward takes data with z, pos, batch and returns (energy, forces) for S2EF.
    """

    def __init__(
        self,
        model_arch="equivariant-transformer",
        embedding_dimension=128,
        num_layers=6,
        num_rbf=64,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        cutoff_lower=0.0,
        cutoff_upper=6.0,
        max_z=100,
        max_num_neighbors=50,
        num_heads=8,
        distance_influence="both",
        neighbor_embedding=True,
        attn_activation="silu",
        output_model="Scalar",
        reduce_op="add",
        vector_cutoff=False,
        equivariance_invariance_group="O(3)",
        static_shapes=False,
    ):
        super().__init__()
        self._backbone = None
        self._build(
            model_arch=model_arch,
            embedding_dimension=embedding_dimension,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            num_heads=num_heads,
            distance_influence=distance_influence,
            neighbor_embedding=neighbor_embedding,
            attn_activation=attn_activation,
            output_model=output_model,
            reduce_op=reduce_op,
            vector_cutoff=vector_cutoff,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
        )

    def _build(self, **kwargs):
        """Build TorchMD-Net from scratch with given config."""
        try:
            from torchmdnet.models.model import create_model

            args = {
                "model": kwargs.get("model_arch", "equivariant-transformer"),
                "embedding_dimension": kwargs.get("embedding_dimension", 128),
                "num_layers": kwargs.get("num_layers", 6),
                "num_rbf": kwargs.get("num_rbf", 64),
                "rbf_type": kwargs.get("rbf_type", "expnorm"),
                "trainable_rbf": kwargs.get("trainable_rbf", False),
                "activation": kwargs.get("activation", "silu"),
                "cutoff_lower": float(kwargs.get("cutoff_lower", 0.0)),
                "cutoff_upper": float(kwargs.get("cutoff_upper", 6.0)),
                "max_z": kwargs.get("max_z", 100),
                "max_num_neighbors": kwargs.get("max_num_neighbors", 50),
                "num_heads": kwargs.get("num_heads", 8),
                "distance_influence": kwargs.get("distance_influence", "both"),
                "neighbor_embedding": kwargs.get("neighbor_embedding", True),
                "attn_activation": kwargs.get("attn_activation", "silu"),
                "output_model": kwargs.get("output_model", "Scalar"),
                "reduce_op": kwargs.get("reduce_op", "add"),
                "derivative": True,
                "prior_model": None,
                "atom_filter": -1,
                "aggr": "add",
                "precision": 32,
                "vector_cutoff": kwargs.get("vector_cutoff", False),
                "equivariance_invariance_group": kwargs.get("equivariance_invariance_group", "O(3)"),
                "static_shapes": kwargs.get("static_shapes", False),
            }
            if args["model"] == "graph-network":
                args["aggr"] = kwargs.get("aggr", "add")
                args["neighbor_embedding"] = kwargs.get("neighbor_embedding", True)
            if args["model"] == "tensornet" or args["model"] == "tensornet2":
                args["equivariance_invariance_group"] = kwargs.get("equivariance_invariance_group", "O(3)")
                args["static_shapes"] = kwargs.get("static_shapes", False)

            self._backbone = create_model(
                args,
                prior_model=None,
                mean=torch.tensor(0.0),
                std=torch.tensor(1.0),
            )

        except ImportError as e:
            raise ImportError("torchmd-net is required for TorchMD-Net. Install via: pip install torchmd-net") from e

    def forward(self, data: Data):
        """
        Forward pass: returns (energy, forces) for S2EF.
        TorchMD-Net uses derivative=True so forces come from autograd.
        """
        z = data.z if hasattr(data, "z") else data.atomic_numbers
        pos = data.pos
        batch = data.batch if hasattr(data, "batch") else torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        # TorchMD-Net always computes forces via autograd (derivative=True),
        # so positions must have requires_grad even during eval
        pos = pos.clone().requires_grad_(True)
        with torch.enable_grad():
            energy, neg_dy = self._backbone(z, pos, batch, box=None)

        forces = neg_dy if neg_dy is not None else torch.zeros_like(pos)
        energy = energy.detach() if not torch.is_grad_enabled() else energy
        if energy.dim() > 1:
            energy = energy.view(-1)
        else:
            energy = energy.flatten()
        return (energy, forces) if forces.numel() > 0 else energy
