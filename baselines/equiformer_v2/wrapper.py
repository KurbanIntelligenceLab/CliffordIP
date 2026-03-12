"""
EquiformerV2 wrapper using vendored implementation.

Based on: https://github.com/atomicarchitects/equiformer_v2
Paper: EquiformerV2 (ICLR 2024)
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data


class EquiformerV2Wrapper(nn.Module):
    """Thin wrapper around EquiformerV2_OC20 that already accepts PyG Data."""

    def __init__(
        self,
        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        lmax_list=[6],
        mmax_list=[2],
        cutoff=12.0,
        max_neighbors=50,
        num_elements=100,
        regress_forces=False,
        **kwargs,  # Catch any extra config params
    ):
        super().__init__()

        from .src.equiformer_v2_oc20 import EquiformerV2_OC20

        # EquiformerV2 constructor expects specific param names
        self.model = EquiformerV2_OC20(
            num_atoms=None,  # Not used
            bond_feat_dim=None,  # Not used
            num_targets=1,  # Energy prediction
            use_pbc=False,
            regress_forces=regress_forces,
            otf_graph=True,
            max_neighbors=max_neighbors,
            max_radius=cutoff,
            max_num_elements=num_elements,
            num_layers=num_layers,
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
        )

    def forward(self, data: Data):
        """Forward pass - directly call model (it accepts PyG Data!)"""
        # EquiformerV2_OC20 expects data.atomic_numbers and data.natoms
        if not hasattr(data, 'atomic_numbers'):
            data.atomic_numbers = data.z
        if not hasattr(data, 'natoms'):
            # Compute number of atoms per graph from batch indices
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.z.size(0), dtype=torch.long, device=data.z.device)
            data.natoms = torch.bincount(batch)

        output = self.model(data)
        # Model returns (energy, forces) tuple when regress_forces=True
        if isinstance(output, tuple):
            energy, forces = output
            return (energy.view(-1), forces)
        else:
            return output.view(-1)
