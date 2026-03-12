"""
PaiNN wrapper using SchNetPack implementation.

Wraps the PaiNN model from SchNetPack (https://github.com/atomistic-machine-learning/schnetpack)
for use with PyG Data objects and our training infrastructure.

Requires: schnetpack (pip install schnetpack)
"""

# PyTorch 2.7 removed T_co from torch.utils.data.dataloader (renamed to _T_co).
# Older schnetpack (<2.1) imports T_co, so patch it before importing schnetpack.
import torch.utils.data.dataloader as _dl
if not hasattr(_dl, 'T_co') and hasattr(_dl, '_T_co'):
    _dl.T_co = _dl._T_co

import torch
import torch.nn as nn
from torch_geometric.data import Data

from ..common.schnetpack_utils import pyg_to_schnetpack


class PaiNNWrapper(nn.Module):
    """
    Wrapper around SchNetPack's PaiNN that accepts PyG Data objects.

    All architecture parameters are exposed as constructor arguments and
    driven by the model YAML config.
    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=4,
        num_rbf=64,
        cutoff=6.0,
        max_neighbors=50,
        num_elements=100,
    ):
        super().__init__()
        self.max_neighbors = max_neighbors
        self.cutoff = cutoff
        self._backbone = None

        self._build(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            num_elements=num_elements,
        )

    def _build(self, **kwargs):
        """Build PaiNN from SchNetPack with given config."""
        try:
            from schnetpack.representation import PaiNN as SchNetPackPaiNN
            from schnetpack.nn import GaussianRBF, CosineCutoff
        except ImportError as e:
            raise ImportError("schnetpack is required for PaiNN. Install via: pip install schnetpack") from e

        cutoff_fn = CosineCutoff(self.cutoff)
        radial_basis = GaussianRBF(n_rbf=kwargs.get("num_rbf", 64), cutoff=self.cutoff)

        self._backbone = SchNetPackPaiNN(
            n_atom_basis=kwargs["hidden_channels"],
            n_interactions=kwargs["num_layers"],
            radial_basis=radial_basis,
            cutoff_fn=cutoff_fn,
        )

    def forward(self, data: Data):
        """
        Forward pass: returns energy for IS2RE (trainer computes forces via autograd).

        Parameters
        ----------
        data : PyG Data or Batch

        Returns
        -------
        energy : Tensor of shape [num_graphs]
        """
        from schnetpack import properties as spk_props

        # Convert PyG Data to SchNetPack format
        inputs = pyg_to_schnetpack(data, self.cutoff)

        # SchNetPack PaiNN forward
        output = self._backbone(inputs)

        # Extract energy (SchNetPack returns dict with multiple keys)
        if isinstance(output, dict):
            # Try different possible keys
            energy = output.get("scalar_representation", output.get("energy", output.get("y")))
            if energy is None:
                for key, value in output.items():
                    if isinstance(value, torch.Tensor) and value.dim() in [1, 2]:
                        energy = value
                        break

            if energy is None:
                raise RuntimeError(f"Could not find energy in SchNetPack PaiNN output. Keys: {output.keys()}")
        else:
            energy = output

        # Sum per-atom representations to get per-graph energy
        batch = inputs[spk_props.idx_m]
        if energy.dim() == 2:  # [n_atoms, hidden_dim]
            n_graphs = int(batch.max().item()) + 1
            per_atom = energy.mean(dim=-1)
            energy_per_graph = torch.zeros(n_graphs, device=energy.device, dtype=energy.dtype)
            energy_per_graph.index_add_(0, batch, per_atom)
            energy = energy_per_graph
        elif energy.dim() == 1:  # [n_atoms]
            n_graphs = int(batch.max().item()) + 1
            energy_per_graph = torch.zeros(n_graphs, device=energy.device, dtype=energy.dtype)
            energy_per_graph.index_add_(0, batch, energy)
            energy = energy_per_graph

        return energy.view(-1)
