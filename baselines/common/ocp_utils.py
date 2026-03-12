"""Shared utilities for OCP model wrappers."""

import torch
from torch_geometric.data import Data


def pyg_to_ocp_batch(data: Data) -> Data:
    """Convert a standard PyG Data/Batch to OCP format.

    Adds ``natoms``, ``cell``, and ``atomic_numbers`` attributes that
    OCP models expect but standard PyG data objects don't carry.
    """
    batch = data.clone()

    if not hasattr(batch, "natoms") or batch.natoms is None:
        if hasattr(batch, "batch") and batch.batch is not None:
            batch.natoms = torch.bincount(batch.batch)
        else:
            batch.natoms = torch.tensor([batch.pos.size(0)])

    if not hasattr(batch, "cell") or batch.cell is None:
        n_graphs = batch.natoms.size(0)
        batch.cell = torch.eye(3, device=batch.pos.device).unsqueeze(0).expand(n_graphs, -1, -1) * 100.0

    if hasattr(batch, "z") and not hasattr(batch, "atomic_numbers"):
        batch.atomic_numbers = batch.z

    return batch
