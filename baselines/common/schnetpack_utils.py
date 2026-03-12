"""Utilities for converting PyG Data to SchNetPack format."""

import torch
from torch_geometric.data import Data


def pyg_to_schnetpack(data: Data, cutoff: float) -> dict:
    """Convert PyG Data/Batch to SchNetPack input format.

    Parameters
    ----------
    data : Data or Batch
        PyG data with z (atomic numbers), pos (positions), optional edge_index.
    cutoff : float
        Cutoff radius for neighbor list construction.

    Returns
    -------
    dict compatible with SchNetPack PaiNN model.
    """
    # Use schnetpack.properties for correct key names across versions
    try:
        from schnetpack import properties as spk_props
    except ImportError:
        spk_props = None

    pos = data.pos
    z = data.z if hasattr(data, "z") else data.atomic_numbers

    if hasattr(data, "batch") and data.batch is not None:
        batch = data.batch
    else:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    # Build edge_index if not provided or rebuild for cutoff
    if not hasattr(data, "edge_index") or data.edge_index is None:
        from torch_cluster import radius_graph

        edge_index = radius_graph(pos, r=cutoff, batch=batch)
    else:
        edge_index = data.edge_index

    n_atoms = torch.bincount(batch)

    # Compute pairwise displacement vectors Rij = pos[j] - pos[i]
    r_ij = pos[edge_index[1]] - pos[edge_index[0]]

    # SchNetPack format — use properties module for version-safe key names
    if spk_props is not None:
        inputs = {
            spk_props.Z: z.long().view(-1),
            spk_props.R: pos,
            spk_props.Rij: r_ij,
            spk_props.idx_i: edge_index[0],
            spk_props.idx_j: edge_index[1],
            spk_props.n_atoms: n_atoms,
            spk_props.idx_m: batch,
        }
        if hasattr(data, "cell") and data.cell is not None:
            inputs[spk_props.cell] = data.cell
        if hasattr(data, "pbc") and data.pbc is not None:
            inputs[spk_props.pbc] = data.pbc
    else:
        inputs = {
            "_atomic_numbers": z.long().view(-1),
            "_positions": pos,
            "_Rij": r_ij,
            "_idx_i": edge_index[0],
            "_idx_j": edge_index[1],
            "_n_atoms": n_atoms,
            "_idx_m": batch,
        }
        if hasattr(data, "cell") and data.cell is not None:
            inputs["_cell"] = data.cell
        if hasattr(data, "pbc") and data.pbc is not None:
            inputs["_pbc"] = data.pbc

    return inputs
