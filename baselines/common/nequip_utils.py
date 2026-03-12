"""Utilities for converting PyG Data to NequIP AtomicDataDict format."""

import torch
from torch_geometric.data import Data


def pyg_to_atomic_data(data: Data, r_max: float, type_names: list) -> dict:
    """Convert PyG Data/Batch to NequIP AtomicDataDict format.

    Parameters
    ----------
    data : Data or Batch
        PyG data with z (or atomic_numbers), pos, batch.
    r_max : float
        Cutoff radius for radius_graph.
    type_names : list
        List of type names, e.g. ["1", "2", ..., "90"] for elements 1-90.
        atom_types will be index into this (z-1 clamped).

    Returns
    -------
    dict compatible with nequip.data.from_dict and model forward.
    """
    pos = data.pos
    z = data.z if hasattr(data, "z") else data.atomic_numbers
    if hasattr(data, "batch") and data.batch is not None:
        batch = data.batch
    else:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    device = pos.device
    dtype = pos.dtype
    n_graphs = int(batch.max().item()) + 1

    # edge_index from radius_graph
    from torch_cluster import radius_graph

    edge_index = radius_graph(pos, r=r_max, batch=batch)

    # atom_types: index into type_names (z-1 clamped to 0..len-1)
    num_types = len(type_names)
    atom_types = (z.long().flatten() - 1).clamp(0, num_types - 1)

    # cell, pbc: no PBC
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n_graphs, -1, -1) * 100.0
    pbc = torch.zeros(n_graphs, 3, dtype=torch.bool, device=device)

    # edge_cell_shift: zeros for non-PBC
    n_edges = edge_index.size(1)
    edge_cell_shift = torch.zeros(n_edges, 3, dtype=dtype, device=device)

    # edge_vectors, edge_lengths (required by NequIP)
    sender, receiver = edge_index[0], edge_index[1]
    edge_vectors = pos[receiver] - pos[sender]
    edge_lengths = torch.linalg.norm(edge_vectors, dim=-1, keepdim=True)

    # num_atoms per graph
    num_atoms = torch.bincount(batch, minlength=n_graphs)
    # ptr: [0, n1, n1+n2, ...] for NequIP batching
    ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), num_atoms.cumsum(0)])

    out = {
        "pos": pos,
        "atomic_numbers": z.long().view(-1, 1),
        "atom_types": atom_types.view(-1, 1),
        "edge_index": edge_index,
        "edge_cell_shift": edge_cell_shift,
        "edge_vectors": edge_vectors,
        "edge_lengths": edge_lengths.squeeze(-1),  # [n_edges] as expected by NequIP
        "cell": cell,
        "pbc": pbc,
        "batch": batch,
        "ptr": ptr,
        "num_atoms": num_atoms,
    }
    return out
