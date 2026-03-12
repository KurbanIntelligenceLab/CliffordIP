"""
Unified GotenNet energy model wrapper.

Replaces 6 inline GotenNet wrapper classes across trainers with a single
configurable module. Supports:
  - GotenNetWrapper (preferred) or GotenNet (fallback) encoder
  - Configurable pooling: mean (scalar tasks), add (energy tasks), or auto
  - Standard MLP head: Linear -> SiLU -> Linear -> 1
"""

import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.pool import radius_graph

_POOLING_MAP = {
    "mean": global_mean_pool,
    "add": global_add_pool,
}

_TASK_POOLING = {
    "scalar": "mean",
    "energy_forces": "add",
    "s2ef": "add",
    "is2re": "add",
}


class GotenNetEnergyModel(nn.Module):
    """
    GotenNet encoder + graph-level pooling + MLP head.

    Parameters
    ----------
    n_atom_basis : int
        Embedding dimension.
    n_interactions : int
        Number of interaction layers.
    cutoff : float
        Cutoff radius for edge construction.
    max_num_neighbors : int
        Max neighbors in radius graph.
    cutoff_fn_name : str
        ``"cosine"`` or ``"polynomial"``.
    pooling : str
        ``"mean"``, ``"add"``, or ``"auto"``. If ``"auto"``, resolved
        from *task_type*.
    task_type : str
        ``"scalar"``, ``"energy_forces"``, ``"s2ef"``, or ``"is2re"``.
        Only used when ``pooling="auto"``.
    """

    def __init__(
        self,
        n_atom_basis: int = 64,
        n_interactions: int = 2,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        cutoff_fn_name: str = "cosine",
        pooling: str = "auto",
        task_type: str = "scalar",
    ):
        super().__init__()
        self.cutoff_val = cutoff
        self.max_num_neighbors = max_num_neighbors

        # Resolve pooling
        if pooling == "auto":
            pooling = _TASK_POOLING.get(task_type, "mean")
        if pooling not in _POOLING_MAP:
            raise ValueError(f"Unknown pooling '{pooling}'. Use 'mean' or 'add'.")
        self.pool = _POOLING_MAP[pooling]

        # Build cutoff function object
        from gotennet.models.components.layers import (  # type: ignore
            CosineCutoff,
            PolynomialCutoff,
        )

        if cutoff_fn_name.lower() in ("cosine", "cosinecutoff"):
            cutoff_fn = CosineCutoff(cutoff)
        elif cutoff_fn_name.lower() in ("polynomial", "polynomialcutoff"):
            cutoff_fn = PolynomialCutoff(cutoff)
        else:
            raise ValueError(f"Unknown cutoff_fn_name={cutoff_fn_name!r}. Use 'cosine' or 'polynomial'.")

        # Prefer GotenNetWrapper; fallback to GotenNet base
        self.mode = None
        try:
            from gotennet import GotenNetWrapper  # type: ignore

            self.mode = "wrapper"
            self.encoder = GotenNetWrapper(
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                cutoff_fn=cutoff_fn,
                max_num_neighbors=max_num_neighbors,
            )
        except ImportError:
            from gotennet import GotenNet  # type: ignore

            self.mode = "base"
            self.encoder = GotenNet(
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                cutoff_fn=cutoff_fn,
            )

        self.head = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 1),
        )

    def forward(self, data):
        if self.mode == "wrapper":
            h, _X = self.encoder(data)
        else:
            edge_index = radius_graph(
                data.pos,
                r=self.cutoff_val,
                batch=data.batch,
                max_num_neighbors=self.max_num_neighbors,
            )
            row, col = edge_index
            edge_vec = data.pos[col] - data.pos[row]
            edge_diff = edge_vec.norm(dim=-1)
            h, _X = self.encoder(data.z, edge_index, edge_diff, edge_vec)

        g = self.pool(h, data.batch)
        return self.head(g).view(-1)
