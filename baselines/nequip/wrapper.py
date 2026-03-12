"""
NequIP wrapper for standalone use with PyG Data objects.

Wraps the NequIP model so it can be used with our standard PyG DataLoader-based
training infrastructure. Trains from scratch.

Uses NequIPGNNModel (nequip 0.16+ API).

Requires: nequip (pip install nequip)
"""

import os
import torch

def _nequip_log(msg: str, *args):
    """Log NequIP diagnostics. Set NEQUIP_DEBUG=1 to enable."""
    if not os.environ.get("NEQUIP_DEBUG"):
        return
    if args:
        msg = msg % args
    print(f"[NequIP] {msg}")
import torch.nn as nn
from torch_geometric.data import Data

from ..common.nequip_utils import pyg_to_atomic_data


class NequIPWrapper(nn.Module):
    """
    Wrapper around NequIP that accepts PyG Data objects.

    Forward converts PyG data to AtomicDataDict format, runs NequIP, returns (energy, forces).
    """

    def __init__(
        self,
        r_max=6.0,
        num_layers=4,
        l_max=1,
        parity=True,
        num_features=32,
        type_embed_num_features=None,
        radial_mlp_depth=2,
        radial_mlp_width=64,
        num_bessels=8,
        polynomial_cutoff_p=6,
        max_num_elements=90,
        do_derivatives=False,
    ):
        super().__init__()
        self.r_max = r_max
        self.do_derivatives = do_derivatives
        type_names = [str(i) for i in range(1, max_num_elements + 1)]
        self.type_names = type_names
        self._backbone = self._build_model(
            r_max=r_max,
            num_layers=num_layers,
            do_derivatives=do_derivatives,
            l_max=l_max,
            parity=parity,
            num_features=num_features,
            type_embed_num_features=type_embed_num_features or num_features,
            radial_mlp_depth=radial_mlp_depth,
            radial_mlp_width=radial_mlp_width,
            num_bessels=num_bessels,
            polynomial_cutoff_p=polynomial_cutoff_p,
            type_names=type_names,
        )

    def _build_model(self, **kwargs):
        """Build NequIP via NequIPGNNModel (nequip 0.16+ API)."""
        try:
            # PyTorch 2.6+ compatibility: e3nn (nequip dep) loads constants.pt which needs slice
            import torch.serialization

            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([slice])
            from nequip.model import NequIPGNNModel
            from nequip.utils.global_state import set_global_state
        except ImportError as e:
            if "nequip" in str(e).lower() and "No module named" in str(e):
                raise ImportError("nequip is required for NequIP. Install via: pip install nequip") from e
            raise

        set_global_state(allow_tf32=False)
        avg_num_neighbors = 50.0
        model = NequIPGNNModel(
            r_max=kwargs["r_max"],
            type_names=kwargs["type_names"],
            num_layers=kwargs["num_layers"],
            l_max=kwargs["l_max"],
            parity=kwargs["parity"],
            num_features=kwargs["num_features"],
            type_embed_num_features=kwargs["type_embed_num_features"],
            radial_mlp_depth=kwargs["radial_mlp_depth"],
            radial_mlp_width=kwargs["radial_mlp_width"],
            num_bessels=kwargs["num_bessels"],
            polynomial_cutoff_p=kwargs["polynomial_cutoff_p"],
            avg_num_neighbors=avg_num_neighbors,
            seed=42,
            model_dtype="float32",
            do_derivatives=kwargs.get("do_derivatives", False),
        )
        return model

    def forward(self, data: Data):
        """Forward pass: returns (energy, forces). Forces may be zeros for scalar-only tasks."""
        atomic_dict = pyg_to_atomic_data(data, self.r_max, self.type_names)
        backbone_dtype = next(self._backbone.parameters()).dtype
        _nequip_log("data.pos dtype=%s shape=%s | backbone_dtype=%s", data.pos.dtype, tuple(data.pos.shape), backbone_dtype)
        _nequip_log("atomic_dict float dtypes: %s", {k: str(v.dtype) for k, v in atomic_dict.items() if hasattr(v, "dtype") and v.is_floating_point()})
        for k in ("pos", "edge_vectors", "edge_lengths", "edge_cell_shift", "cell"):
            if k in atomic_dict and atomic_dict[k].is_floating_point():
                atomic_dict[k] = atomic_dict[k].to(backbone_dtype)
        if atomic_dict["pos"].requires_grad is False:
            atomic_dict["pos"] = atomic_dict["pos"].clone().requires_grad_(True)
        # When computing forces (do_derivatives=True), remove pre-computed
        # edge_vectors so NequIP's ForceStressOutput takes the positions branch
        # which properly supports training (create_graph=self.training).
        # The edge_vectors branch is inference-only and frees the graph.
        if self.do_derivatives:
            atomic_dict.pop("edge_vectors", None)
            atomic_dict.pop("edge_lengths", None)
        try:
            out = self._backbone(atomic_dict)
        except Exception as e:
            if os.environ.get("NEQUIP_DEBUG"):
                dtypes = {k: (str(v.dtype) if hasattr(v, "dtype") else type(v).__name__) for k, v in atomic_dict.items()}
                print(f"[NequIP] backbone FAILED: {e} | dtypes: {dtypes}")
            raise
        energy = out["total_energy"].view(-1)
        forces = out.get("forces")
        if forces is None:
            forces = torch.zeros_like(atomic_dict["pos"], device=energy.device, dtype=energy.dtype)
        return (energy, forces)
