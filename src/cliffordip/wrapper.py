"""Training wrapper: neighbor construction, torch.compile and the DeNS auxiliary loss."""

import torch
import torch.nn as nn

from cliffordip.config import CliffordIPConfig
from cliffordip.interaction import CliffordNet
from cliffordip.neighbors import build_edges


class CliffordIPWrapper(nn.Module):
    """Neighbor construction plus the CliffordIP network.

    Example::
        model = CliffordIPWrapper(n_channels=52, n_interactions=5, cutoff=6.0)
        energy, forces = model(data)
        energy, forces, dens_loss = model.forward_with_dens(data)
    """

    def __init__(self, config: CliffordIPConfig | None = None, **overrides):
        super().__init__()
        self.config = (config or CliffordIPConfig()).replace(**overrides)
        cfg = self.config

        self.cutoff = cfg.cutoff
        self.max_neighbors = cfg.max_neighbors
        self.use_dens = cfg.use_dens

        self._model: nn.Module = CliffordNet(
            n_atom_types=cfg.n_atom_types,
            n_channels=cfg.n_channels,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            n_hidden_output=cfg.n_hidden_output,
            use_attention=cfg.use_attention,
            use_self_interaction=cfg.use_self_interaction,
            max_body_order=cfg.max_body_order,
            use_multiscale=cfg.use_multiscale,
            use_gp_readout=cfg.use_gp_readout,
            n_heads=cfg.n_heads,
            use_dens=cfg.use_dens,
            dens_noise_std=cfg.dens_noise_std,
        )

        if cfg.use_compile:
            self._model = torch.compile(  # type: ignore[assignment]
                self._model, mode=cfg.compile_mode, dynamic=True
            )

    @classmethod
    def from_mapping(cls, mapping) -> "CliffordIPWrapper":
        """Build from a config mapping such as ``cfg.model``."""
        return cls(CliffordIPConfig.from_mapping(mapping))

    def _get_raw_model(self):
        """The network itself, unwrapping torch.compile."""
        return getattr(self._model, "_orig_mod", self._model)

    def _build_edges(self, data):
        """Radius graph for ``data``, periodic when a unit cell is present."""
        return build_edges(data, self.cutoff, self.max_neighbors)

    def forward(self, data):
        """Return per-graph energies and per-atom forces."""
        edge_index = self._build_edges(data)
        energy, forces = self._model(data.z, data.pos, edge_index, data.batch)
        return energy.view(-1), forces

    def forward_with_dens(self, data):
        """Forward with DeNS denoising auxiliary loss.

        Returns:
            (energy, forces, dens_loss)
        """
        edge_index = self._build_edges(data)
        energy, forces, dens_loss = self._get_raw_model().forward_with_dens(
            data.z, data.pos, edge_index, data.batch
        )
        return energy.view(-1), forces, dens_loss
