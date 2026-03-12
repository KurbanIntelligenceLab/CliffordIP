"""
CliffordIP message-passing GNN backbone.

Grade-sparse GP dispatch, equivariant attention, multi-body interactions,
per-layer energy readout, and DeNS auxiliary objective support.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_softmax

from .cliffordip import (
    ALL_GRADES,
    CliffordAlgebra,
    CliffordIPGateActivation,
    CliffordIPLinear,
    CliffordIPNorm,
    compute_gp_output_grades,
    compute_layer_grades,
    compute_l2_features,
    make_grades01_mv,
    make_scalar_mv,
)


# ============================================================
# Radial Basis + Cutoff
# ============================================================


class RadialBasisFunctions(nn.Module):
    """Gaussian RBF for distance encoding."""

    def __init__(self, n_rbf: int = 20, cutoff: float = 5.0, trainable: bool = False):
        super().__init__()
        self.n_rbf = n_rbf
        offsets = torch.linspace(0.0, cutoff, n_rbf)
        widths = torch.full((n_rbf,), (offsets[1] - offsets[0]).item())
        if trainable:
            self.offsets = nn.Parameter(offsets)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("offsets", offsets)
            self.register_buffer("widths", widths)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        return torch.exp(-0.5 * ((dist - self.offsets) / self.widths) ** 2)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope."""

    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.cos(dist * torch.pi / self.cutoff) + 1.0) * (
            dist < self.cutoff
        ).float()


# ============================================================
# Edge Embedding with L=2 augmentation
# ============================================================


class CliffordEdgeEmbedding(nn.Module):
    """Edge features as grade-(0,1) multivectors + L=2 invariant features.

    The L=2 symmetric traceless features (5 components from d⊗d)
    are injected as additional invariant inputs to the radial networks,
    giving the model access to d-orbital angular information that
    Cl(3,0) cannot natively represent.
    """

    def __init__(
        self,
        n_rbf: int = 20,
        n_channels: int = 128,
        cutoff: float = 5.0,
        use_l2: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.use_l2 = use_l2
        self.rbf = RadialBasisFunctions(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        # Input dim: RBF + optional L=2 features
        in_dim = n_rbf + (5 if use_l2 else 0)

        self.scalar_net = nn.Sequential(
            nn.Linear(in_dim, n_channels),
            nn.SiLU(),
            nn.Linear(n_channels, n_channels),
        )
        self.vector_net = nn.Sequential(
            nn.Linear(in_dim, n_channels),
            nn.SiLU(),
            nn.Linear(n_channels, n_channels),
        )

    def forward(
        self, dist: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dist: (E,) distances
            direction: (E, 3) unit vectors
        Returns:
            (E, C, 8) multivectors with grades 0 and 1 populated
        """
        rbf = self.rbf(dist)  # (E, n_rbf)
        env = self.cutoff_fn(dist)  # (E,)

        # Concatenate RBF with L=2 features for richer angular info
        if self.use_l2:
            l2 = compute_l2_features(direction)  # (E, 5)
            feat = torch.cat([rbf, l2], dim=-1)  # (E, n_rbf+5)
        else:
            feat = rbf

        # Grade-0: scalar weights
        g0 = (self.scalar_net(feat) * env.unsqueeze(-1)).unsqueeze(-1)  # (E, C, 1)

        # Grade-1: direction × invariant weight
        v_w = self.vector_net(feat) * env.unsqueeze(-1)  # (E, C)
        g1 = v_w.unsqueeze(-1) * direction.unsqueeze(-2)  # (E, C, 3)

        return make_grades01_mv(g0, g1)


# ============================================================
# Atom Embedding
# ============================================================


class CliffordAtomEmbedding(nn.Module):
    """Atomic numbers → scalar (grade-0) multivectors."""

    def __init__(self, n_atom_types: int = 100, n_channels: int = 128):
        super().__init__()
        self.embed = nn.Embedding(n_atom_types, n_channels)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        s = self.embed(atomic_numbers)  # (N, C)
        return make_scalar_mv(s)  # (N, C, 8) — single alloc


# ============================================================
# Equivariant Attention
# ============================================================


class CliffordAttention(nn.Module):
    """Equivariant dot-product attention over neighbors.

    Query/Key from grade-0 (invariant) for attention weights.
    Value is the full multivector message.
    Compatible with scatter_softmax for variable neighbor counts.
    """

    def __init__(self, n_channels: int, n_heads: int = 4, n_rbf: int = 20):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = n_channels // n_heads
        assert n_channels % n_heads == 0

        self.q_proj = nn.Linear(n_channels, n_channels, bias=False)
        self.k_proj = nn.Linear(n_channels, n_channels, bias=False)
        self.rbf_proj = nn.Linear(n_rbf, n_channels, bias=False)
        self.rbf = RadialBasisFunctions(n_rbf)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        h_i: torch.Tensor,      # (E, C) — receiver grade-0 features
        h_j: torch.Tensor,      # (E, C) — sender grade-0 features
        dist: torch.Tensor,     # (E,)
        dst: torch.Tensor,      # (E,) — destination indices
        n_nodes: int,
        rbf: Optional[torch.Tensor] = None,  # (E, n_rbf) pre-computed RBF
    ) -> torch.Tensor:
        """Returns (E, 1, 1) attention weights for each edge."""
        q = self.q_proj(h_i)  # (E, C)
        k = self.k_proj(h_j)  # (E, C)
        rbf_w = self.rbf_proj(rbf if rbf is not None else self.rbf(dist))  # (E, C)

        # Multi-head dot-product attention
        E, C = q.shape
        q = q.view(E, self.n_heads, self.head_dim)
        k = k.view(E, self.n_heads, self.head_dim)
        rbf_w = rbf_w.view(E, self.n_heads, self.head_dim)

        attn_logits = (q * k * rbf_w).sum(-1) * self.scale  # (E, n_heads)

        # Softmax over neighbors of each receiver
        attn = scatter_softmax(attn_logits, dst, dim=0)  # (E, n_heads)
        attn = attn.mean(dim=-1)  # (E,) average over heads

        return attn.unsqueeze(-1).unsqueeze(-1)  # (E, 1, 1) for broadcasting


# ============================================================
# Message Function with Attention + Grade-Sparse GP
# ============================================================


class CliffordMessageFunction(nn.Module):
    """Grade-aware message with equivariant attention and GP dispatch.

    m_ij = attn_ij * [W_he·GP(H_j, E_ij) + W_eh·GP(E_ij, H_j) + skip(H_j)]

    Uses grade-sparse GP dispatch for early layers.
    """

    def __init__(
        self,
        n_channels: int,
        n_rbf: int,
        edge_grades: Tuple[int, ...],
        node_input_grades: Tuple[int, ...],
        gp_output_grades: Tuple[int, ...],
        use_attention: bool = True,
        n_heads: int = 4,
    ):
        super().__init__()
        self.alg = CliffordAlgebra()
        self.n_channels = n_channels
        self.node_input_grades = node_input_grades
        self.edge_grades = edge_grades
        self.use_attention = use_attention

        self.pre_edge = CliffordIPLinear(
            n_channels, n_channels, bias=False, active_grades=edge_grades
        )
        self.proj_he = CliffordIPLinear(
            n_channels, n_channels, bias=False, active_grades=gp_output_grades
        )
        self.proj_eh = CliffordIPLinear(
            n_channels, n_channels, bias=False, active_grades=gp_output_grades
        )

        # Scalar skip
        self.skip = nn.Linear(n_channels, n_channels, bias=False)

        if use_attention:
            self.attention = CliffordAttention(n_channels, n_heads, n_rbf)
        else:
            # Fallback: radial gating
            self.rbf = RadialBasisFunctions(n_rbf)
            self.radial_gate = nn.Sequential(
                nn.Linear(n_rbf, n_channels),
                nn.SiLU(),
                nn.Linear(n_channels, n_channels),
                nn.Sigmoid(),
            )

    def forward(
        self,
        h_j: torch.Tensor,
        h_i: torch.Tensor,
        edge_mv: torch.Tensor,
        dist: torch.Tensor,
        dst: torch.Tensor,
        n_nodes: int,
        rbf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e = self.pre_edge(edge_mv)

        # Grade-sparse GP dispatch
        gp_he = self.alg.dispatch_gp(h_j, e, self.node_input_grades, self.edge_grades)
        gp_eh = self.alg.dispatch_gp(e, h_j, self.edge_grades, self.node_input_grades)

        # Scalar skip (fused — avoids (E, C, 8) zero allocation)
        msg = self.proj_he(gp_he) + self.proj_eh(gp_eh)
        msg[..., 0] = msg[..., 0] + self.skip(h_j[..., 0])

        # Attention or radial gating
        if self.use_attention:
            attn = self.attention(
                h_i[..., 0], h_j[..., 0], dist, dst, n_nodes, rbf=rbf
            )
            msg = msg * attn
        else:
            rbf_val = rbf if rbf is not None else self.rbf(dist)
            gate = self.radial_gate(rbf_val).unsqueeze(-1)
            msg = msg * gate

        return msg


# ============================================================
# Self-Interaction Layer
# ============================================================


class CliffordSelfInteraction(nn.Module):
    """Per-node GP self-interaction that mixes grades across channels.

    h_self = GP(W_proj · h, h)

    Provides grade mixing that CliffordIPLinear cannot do alone.
    """

    def __init__(self, n_channels: int, active_grades: Tuple[int, ...] = ALL_GRADES):
        super().__init__()
        self.alg = CliffordAlgebra()
        self.proj = CliffordIPLinear(
            n_channels, n_channels, bias=False, active_grades=active_grades
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.alg.geometric_product(self.proj(h), h)


# ============================================================
# Multi-Body Interaction (MACE-style)
# ============================================================


class CliffordMultiBodyInteraction(nn.Module):
    """Iterated GP for higher body-order correlations.

    After aggregation:
        agg        = sum_j m_ij            (2-body)
        three_body = GP(agg, agg)          (3-body)
        four_body  = GP(three_body, agg)   (4-body, optional)

    Each body order gets a learnable weight.
    """

    def __init__(
        self,
        n_channels: int,
        active_grades: Tuple[int, ...] = ALL_GRADES,
        max_body_order: int = 3,
    ):
        super().__init__()
        self.alg = CliffordAlgebra()
        self.max_body_order = max_body_order
        self.active_grades = active_grades

        self.w2 = CliffordIPLinear(
            n_channels, n_channels, bias=False, active_grades=active_grades
        )
        self.w3 = CliffordIPLinear(
            n_channels, n_channels, bias=False, active_grades=active_grades
        )
        if max_body_order >= 4:
            self.w4 = CliffordIPLinear(
                n_channels, n_channels, bias=False, active_grades=active_grades
            )

    def forward(self, agg: torch.Tensor) -> torch.Tensor:
        """(N, C, 8) aggregated messages → (N, C, 8) multi-body features."""
        out = self.w2(agg)  # 2-body

        if self.max_body_order >= 3:
            three_body = self.alg.dispatch_gp(agg, agg, self.active_grades, self.active_grades)
            out = out + self.w3(three_body)

            if self.max_body_order >= 4:
                gp_out = compute_gp_output_grades(self.active_grades, self.active_grades)
                four_body = self.alg.dispatch_gp(three_body, agg, gp_out, self.active_grades)
                out = out + self.w4(four_body)

        return out


# ============================================================
# Update Function
# ============================================================


class CliffordUpdateFunction(nn.Module):
    """Grade-aware node update with self-interaction + residual.

    H' = Norm(W_out · σ(W_in · [H ‖ M ‖ self_int]) + H)
    """

    def __init__(
        self,
        n_channels: int,
        input_grades: Tuple[int, ...],
        output_grades: Tuple[int, ...],
        use_self_interaction: bool = True,
    ):
        super().__init__()
        self.use_self_interaction = use_self_interaction

        cat_mult = 3 if use_self_interaction else 2
        self.linear_in = CliffordIPLinear(
            cat_mult * n_channels, n_channels, bias=True, active_grades=output_grades
        )
        self.activation = CliffordIPGateActivation(
            n_channels, active_grades=output_grades
        )
        self.linear_out = CliffordIPLinear(
            n_channels, n_channels, bias=True, active_grades=output_grades
        )
        self.norm = CliffordIPNorm(n_channels, active_grades=output_grades)

        if use_self_interaction:
            self.self_int = CliffordSelfInteraction(
                n_channels, active_grades=output_grades
            )

    def forward(self, h: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        if self.use_self_interaction:
            si = self.self_int(h)
            combined = torch.cat([h, agg, si], dim=-2)  # (N, 3C, 8)
        else:
            combined = torch.cat([h, agg], dim=-2)  # (N, 2C, 8)

        out = self.linear_in(combined)
        out = self.activation(out)
        out = self.linear_out(out)

        # Residual: pad h to match output grades if needed
        if h.shape[-2] == out.shape[-2]:
            return self.norm(out + h)
        else:
            return self.norm(out)


# ============================================================
# Interaction Block
# ============================================================


class CliffordInteractionBlock(nn.Module):
    """Single grade-aware Clifford MP layer with all improvements.

    Features:
      - Grade-sparse GP dispatch
      - Equivariant attention (optional)
      - Self-interaction
      - Multi-body interaction (MACE-style)
    """

    def __init__(
        self,
        n_channels: int,
        n_rbf: int,
        edge_grades: Tuple[int, ...],
        node_input_grades: Tuple[int, ...],
        node_output_grades: Tuple[int, ...],
        use_attention: bool = True,
        use_self_interaction: bool = True,
        max_body_order: int = 3,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_channels = n_channels

        gp_out = compute_gp_output_grades(node_input_grades, edge_grades)

        self.message_fn = CliffordMessageFunction(
            n_channels,
            n_rbf,
            edge_grades=edge_grades,
            node_input_grades=node_input_grades,
            gp_output_grades=gp_out,
            use_attention=use_attention,
            n_heads=n_heads,
        )

        union_grades = tuple(sorted(set(node_input_grades) | set(gp_out)))

        # Multi-body interaction on aggregated messages
        self.multi_body = CliffordMultiBodyInteraction(
            n_channels,
            active_grades=union_grades,
            max_body_order=max_body_order,
        )

        self.update_fn = CliffordUpdateFunction(
            n_channels,
            input_grades=union_grades,
            output_grades=node_output_grades,
            use_self_interaction=use_self_interaction,
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mv: torch.Tensor,
        dist: torch.Tensor,
        rbf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        h_j = h[src]
        h_i = h[dst]
        N = h.shape[0]

        messages = self.message_fn(h_j, h_i, edge_mv, dist, dst, N, rbf=rbf)

        # Aggregate
        agg = scatter(messages, dst, dim=0, dim_size=N, reduce="sum")

        # Multi-body interaction
        agg = self.multi_body(agg)

        return self.update_fn(h, agg)


# ============================================================
# Output Block with Multi-Scale Readout
# ============================================================


class CliffordOutputBlock(nn.Module):
    """Energy/force output with GP readout mixing + multi-scale.

    GP self-interaction routes all grades into grades 0 and 1.
    Multi-scale: collects per-layer energy contributions (MACE-style).

    When use_gp_readout=False, skips GP mixing and uses raw node features only.
    """

    def __init__(self, n_channels: int = 128, n_hidden: int = 64, use_gp_readout: bool = True):
        super().__init__()
        self.use_gp_readout = use_gp_readout
        self.alg = CliffordAlgebra()

        self.pre_mix = CliffordIPLinear(
            n_channels, n_channels, bias=False, active_grades=ALL_GRADES
        )

        head_in = 2 * n_channels if use_gp_readout else n_channels

        self.energy_head = nn.Sequential(
            nn.Linear(head_in, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1),
        )

        self.force_head = nn.Linear(head_in, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):
        if self.use_gp_readout:
            # GP readout mixing
            h_pre = self.pre_mix(h)
            h_mixed = self.alg.geometric_product(h_pre, h)

            # Energy: grade-0 scalars
            s_orig = h[..., 0]
            s_mixed = h_mixed[..., 0]
            scalars = torch.cat([s_orig, s_mixed], dim=-1)

            # Forces: grade-1 vectors
            v_orig = h[..., 1:4]
            v_mixed = h_mixed[..., 1:4]
            vectors = torch.cat([v_orig, v_mixed], dim=-2)
        else:
            # No GP mixing — use raw features only
            scalars = h[..., 0]
            vectors = h[..., 1:4]

        atom_energy = self.energy_head(scalars).squeeze(-1)

        if batch is not None:
            n_mol = batch.max().item() + 1
            energy = scatter(atom_energy, batch, dim=0, dim_size=n_mol, reduce="sum")
        else:
            energy = atom_energy.sum(dim=0, keepdim=True)

        forces = self.force_head(vectors.transpose(-1, -2)).squeeze(-1)

        return energy, forces


class PerLayerEnergyReadout(nn.Module):
    """Lightweight per-layer energy head for multi-scale readout."""

    def __init__(self, n_channels: int, n_hidden: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(n_channels, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, h: torch.Tensor, batch: Optional[torch.Tensor] = None):
        atom_e = self.head(h[..., 0]).squeeze(-1)  # grade-0 → energy
        if batch is not None:
            n_mol = batch.max().item() + 1
            return scatter(atom_e, batch, dim=0, dim_size=n_mol, reduce="sum")
        return atom_e.sum(dim=0, keepdim=True)


# ============================================================
# Full Model
# ============================================================


class CliffordNet(nn.Module):
    """CliffordIP equivariant GNN for molecular energy and force prediction.

    Architecture:
        1. Atom embed → (N, C, 8) scalar-only multivectors
        2. Edge embed → (E, C, 8) grades (0, 1) + L=2 invariant features
        3. N interaction blocks with progressive grade activation,
           grade-sparse GP dispatch, equivariant attention,
           self-interaction, multi-body (MACE-style), per-layer readout
        4. Output: GP readout mixing → energy + forces
    """

    def __init__(
        self,
        n_atom_types: int = 100,
        n_channels: int = 128,
        n_interactions: int = 5,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        n_hidden_output: int = 64,
        max_neighbors: int = 50,
        direct_forces: bool = True,
        use_attention: bool = True,
        use_self_interaction: bool = True,
        max_body_order: int = 3,
        use_l2: bool = True,
        use_multiscale: bool = True,
        use_gp_readout: bool = True,
        n_heads: int = 4,
        # DeNS
        use_dens: bool = False,
        dens_noise_std: float = 0.01,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.direct_forces = direct_forces
        self.n_channels = n_channels
        self.use_multiscale = use_multiscale
        self.use_dens = use_dens
        self.dens_noise_std = dens_noise_std

        self.atom_embed = CliffordAtomEmbedding(n_atom_types, n_channels)
        self.edge_embed = CliffordEdgeEmbedding(
            n_rbf, n_channels, cutoff, use_l2=use_l2
        )

        # Grade schedule
        edge_grades = (0, 1)
        layer_output_grades = compute_layer_grades(n_interactions, edge_grades)

        # Interaction blocks
        self.interactions = nn.ModuleList()
        node_grades: Tuple[int, ...] = (0,)
        for i in range(n_interactions):
            out_grades = layer_output_grades[i]
            self.interactions.append(
                CliffordInteractionBlock(
                    n_channels,
                    n_rbf,
                    edge_grades=edge_grades,
                    node_input_grades=node_grades,
                    node_output_grades=out_grades,
                    use_attention=use_attention,
                    use_self_interaction=use_self_interaction,
                    max_body_order=max_body_order,
                    n_heads=n_heads,
                )
            )
            node_grades = out_grades

        # Multi-scale per-layer readout
        if use_multiscale:
            self.layer_readouts = nn.ModuleList([
                PerLayerEnergyReadout(n_channels, n_hidden_output)
                for _ in range(n_interactions)
            ])

        self.output = CliffordOutputBlock(n_channels, n_hidden_output, use_gp_readout=use_gp_readout)

        self._grade_schedule = layer_output_grades

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):
        if not self.direct_forces:
            pos = pos.clone().requires_grad_(True)

        src, dst = edge_index
        rel_pos = pos[dst] - pos[src]
        dist = torch.sqrt(torch.sum(rel_pos ** 2, dim=-1) + 1e-8)
        direction = rel_pos / (dist.unsqueeze(-1) + 1e-8)

        h = self.atom_embed(atomic_numbers)
        rbf = self.edge_embed.rbf(dist)  # compute RBF once, pass to all layers
        edge_mv = self.edge_embed(dist, direction)

        multiscale_energy = None
        for i, interaction in enumerate(self.interactions):
            h = interaction(h, edge_index, edge_mv, dist, rbf=rbf)
            if self.use_multiscale:
                le = self.layer_readouts[i](h, batch)
                multiscale_energy = le if multiscale_energy is None else multiscale_energy + le

        energy, forces = self.output(h, batch)

        if multiscale_energy is not None:
            energy = energy + multiscale_energy

        if not self.direct_forces:
            forces = -torch.autograd.grad(
                energy.sum(),
                pos,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]

        return energy, forces

    def forward_with_dens(
        self,
        atomic_numbers: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):
        """Forward with DeNS denoising auxiliary objective.

        During training, also runs on noise-perturbed positions
        and returns the denoising loss.
        """
        # Clean forward pass
        energy, forces = self.forward(atomic_numbers, pos, edge_index, batch)

        dens_loss = torch.tensor(0.0, device=pos.device)
        if self.training and self.use_dens:
            noise = torch.randn_like(pos) * self.dens_noise_std
            pos_noisy = pos + noise

            # Recompute edges for noisy positions
            rel_pos_n = pos_noisy[edge_index[1]] - pos_noisy[edge_index[0]]
            dist_n = torch.sqrt(torch.sum(rel_pos_n ** 2, dim=-1) + 1e-8)
            direction_n = rel_pos_n / (dist_n.unsqueeze(-1) + 1e-8)

            h_n = self.atom_embed(atomic_numbers)
            rbf_n = self.edge_embed.rbf(dist_n)
            edge_mv_n = self.edge_embed(dist_n, direction_n)

            for interaction in self.interactions:
                h_n = interaction(h_n, edge_index, edge_mv_n, dist_n, rbf=rbf_n)

            # Predict noise from grade-1 (vector) features
            noise_pred = h_n[..., 1:4].mean(dim=-2)  # (N, 3)
            dens_loss = F.mse_loss(noise_pred, -noise)

        return energy, forces, dens_loss

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_grade_schedule(self):
        print("Grade activation schedule:")
        print("  Edge:  (0, 1)")
        print("  Atoms: (0,)")
        for i, g in enumerate(self._grade_schedule):
            print(f"  Layer {i}: {g}")


# Public alias — CliffordNet is the implementation, CliffordIP is the exported name
CliffordIP = CliffordNet


# ============================================================
# Smoke Test
# ============================================================


def smoke_test():
    import time

    print("=" * 60)
    print("CliffordIP Smoke Test")
    print("=" * 60)

    device = "cpu"
    torch.manual_seed(42)

    N, C = 10, 64
    n_interactions = 5

    atomic_numbers = torch.randint(1, 50, (N,), device=device)
    pos = torch.randn(N, 3, device=device) * 2.0
    batch = torch.zeros(N, dtype=torch.long, device=device)

    cutoff = 5.0
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j and (pos[i] - pos[j]).norm() < cutoff:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

    print(f"\nAtoms: {N}, Edges: {edge_index.shape[1]}, Channels: {C}")

    model = CliffordNet(
        n_channels=C,
        n_interactions=n_interactions,
        cutoff=cutoff,
        direct_forces=True,
        use_attention=True,
        use_self_interaction=True,
        max_body_order=3,
        use_l2=True,
        use_multiscale=True,
        use_dens=True,
    ).to(device)

    print(f"Parameters: {model.num_params:,}")
    model.print_grade_schedule()

    # Forward
    t0 = time.time()
    energy, forces = model(atomic_numbers, pos, edge_index, batch)
    t1 = time.time()

    print(f"\nForward pass: {(t1 - t0) * 1000:.1f} ms")
    print(f"Energy: {energy.item():.6f}")
    print(f"Forces shape: {forces.shape}, norm: {forces.norm():.6f}")

    # Forward with DeNS
    model.train()
    energy, forces, dens_loss = model.forward_with_dens(
        atomic_numbers, pos, edge_index, batch
    )
    print(f"DeNS loss: {dens_loss.item():.6f}")

    # Equivariance
    print("\n--- Equivariance Test ---")
    model.eval()
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1

    with torch.no_grad():
        e1, f1 = model(atomic_numbers, pos, edge_index, batch)
        pos_rot = pos @ Q.T
        sr, dr = [], []
        for i in range(N):
            for j in range(N):
                if i != j and (pos_rot[i] - pos_rot[j]).norm() < cutoff:
                    sr.append(i)
                    dr.append(j)
        ei_rot = torch.tensor([sr, dr], dtype=torch.long, device=device)
        e2, f2 = model(atomic_numbers, pos_rot, ei_rot, batch)

    e_err = (e1 - e2).abs().item()
    f_err = (f2 - f1 @ Q.T).abs().max().item()
    e_pass = e_err < 1e-3  # slightly looser for attention
    f_pass = f_err < 1e-3
    print(f"Energy invariance error: {e_err:.2e}  {'✓' if e_pass else '✗'}")
    print(f"Force equivariance error: {f_err:.2e}  {'✓' if f_pass else '✗'}")

    # Gradients
    print("\n--- Gradient Test ---")
    model.train()
    model.zero_grad()
    energy, forces = model(atomic_numbers, pos, edge_index, batch)
    loss = energy.sum() + forces.norm()
    loss.backward()

    total, active, dead = 0, 0, []
    for name, p in model.named_parameters():
        if p.requires_grad:
            total += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                active += 1
            else:
                dead.append(name)

    grad_pass = active == total
    print(f"Parameters with gradients: {active}/{total}")
    if dead:
        print("Missing gradients:")
        for n in dead[:15]:
            print(f"  - {n}")
    print(f"All gradients present: {'✓' if grad_pass else '✗'}")

    print("\n" + "=" * 60)
    all_pass = e_pass and f_pass and grad_pass
    print(f"{'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    smoke_test()
