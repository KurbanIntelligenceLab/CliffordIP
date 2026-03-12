"""
CliffordIP training wrapper — data_wrapper interface for MLIP benchmarking.

Provides torch.compile, EMA, DeNS auxiliary loss, and multi-GPU support.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.nn.pool import radius_graph

from cliffordip.interaction import CliffordNet


# ============================================================
# Exponential Moving Average
# ============================================================


class ExponentialMovingAverage:
    """EMA of model parameters. ~2-5% accuracy gain for free.

    Usage:
        ema = ExponentialMovingAverage(model, decay=0.999)
        # During training:
        ema.update()
        # During eval:
        ema.apply_shadow()  # swap to EMA weights
        model.eval(); validate()
        ema.restore()       # swap back to training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================
# Main Wrapper
# ============================================================


class CliffordIPWrapper(nn.Module):
    """CliffordIP model wrapper implementing the data_wrapper interface.

    Example:
        model = CliffordIPWrapper(
            n_channels=128,
            n_interactions=5,
            use_attention=True,
            use_self_interaction=True,
            max_body_order=3,
            use_l2=True,
            use_multiscale=True,
            use_compile=True,
        )
        # Training loop:
        energy, forces = model(data)
        # or with DeNS:
        energy, forces, dens_loss = model.forward_with_dens(data)
    """

    def __init__(
        self,
        # Architecture
        n_atom_types: int = 100,
        n_channels: int = 128,
        n_interactions: int = 5,
        n_rbf: int = 20,
        cutoff: float = 5.0,
        n_hidden_output: int = 64,
        max_neighbors: int = 50,
        direct_forces: bool = True,
        # Accuracy features
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
        # Speed features
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_ema = use_ema
        self.use_dens = use_dens

        self._model = CliffordNet(
            n_atom_types=n_atom_types,
            n_channels=n_channels,
            n_interactions=n_interactions,
            n_rbf=n_rbf,
            cutoff=cutoff,
            n_hidden_output=n_hidden_output,
            max_neighbors=max_neighbors,
            direct_forces=direct_forces,
            use_attention=use_attention,
            use_self_interaction=use_self_interaction,
            max_body_order=max_body_order,
            use_l2=use_l2,
            use_multiscale=use_multiscale,
            use_gp_readout=use_gp_readout,
            n_heads=n_heads,
            use_dens=use_dens,
            dens_noise_std=dens_noise_std,
        )

        # torch.compile (applied after model creation, before DDP)
        if use_compile:
            try:
                self._model = torch.compile(
                    self._model, mode=compile_mode, dynamic=True
                )
                print(f"[CliffordIPWrapper] torch.compile enabled (mode={compile_mode})")
            except Exception as e:
                print(f"[CliffordIPWrapper] torch.compile failed: {e}, using eager mode")

        # EMA (initialized after compile so it tracks the right params)
        self._ema = None
        self._ema_decay = ema_decay

    def init_ema(self):
        """Initialize EMA after model is on device. Call after .to(device)."""
        if self.use_ema:
            # Access underlying model if compiled
            model = self._get_raw_model()
            self._ema = ExponentialMovingAverage(model, decay=self._ema_decay)
            print(f"[CliffordIPWrapper] EMA initialized (decay={self._ema_decay})")

    def _get_raw_model(self):
        """Get the raw model (unwrap compile if needed)."""
        m = self._model
        if hasattr(m, '_orig_mod'):
            return m._orig_mod
        return m

    def update_ema(self):
        """Call after each optimizer.step()."""
        if self._ema is not None:
            self._ema.update()

    def apply_ema(self):
        """Swap to EMA weights for evaluation."""
        if self._ema is not None:
            self._ema.apply_shadow()

    def restore_from_ema(self):
        """Swap back to training weights."""
        if self._ema is not None:
            self._ema.restore()

    def _build_edges(self, data):
        """Build radius graph from PyG data."""
        return radius_graph(
            data.pos,
            r=self.cutoff,
            batch=data.batch,
            max_num_neighbors=self.max_neighbors,
        )

    def forward(self, data):
        """Standard forward pass.

        Returns:
            (energy, forces) if direct_forces else energy
            energy: (num_graphs,)
            forces: (num_atoms, 3)
        """
        edge_index = self._build_edges(data)
        energy, forces = self._model(
            data.z, data.pos, edge_index, data.batch
        )
        if self._get_raw_model().direct_forces:
            return energy.view(-1), forces
        return energy.view(-1)

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


# ============================================================
# Training Utilities
# ============================================================


def build_optimizer(
    model: nn.Module,
    lr: float = 2e-4,
    weight_decay: float = 1e-3,
    amsgrad: bool = False,
) -> torch.optim.Optimizer:
    """AdamW optimizer with recommended MLIP hyperparameters."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 1000,
    min_lr: float = 1e-6,
    mode: str = "cosine",
):
    """LR scheduler: linear warmup + cosine decay.

    Args:
        mode: "cosine" for CosineAnnealingLR (large datasets)
              "plateau" for ReduceLROnPlateau (small datasets)
    """
    if mode == "cosine":
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
        )

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr,
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    elif mode == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")


def compute_loss(
    energy_pred: torch.Tensor,
    forces_pred: torch.Tensor,
    energy_target: torch.Tensor,
    forces_target: torch.Tensor,
    energy_weight: float = 4.0,
    force_weight: float = 100.0,
    dens_loss: Optional[torch.Tensor] = None,
    dens_weight: float = 0.1,
    force_loss_type: str = "l2mae",
) -> Dict[str, torch.Tensor]:
    """Compute energy + force loss with recommended weighting.

    Args:
        force_loss_type: "l2mae" (per-atom L2 norm, then mean)
                        or "mae" (component-wise MAE)
    Returns:
        dict with 'total', 'energy', 'forces', and optionally 'dens' keys.
    """
    # Energy MAE
    energy_loss = torch.mean(torch.abs(energy_pred - energy_target))

    # Force loss
    if force_loss_type == "l2mae":
        force_err = torch.sqrt(
            torch.sum((forces_pred - forces_target) ** 2, dim=-1) + 1e-8
        )
        force_loss = torch.mean(force_err)
    else:
        force_loss = torch.mean(torch.abs(forces_pred - forces_target))

    total = energy_weight * energy_loss + force_weight * force_loss

    result = {
        "total": total,
        "energy": energy_loss,
        "forces": force_loss,
    }

    if dens_loss is not None:
        total = total + dens_weight * dens_loss
        result["total"] = total
        result["dens"] = dens_loss

    return result


def two_phase_loss_weights(
    epoch: int, total_epochs: int, phase2_start_fraction: float = 0.8
) -> Dict[str, float]:
    """MACE-style two-phase training: emphasize forces early, energy late.

    Phase 1 (0 to 80%): energy_weight=4, force_weight=100
    Phase 2 (80% to 100%): energy_weight=1000, force_weight=10
    """
    if epoch < int(total_epochs * phase2_start_fraction):
        return {"energy_weight": 4.0, "force_weight": 100.0}
    else:
        return {"energy_weight": 1000.0, "force_weight": 10.0}


# ============================================================
# Quick training example
# ============================================================


def example_training_loop():
    """Pseudocode demonstrating the full training setup."""
    print("=" * 60)
    print("Example training loop (pseudocode)")
    print("=" * 60)

    print("""
    # ---- Model setup ----
    model = CliffordIPWrapper(
        n_channels=128,
        n_interactions=5,
        cutoff=5.0,
        use_attention=True,
        use_self_interaction=True,
        max_body_order=3,
        use_l2=True,
        use_multiscale=True,
        use_compile=True,          # 2-4× speedup
        use_ema=True,              # ~2-5% accuracy gain
        use_dens=True,             # ~5-10% force MAE improvement
    ).cuda()
    model.init_ema()

    optimizer = build_optimizer(model, lr=2e-4, weight_decay=1e-3)
    scheduler = build_scheduler(optimizer, total_steps=200000, warmup_steps=1000)

    # ---- Training with BF16 mixed precision ----
    scaler = torch.amp.GradScaler()

    for epoch in range(total_epochs):
        loss_weights = two_phase_loss_weights(epoch, total_epochs)

        for data in train_loader:
            data = data.cuda()
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if model.use_dens:
                    energy, forces, dens_loss = model.forward_with_dens(data)
                else:
                    energy, forces = model(data)
                    dens_loss = None

                losses = compute_loss(
                    energy, forces,
                    data.energy, data.forces,
                    dens_loss=dens_loss,
                    **loss_weights,
                )

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            model.update_ema()

        # ---- Validation with EMA ----
        model.apply_ema()
        model.eval()
        # ... validate ...
        model.restore_from_ema()
        model.train()
    """)


if __name__ == "__main__":
    example_training_loop()

    print("\n--- Quick model test ---")
    import torch

    model = CliffordIPWrapper(
        n_channels=64,
        n_interactions=3,
        use_attention=True,
        use_self_interaction=True,
        max_body_order=3,
        use_l2=True,
        use_multiscale=True,
        use_compile=False,  # skip compile for quick test
        use_ema=False,
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get the raw model to print grade schedule
    raw = model._get_raw_model()
    raw.print_grade_schedule()