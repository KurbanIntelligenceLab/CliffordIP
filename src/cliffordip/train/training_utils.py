"""Model forwarding, loss dispatch, optimizer construction and checkpoint migration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def forward_model(model: nn.Module, data, interface: str):
    """Call ``model`` through its interface: ``pyg_native`` or ``data_wrapper``."""
    if interface == "pyg_native":
        return model(data.z, data.pos, data.batch).view(-1)
    elif interface == "data_wrapper":
        out = model(data)
        if isinstance(out, tuple):
            return out
        return out.view(-1)
    else:
        raise ValueError(
            f"Unknown model interface '{interface}'. Use 'pyg_native' or 'data_wrapper'."
        )


def _per_atom_mae_loss(pred, target, natoms_per_graph):
    """PerAtomMAELoss: L1(pred/natoms, target/natoms).

    Matches fairchem/core/modules/loss.py ``DDPLoss("per_atom_mae")``.
    """
    return F.l1_loss(pred / natoms_per_graph, target / natoms_per_graph)


def _l2norm_loss(pred, target):
    """L2NormLoss: mean of per-atom L2 norms of the error vector.

    Matches fairchem/core/modules/loss.py ``DDPLoss("l2norm")``.
    """
    return torch.linalg.vector_norm(pred - target, ord=2, dim=-1).mean()


def _get_free_atom_mask(data):
    """Return boolean mask of free (non-fixed) atoms, or None if unavailable.

    OC20/OC22 convention:
        - ``data.fixed`` (bool tensor): True = fixed atom
        - ``data.tags`` (int tensor): 0 = sub-surface (fixed), 1 = surface, 2 = adsorbate
    """
    if hasattr(data, "fixed") and data.fixed is not None:
        return ~data.fixed.bool()
    if hasattr(data, "tags") and data.tags is not None:
        return data.tags > 0
    return None


# ---------------------------------------------------------------------------
# Task-dispatched loss computation
# ---------------------------------------------------------------------------


def compute_loss(
    model: nn.Module,
    data,
    cfg: DictConfig,
    device: torch.device,
    runtime_stats: dict | None = None,
    training: bool = True,
) -> torch.Tensor:
    """Compute loss based on task_type, dispatching to the correct strategy.

    Parameters
    ----------
    model : nn.Module
    data : PyG Batch (already on device)
    cfg : DictConfig with dataset.task_type, model.interface, training.loss, etc.
    device : torch.device
    runtime_stats : optional dict with z-score mean/std for QM9-style normalization
    training : bool, whether we're in training (affects autograd graph creation)
    """
    interface = cfg.model.interface
    task_type = cfg.dataset.get("task_type", "scalar")
    loss_type = cfg.training.get("loss", "mse")
    loss_fn = F.mse_loss if loss_type == "mse" else F.l1_loss

    if task_type == "scalar":
        pred = forward_model(model, data, interface)
        if isinstance(pred, tuple):
            pred = pred[0]  # energy only
        if runtime_stats and "target_idx" in runtime_stats:
            target = data.y[:, runtime_stats["target_idx"]].view(-1)
        else:
            target = data.y.view(-1)
        # Match target dtype to pred (e.g. NequIP outputs float64)
        target = target.to(pred.dtype)

        if runtime_stats and "mean" in runtime_stats:
            mean = runtime_stats["mean"].to(device)
            std = runtime_stats["std"].to(device)
            target = (target - mean) / std

        return loss_fn(pred, target)

    elif task_type in ("energy_forces", "s2ef"):
        data.pos.requires_grad_(True)
        energy_pred, forces_pred = forward_model(model, data, interface)

        energy_target = data.energy.view(-1) if hasattr(data, "energy") else data.y.view(-1)
        energy_target = energy_target.to(energy_pred.dtype)
        forces_target = data.force.to(forces_pred.dtype)

        # Free-atom filtering (OC20/OC22: fixed atoms have ~zero force)
        if cfg.dataset.get("train_on_free_atoms", False):
            free_mask = _get_free_atom_mask(data)
            if free_mask is not None:
                forces_pred = forces_pred[free_mask]
                forces_target = forces_target[free_mask]

        # Energy loss (configurable: per_atom_mae for S2EF, default fallback)
        e_loss_type = cfg.dataset.get("energy_loss", None)
        if e_loss_type == "per_atom_mae":
            natoms_per_graph = torch.bincount(data.batch).to(energy_pred.dtype)
            loss_e = _per_atom_mae_loss(energy_pred, energy_target, natoms_per_graph)
        else:
            loss_e = loss_fn(energy_pred, energy_target)

        # Force loss (configurable: l2norm for S2EF, default fallback)
        f_loss_type = cfg.dataset.get("force_loss", None)
        if f_loss_type == "l2norm":
            loss_f = _l2norm_loss(forces_pred, forces_target)
        else:
            loss_f = loss_fn(forces_pred, forces_target)

        energy_weight = cfg.dataset.get("energy_weight", 1.0)
        force_weight = cfg.dataset.get("force_weight", 1.0)
        return energy_weight * loss_e + force_weight * loss_f

    elif task_type == "is2re":
        pred = forward_model(model, data, interface)
        if isinstance(pred, tuple):
            pred = pred[0]  # energy only
        target = data.y.view(-1)
        return loss_fn(pred, target)

    else:
        raise ValueError(
            f"Unknown task_type '{task_type}'. Use 'scalar', 'energy_forces', 's2ef', or 'is2re'."
        )


# ---------------------------------------------------------------------------
# Task-dispatched evaluation
# ---------------------------------------------------------------------------


def build_optimizer(cfg: DictConfig, model: nn.Module):
    """Build optimizer from config."""
    lr = cfg.training.lr
    weight_decay = cfg.training.get("weight_decay", 0.0)
    opt_name = cfg.training.get("optimizer", "adam")

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'. Use 'adam' or 'adamw'.")


def build_scheduler(cfg: DictConfig, optimizer):
    """Build LR scheduler from config, or return None if not configured.

    Supported types:
        step:     StepLR with step_size and gamma
        cosine:   CosineAnnealingLR with T_max (defaults to training.epochs)
        plateau:  ReduceLROnPlateau (requires scheduler.step(val_metric) per epoch)
    """
    sched_cfg = cfg.training.get("lr_scheduler", None)
    if not sched_cfg:
        return None

    sched_type = sched_cfg.get("type", None)
    if sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 10),
            gamma=sched_cfg.get("gamma", 0.8),
        )
    elif sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", cfg.training.epochs),
            eta_min=sched_cfg.get("eta_min", 0.0),
        )
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 10),
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )
    else:
        raise ValueError(
            f"Unknown scheduler type '{sched_type}'. Use 'step', 'cosine', or 'plateau'."
        )


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------
