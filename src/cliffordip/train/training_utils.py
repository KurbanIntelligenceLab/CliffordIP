"""
Shared training utilities for all dataset trainers.

Provides standardized functions for seeding, device resolution, output
directory creation, checkpointing, logging, model forwarding, force
computation via autograd, and task-dispatched loss/evaluation.
"""

import copy
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch_scatter import scatter_max as _scatter_max
from tqdm import tqdm

# Module-level thread pool for async checkpoint saving (1 writer thread)
_CKPT_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_CKPT_FUTURE = None

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve device from config. ``'auto'`` picks cuda if available."""
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------


def build_output_dirs(
    cfg: DictConfig,
    extra_parts: Optional[List[str]] = None,
) -> Tuple[str, str, str]:
    """
    Create and return ``(base_dir, model_dir, logs_dir)``.

    Structure::

        {output_root}/{dataset_name}/{model_name}/{target}/{seed}/[extra_parts...]
            models/
            logs/

    Parameters
    ----------
    cfg : DictConfig
        Merged config with ``output_root``, ``dataset.name``, ``model.name``,
        ``dataset.target_name`` (or ``dataset.task_type``), and ``seed``.
    extra_parts : list[str] or None
        Additional path segments (e.g. fold number, molecule name).
    """
    parts = [
        cfg.get("output_root", "results"),
        cfg.model.name,
        cfg.dataset.name,
    ]

    # Target or task type as sub-directory
    target = cfg.dataset.get("target_name", cfg.dataset.get("task_type", "default"))
    parts.append(str(target))
    parts.append(str(cfg.seed))

    if extra_parts:
        parts.extend([str(p) for p in extra_parts])

    base_dir = os.path.join(*parts)
    model_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return base_dir, model_dir, logs_dir


# ---------------------------------------------------------------------------
# Checkpointing & logging
# ---------------------------------------------------------------------------


def _save_ckpt_to_disk(ckpt, path):
    """Write checkpoint dict to disk (runs in background thread)."""
    torch.save(ckpt, path)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    epoch: int = 0,
    metrics: Optional[dict] = None,
    cfg: Optional[DictConfig] = None,
    scheduler=None,
    scaler=None,
    ema_model=None,
    training_state: Optional[dict] = None,
):
    """Save a standardized checkpoint dict to *path* asynchronously.

    Clones state dicts to CPU and submits the actual write to a background
    thread so training can continue without blocking on disk I/O.

    Parameters
    ----------
    training_state : optional dict
        Resumable training state: best_val, best_epoch, es_counter, logs.
    """
    global _CKPT_FUTURE
    # Wait for any previous save to finish before starting a new one
    if _CKPT_FUTURE is not None:
        _CKPT_FUTURE.result()

    ckpt = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
    }
    # Move tensors to CPU to avoid holding GPU memory in background thread
    ckpt["model_state_dict"] = {k: v.cpu() for k, v in ckpt["model_state_dict"].items()}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = copy.deepcopy(optimizer.state_dict())
    if metrics is not None:
        ckpt["metrics"] = metrics
    if cfg is not None:
        ckpt["config"] = OmegaConf.to_container(cfg, resolve=True)
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = copy.deepcopy(scheduler.state_dict())
    if scaler is not None:
        ckpt["scaler_state_dict"] = copy.deepcopy(scaler.state_dict())
    if ema_model is not None:
        ckpt["ema_state_dict"] = copy.deepcopy(ema_model.state_dict())
        ckpt["ema_state_dict"] = {k: v.cpu() for k, v in ckpt["ema_state_dict"].items()}
    if training_state is not None:
        ckpt["training_state"] = copy.deepcopy(training_state)

    _CKPT_FUTURE = _CKPT_EXECUTOR.submit(_save_ckpt_to_disk, ckpt, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    scaler=None, ema_model=None, device=None):
    """Load a checkpoint and restore all training state.

    Returns
    -------
    dict or None
        The training_state dict (best_val, best_epoch, es_counter, logs)
        if present in the checkpoint, else None.
        Also returns the epoch number.
    """
    import warnings
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    sd, grade_migrated = _migrate_per_grade_linear(sd)
    sd, l2_migrated = _migrate_l2_edge_embed(sd)

    # Drop checkpoint keys not present in the model (e.g. higher-grade weights
    # from layers that now use a narrower progressive grade schedule).
    model_keys = set(model.state_dict().keys())
    extra = [k for k in sd if k not in model_keys]
    missing = [k for k in model_keys if k not in sd]
    if extra:
        for k in extra:
            del sd[k]
    if grade_migrated:
        warnings.warn(
            f"Checkpoint {path!r} uses the legacy CliffordIPLinear format "
            "(single weight[8,C,C]). Auto-migrated to per-grade w0/w1/w2/w3.",
            UserWarning, stacklevel=2,
        )
    if l2_migrated:
        warnings.warn(
            f"Checkpoint {path!r}: edge embedding weights truncated from "
            "[C, n_rbf+5] to [C, n_rbf] (legacy use_l2=True format).",
            UserWarning, stacklevel=2,
        )
    if extra:
        warnings.warn(
            f"Checkpoint {path!r}: dropped {len(extra)} keys not present in the "
            f"current model (grade-schedule narrowing). Dropped: {extra[:5]}{'...' if len(extra) > 5 else ''}",
            UserWarning, stacklevel=2,
        )
    if missing:
        warnings.warn(
            f"Checkpoint {path!r}: {len(missing)} model keys have no matching "
            f"checkpoint weight and will use random init: {missing[:5]}{'...' if len(missing) > 5 else ''}",
            UserWarning, stacklevel=2,
        )
    model.load_state_dict(sd, strict=not bool(missing))
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ema_model is not None and "ema_state_dict" in ckpt:
        ema_model.load_state_dict(ckpt["ema_state_dict"])
    epoch = ckpt.get("epoch", 0)
    training_state = ckpt.get("training_state", None)
    return epoch, training_state


def _migrate_per_grade_linear(sd: dict) -> tuple[dict, bool]:
    """Convert legacy CliffordIPLinear ``weight`` tensors to per-grade format.

    Legacy: single ``weight`` of shape ``[8, C_out, C_in]`` + ``grade_mask`` buffer.
    Current: four tensors ``w0, w1, w2, w3`` each of shape ``[C_out, C_in]``.

    Returns the (possibly modified) state dict and a bool indicating whether
    any migration was applied.
    """
    new_sd: dict = {}
    migrated = False
    for key, val in sd.items():
        if key.endswith(".weight") and val.dim() == 3 and val.shape[0] == 8:
            prefix = key[: -len(".weight")]
            new_sd[f"{prefix}.w0"] = val[0]
            new_sd[f"{prefix}.w1"] = val[1:4].mean(0)
            new_sd[f"{prefix}.w2"] = val[4:7].mean(0)
            new_sd[f"{prefix}.w3"] = val[7]
            migrated = True
        elif key.endswith(".grade_mask"):
            pass  # buffer removed in new model
        else:
            new_sd[key] = val
    return new_sd, migrated


def _migrate_l2_edge_embed(sd: dict) -> tuple[dict, bool]:
    """Truncate legacy edge embedding weights from shape [C, n_rbf+5] to [C, n_rbf].

    Early checkpoints were saved with ``use_l2=True``, which appended 5 extra
    columns to the first linear layer of ``scalar_net`` and ``vector_net``.
    The current architecture uses only the n_rbf columns.

    Returns the (possibly modified) state dict and a bool indicating whether
    any migration was applied.
    """
    # Detect the RBF size from the stored offsets buffer
    rbf_key = next((k for k in sd if k.endswith("edge_embed.rbf.offsets")), None)
    if rbf_key is None:
        return sd, False

    n_rbf = sd[rbf_key].shape[0]
    migrated = False
    for net in ("scalar_net", "vector_net"):
        w_key = rbf_key.replace("edge_embed.rbf.offsets", f"edge_embed.{net}.0.weight")
        if w_key in sd and sd[w_key].shape[1] == n_rbf + 5:
            sd[w_key] = sd[w_key][:, :n_rbf].contiguous()
            migrated = True
    return sd, migrated


def migrate_checkpoint(old_path: str, new_path: str | None = None) -> dict:
    """Convert legacy checkpoint formats to the current format.

    Handles two migrations (applied in order, both idempotent):

    1. **Per-grade CliffordIPLinear weights** — legacy single ``weight`` tensor
       of shape ``[8, C_out, C_in]`` → four tensors ``w0, w1, w2, w3`` each
       of shape ``[C_out, C_in]``.  Stale ``grade_mask`` buffers are dropped.

    2. **L=2 edge embedding columns** — early checkpoints have
       ``edge_embed.scalar_net.0.weight`` / ``vector_net.0.weight`` with
       shape ``[C, n_rbf+5]``.  The last 5 columns are stripped to give
       ``[C, n_rbf]``.

    Parameters
    ----------
    old_path : str
        Path to the legacy checkpoint file.
    new_path : str or None
        If given, save the converted checkpoint to this path.

    Returns
    -------
    dict
        The converted checkpoint (same structure, updated model_state_dict).
    """
    ckpt = torch.load(old_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    sd, _ = _migrate_per_grade_linear(sd)
    sd, _ = _migrate_l2_edge_embed(sd)
    ckpt["model_state_dict"] = sd
    if new_path is not None:
        torch.save(ckpt, new_path)
    return ckpt


def wait_for_checkpoint():
    """Block until the last async checkpoint save completes."""
    global _CKPT_FUTURE
    if _CKPT_FUTURE is not None:
        _CKPT_FUTURE.result()
        _CKPT_FUTURE = None


def save_logs(
    path: str,
    logs: dict,
    cfg: Optional[DictConfig] = None,
):
    """Save standardised JSON logs with the full resolved config embedded."""
    output = dict(logs)
    if cfg is not None:
        output["config"] = OmegaConf.to_container(cfg, resolve=True)
    output["timestamp"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(output, f, indent=4)


# ---------------------------------------------------------------------------
# Weights & Biases (optional)
# ---------------------------------------------------------------------------


def init_wandb(
    cfg: DictConfig,
    extra_tags: Optional[List[str]] = None,
):
    """Initialise a W&B run if ``cfg.logging.wandb.enabled`` is true.

    Returns the ``wandb.Run`` object, or ``None`` when disabled.
    """
    wandb_cfg = cfg.logging.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    import wandb

    tags = list(wandb_cfg.get("tags", []))
    if extra_tags:
        tags.extend(extra_tags)

    run = wandb.init(
        project=wandb_cfg.get("project", "mlip-bench"),
        entity=wandb_cfg.get("entity", None),
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=tags or None,
        name=f"{cfg.model.name}_{cfg.dataset.name}",
        reinit=True,
    )
    return run


def log_wandb(run, metrics: dict, step: int):
    """Log *metrics* to an active W&B run.  No-op when *run* is ``None``."""
    if run is None:
        return
    run.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Model forward helpers
# ---------------------------------------------------------------------------


def forward_model(
    model: nn.Module,
    data,
    interface: str,
):
    """
    Call *model* with the correct interface.

    Parameters
    ----------
    model : nn.Module
    data : PyG Data / Batch
    interface : str
        ``"pyg_native"`` -> ``model(data.z, data.pos, data.batch)``
        ``"data_wrapper"`` -> ``model(data)``

    Returns
    -------
    Energy tensor [num_graphs], or tuple (energy, forces) when model returns both
    (e.g. PaiNN with direct_forces).
    """
    if interface == "pyg_native":
        return model(data.z, data.pos, data.batch).view(-1)
    elif interface == "data_wrapper":
        out = model(data)
        if isinstance(out, tuple):
            return out
        return out.view(-1)
    else:
        raise ValueError(f"Unknown model interface '{interface}'. Use 'pyg_native' or 'data_wrapper'.")


def compute_forces(
    energy: torch.Tensor,
    pos: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Compute forces as ``-dE/dR`` via autograd.

    Parameters
    ----------
    energy : Tensor [num_graphs]
    pos : Tensor [num_atoms, 3]  (must have ``requires_grad=True``)
    create_graph : bool
        ``True`` during training, ``False`` during evaluation.
    """
    # Force autograd to run in fp32 even under AMP — half-precision gradients
    # are numerically unstable for force computation.
    with torch.amp.autocast("cuda", enabled=False):
        energy_f32 = energy.float()
        pos_f32 = pos.float()
        grad_outputs = torch.autograd.grad(
            energy_f32.sum(),
            pos,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,  # isolated atoms may not contribute to energy
        )[0]
    if grad_outputs is None:
        return torch.zeros_like(pos)
    return -grad_outputs


# ---------------------------------------------------------------------------
# S2EF-specific loss helpers (FairChem-aligned)
# ---------------------------------------------------------------------------


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
    runtime_stats: Optional[dict] = None,
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
        out = forward_model(model, data, interface)
        if isinstance(out, tuple):
            energy_pred, forces_pred = out
        else:
            energy_pred = out
            forces_pred = compute_forces(energy_pred, data.pos, create_graph=training)

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
        raise ValueError(f"Unknown task_type '{task_type}'. Use 'scalar', 'energy_forces', 's2ef', or 'is2re'.")


# ---------------------------------------------------------------------------
# Task-dispatched evaluation
# ---------------------------------------------------------------------------


def evaluate_epoch(
    model: nn.Module,
    loader,
    cfg: DictConfig,
    device: torch.device,
    runtime_stats: Optional[dict] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """Evaluate model on a loader, returning task-appropriate metrics.

    Returns a dict with keys depending on task_type:
        scalar:                {"mse": ..., "mae": ...}
        energy_forces / s2ef:  {"energy_mae": ..., "force_mae": ..., "force_cos": ..., "efwt": ...}
        is2re:                 {"mae": ..., "ewt": ...}

    Note: energy_forces/s2ef tasks need gradients for autograd force computation,
    so we use torch.no_grad() selectively for tasks that don't need it.

    Parameters
    ----------
    amp_dtype : optional torch.dtype
        If not None, wrap forward passes with torch.amp.autocast('cuda', dtype=amp_dtype).
    """
    model.eval()
    interface = cfg.model.interface
    task_type = cfg.dataset.get("task_type", "scalar")
    use_amp = amp_dtype is not None and device.type == "cuda"

    if task_type == "scalar":
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype or torch.float32):
            return _eval_scalar(model, loader, interface, device, runtime_stats)
    elif task_type in ("energy_forces", "s2ef"):
        return _eval_energy_forces(model, loader, interface, device, cfg, amp_dtype=amp_dtype)
    elif task_type == "is2re":
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype or torch.float32):
            return _eval_is2re(model, loader, interface, device, cfg)
    else:
        raise ValueError(f"Unknown task_type '{task_type}'.")


def _eval_scalar(model, loader, interface, device, runtime_stats):
    total_mse = 0.0
    total_mae = 0.0
    n = 0

    has_zscore = runtime_stats and "mean" in runtime_stats
    if has_zscore:
        mean = runtime_stats["mean"].to(device)
        std = runtime_stats["std"].to(device)
        target_idx = runtime_stats["target_idx"]

    for data in tqdm(loader, desc="  Val", unit="batch", leave=False):
        data = data.to(device)
        pred_norm = forward_model(model, data, interface)
        if isinstance(pred_norm, tuple):
            pred_norm = pred_norm[0]

        if has_zscore:
            y = data.y[:, target_idx].view(-1)
            y_norm = (y - mean) / std
            total_mse += F.mse_loss(pred_norm, y_norm, reduction="sum").item()
            pred = pred_norm * std + mean
            total_mae += F.l1_loss(pred, y, reduction="sum").item()
        else:
            y = data.y.view(-1)
            total_mse += F.mse_loss(pred_norm, y, reduction="sum").item()
            total_mae += (pred_norm - y).abs().sum().item()

        n += data.num_graphs

    n = max(1, n)
    return {"mse": total_mse / n, "mae": total_mae / n}


def _eval_energy_forces(model, loader, interface, device, cfg, amp_dtype=None):
    eval_on_free = cfg.dataset.get("eval_on_free_atoms", False)
    use_amp = amp_dtype is not None and device.type == "cuda"
    total_energy_ae = 0.0
    total_force_ae = 0.0
    total_cos_sim = 0.0
    n_structures = 0
    n_force_components = 0
    n_force_atoms = 0
    n_efwt = 0

    for data in tqdm(loader, desc="  Val", unit="batch", leave=False):
        data = data.to(device)
        data.pos.requires_grad_(True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype or torch.float32):
            out = forward_model(model, data, interface)
        if isinstance(out, tuple):
            energy_pred, forces_pred = out
        else:
            energy_pred = out
            forces_pred = compute_forces(energy_pred, data.pos, create_graph=False)

        energy_target = data.energy.view(-1) if hasattr(data, "energy") else data.y.view(-1)
        forces_target = data.force.to(forces_pred.dtype)

        # Free-atom filtering for evaluation
        free_mask = None
        if eval_on_free:
            free_mask = _get_free_atom_mask(data)

        if free_mask is not None:
            fp = forces_pred[free_mask]
            ft = forces_target[free_mask]
            batch_free = data.batch[free_mask]
        else:
            fp = forces_pred
            ft = forces_target
            batch_free = data.batch

        # Energy MAE (per-structure)
        energy_ae = (energy_pred - energy_target).abs()
        total_energy_ae += energy_ae.sum().item()

        # Force MAE (component-wise, averaged over atoms)
        total_force_ae += (fp - ft).abs().sum().item()
        n_force_components += ft.numel()  # 3 components per atom

        # Force cosine similarity (per-atom, dim=-1)
        cos = torch.cosine_similarity(fp, ft, dim=-1)
        total_cos_sim += cos.sum().item()
        n_force_atoms += cos.numel()

        # EFwT: per-structure (energy < 0.02 eV AND max force error < 0.03 eV/A)
        per_atom_ferr = (fp - ft).norm(dim=-1)  # L2 norm per atom
        n_graphs = energy_pred.size(0)
        max_ferr_per_graph, _ = _scatter_max(per_atom_ferr, batch_free, dim=0, dim_size=n_graphs)
        n_efwt += ((energy_ae < 0.02) & (max_ferr_per_graph < 0.03)).sum().item()

        n_structures += n_graphs

    n = max(1, n_structures)
    nf = max(1, n_force_components)
    na = max(1, n_force_atoms)
    return {
        "energy_mae": total_energy_ae / n,
        "force_mae": total_force_ae / nf,
        "force_cos": total_cos_sim / na,
        "efwt": n_efwt / n * 100.0,
    }


def _eval_is2re(model, loader, interface, device, cfg):
    ewt_threshold = cfg.dataset.get("ewt_threshold", 0.02)
    total_ae = 0.0
    n_within = 0
    n_total = 0

    for data in tqdm(loader, desc="  Val", unit="batch", leave=False):
        data = data.to(device)
        # Some models (NequIP, etc.) compute forces via autograd during forward and need grad enabled
        with torch.enable_grad():
            pred = forward_model(model, data, interface)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred.detach()
        target = data.y.view(-1)

        ae = (pred - target).abs()
        total_ae += ae.sum().item()
        n_within += (ae <= ewt_threshold).sum().item()
        n_total += target.numel()

    n = max(1, n_total)
    return {"mae": total_ae / n, "ewt": n_within / n * 100.0}


# ---------------------------------------------------------------------------
# Multi-split evaluation (OC20/OC22 benchmark)
# ---------------------------------------------------------------------------

# Catalysis subset dataset name → builder function mapping
_CATALYSIS_SUBSET_BUILDERS = {
    "oc20_s2ef_co2rr": "cliffordip.data.oc20_co2rr.build_oc20_co2rr_dataset",
    "oc20_s2ef_nrr": "cliffordip.data.oc20_nrr.build_oc20_nrr_dataset",
    "oc20_s2ef_c2": "cliffordip.data.oc20_c2.build_oc20_c2_dataset",
}


def _build_eval_dataset(ds_name, data_root, split, max_samples, oc22, cfg):
    """Build the right dataset for evaluation — catalysis subsets use filtered builders."""
    if ds_name in _CATALYSIS_SUBSET_BUILDERS:
        import importlib

        mod_path, func_name = _CATALYSIS_SUBSET_BUILDERS[ds_name].rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        builder = getattr(mod, func_name)
        return builder(root=data_root, split=split, max_samples=max_samples)

    from cliffordip.data.oc20_dataloader import OC20LMDBDataset

    return OC20LMDBDataset(
        root=data_root,
        task=cfg.dataset.get("task", cfg.dataset.get("task_type", "s2ef")),
        split=split,
        max_samples=max_samples,
        oc22=oc22,
    )


def evaluate_multi_split(model, cfg, device, splits, runtime_stats=None, eval_fraction=None):
    """Evaluate model on multiple val/test splits.

    Parameters
    ----------
    model : nn.Module
    cfg : DictConfig
    device : torch.device
    splits : list[str]
        Split names, e.g. ["val_id", "val_ood_ads", "val_ood_cat", "val_ood_both"].
    runtime_stats : optional dict for z-score normalization
    eval_fraction : float, optional
        Fraction of each split to use (0.0–1.0). None or 1.0 = full split.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{split_name: metrics_dict}``.
    """
    from cliffordip.train.dataset_registry import build_loader
    from cliffordip.data.oc20_dataloader import OC20LMDBDataset

    results = {}
    task_type = cfg.dataset.get("task_type", "scalar")
    interface = cfg.model.interface
    oc22 = "oc22" in cfg.dataset.name
    base_max = cfg.dataset.get("max_val_samples", None)
    ds_name = cfg.dataset.get("name", "")

    model.eval()
    for split in splits:
        try:
            max_samples = base_max
            if eval_fraction is not None and 0 < eval_fraction < 1:
                ds_temp = OC20LMDBDataset(
                    root=cfg.dataset.data_root,
                    task=cfg.dataset.get("task", cfg.dataset.get("task_type", "s2ef")),
                    split=split,
                    max_samples=None,
                    oc22=oc22,
                )
                n = len(ds_temp)
                max_samples = max(1, int(n * eval_fraction))
                if base_max is not None:
                    max_samples = min(max_samples, base_max)
            ds = _build_eval_dataset(ds_name, cfg.dataset.data_root, split, max_samples, oc22, cfg)
        except FileNotFoundError:
            print(f"  Skipping {split} (data not found)")
            continue

        loader = build_loader(ds, cfg, shuffle=False)
        if task_type in ("energy_forces", "s2ef"):
            metrics = _eval_energy_forces(model, loader, interface, device, cfg)
        elif task_type == "is2re":
            with torch.no_grad():
                metrics = _eval_is2re(model, loader, interface, device, cfg)
        else:
            with torch.no_grad():
                metrics = _eval_scalar(model, loader, interface, device, runtime_stats)
        results[split] = metrics
        summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  {split}: {summary}")

    return results


def build_oc20_loaders(cfg: DictConfig, splits: list, eval_fraction=None):
    """Build OC20/OC22 loaders once for reuse across multiple model evals.

    Returns
    -------
    dict[str, DataLoader]
        {split_name: DataLoader}
    """
    from cliffordip.train.dataset_registry import build_loader
    from cliffordip.data.oc20_dataloader import OC20LMDBDataset

    loaders = {}
    task_type = cfg.dataset.get("task_type", "scalar")
    oc22 = "oc22" in cfg.dataset.name
    base_max = cfg.dataset.get("max_val_samples", None)
    task = cfg.dataset.get("task", cfg.dataset.get("task_type", "s2ef"))
    ds_name = cfg.dataset.get("name", "")

    for split in splits:
        try:
            max_samples = base_max
            if eval_fraction is not None and 0 < eval_fraction < 1:
                ds_temp = OC20LMDBDataset(
                    root=cfg.dataset.data_root,
                    task=task,
                    split=split,
                    max_samples=None,
                    oc22=oc22,
                )
                n = len(ds_temp)
                max_samples = max(1, int(n * eval_fraction))
                if base_max is not None:
                    max_samples = min(max_samples, base_max)
            ds = _build_eval_dataset(ds_name, cfg.dataset.data_root, split, max_samples, oc22, cfg)
        except FileNotFoundError:
            print(f"  Skipping {split} (data not found)")
            continue
        loaders[split] = build_loader(ds, cfg, shuffle=False)
    return loaders


def evaluate_with_loaders(model, cfg, device, loaders_by_split, runtime_stats=None):
    """Evaluate model on pre-built loaders. Use for shared-data eval across multiple models.

    loaders_by_split : dict[str, iterable]
        {split_name: DataLoader or list of batches}
    """
    results = {}
    task_type = cfg.dataset.get("task_type", "scalar")
    interface = cfg.model.interface

    model.eval()
    for split, loader_or_batches in loaders_by_split.items():
        if task_type in ("energy_forces", "s2ef"):
            metrics = _eval_energy_forces(model, loader_or_batches, interface, device, cfg)
        elif task_type == "is2re":
            with torch.no_grad():
                metrics = _eval_is2re(model, loader_or_batches, interface, device, cfg)
        else:
            with torch.no_grad():
                metrics = _eval_scalar(model, loader_or_batches, interface, device, runtime_stats)
        results[split] = metrics
        summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  {split}: {summary}")
    return results


# ---------------------------------------------------------------------------
# Optimizer & scheduler builders
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
        raise ValueError(f"Unknown scheduler type '{sched_type}'. Use 'step', 'cosine', or 'plateau'.")


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------


def build_ema(cfg: DictConfig, model: nn.Module) -> Optional[torch.optim.swa_utils.AveragedModel]:
    """Build an EMA shadow model if configured, otherwise return None.

    Uses torch.optim.swa_utils.AveragedModel with exponential decay.
    """
    ema_cfg = cfg.training.get("ema", None)
    if not ema_cfg or not ema_cfg.get("enabled", False):
        return None

    decay = ema_cfg.get("decay", 0.999)

    def ema_avg(avg_param, model_param, num_averaged):
        return decay * avg_param + (1 - decay) * model_param

    return torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
