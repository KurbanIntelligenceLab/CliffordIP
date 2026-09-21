"""CliffordIP Lightning Module."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from cliffordip.train.checkpoints import migrate_state_dict
from cliffordip.train.training_utils import (
    _get_free_atom_mask,
    _l2norm_loss,
    _per_atom_mae_loss,
    build_optimizer,
    build_scheduler,
    compute_loss,
    forward_model,
)
from cliffordip.wrapper import CliffordIPWrapper


class CliffordIPLightningModule(L.LightningModule):
    """Lightning module wrapping CliffordIPWrapper for training on any MLIP dataset.

    Handles S2EF, IS2RE, and scalar regression tasks.
    AMP, DDP, gradient clipping, EMA, and checkpointing are all
    delegated to the Lightning Trainer.

    Usage::
        from cliffordip.lightning import CliffordIPLightningModule, CliffordIPDataModule
        from cliffordip.train.config_utils import load_config

        cfg = load_config()
        module = CliffordIPLightningModule(cfg)
        datamodule = CliffordIPDataModule(cfg)
        trainer = L.Trainer(max_epochs=10, precision="bf16-mixed")
        trainer.fit(module, datamodule)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        # Save hyperparameters (serialises cfg as a plain dict for checkpoints)
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})

        self.cfg = cfg
        self.model = CliffordIPWrapper.from_mapping(OmegaConf.to_container(cfg.model, resolve=True))

        self._task_type: str = cfg.dataset.get("task_type", "scalar")
        self._interface: str = cfg.model.get("interface", "data_wrapper")
        self.dens_weight: float = 0.0  # Set by DeNSCallback if active

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Accept state dicts written by earlier versions."""
        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = migrate_state_dict(checkpoint["state_dict"])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch, training=True)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        metrics = self._compute_metrics(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    # ------------------------------------------------------------------
    # Optimizers / schedulers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:
        opt = build_optimizer(self.cfg, self.model)
        sched = build_scheduler(self.cfg, opt)
        if sched is None:
            return {"optimizer": opt}
        monitor = "val/energy_mae" if self._task_type in ("energy_forces", "s2ef") else "val/mae"
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": monitor},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_loss(self, data: Any, training: bool = True) -> torch.Tensor:
        use_dens = self.dens_weight > 0 and self.cfg.model.get("use_dens", False)

        if not use_dens:
            # Delegate entirely to the shared compute_loss utility
            return compute_loss(self.model, data, self.cfg, self.device, training=training)

        # DeNS path: single forward_with_dens call to avoid a double forward pass.
        # Only applicable for S2EF tasks where forces are needed.
        cfg = self.cfg
        loss_type = cfg.training.get("loss", "mse")
        loss_fn = F.mse_loss if loss_type == "mse" else F.l1_loss

        data.pos.requires_grad_(True)
        energy_pred, forces_pred, dens_loss = self.model.forward_with_dens(data)

        energy_target = (data.energy.view(-1) if hasattr(data, "energy") else data.y.view(-1)).to(
            energy_pred.dtype
        )
        forces_target = data.force.to(forces_pred.dtype)

        if cfg.dataset.get("train_on_free_atoms", False):
            mask = _get_free_atom_mask(data)
            if mask is not None:
                forces_pred = forces_pred[mask]
                forces_target = forces_target[mask]

        e_loss_type = cfg.dataset.get("energy_loss", None)
        loss_e = (
            _per_atom_mae_loss(
                energy_pred, energy_target, torch.bincount(data.batch).to(energy_pred.dtype)
            )
            if e_loss_type == "per_atom_mae"
            else loss_fn(energy_pred, energy_target)
        )
        f_loss_type = cfg.dataset.get("force_loss", None)
        loss_f = (
            _l2norm_loss(forces_pred, forces_target)
            if f_loss_type == "l2norm"
            else loss_fn(forces_pred, forces_target)
        )

        e_w = cfg.dataset.get("energy_weight", 1.0)
        f_w = cfg.dataset.get("force_weight", 1.0)
        return e_w * loss_e + f_w * loss_f + self.dens_weight * dens_loss

    def _compute_metrics(self, data: Any) -> dict[str, torch.Tensor]:
        task_type = self._task_type
        cfg = self.cfg

        if task_type == "scalar":
            with torch.no_grad():
                pred = forward_model(self.model, data, self._interface)
                if isinstance(pred, tuple):
                    pred = pred[0]
                target = data.y.view(-1).to(pred.dtype)
                return {"mae": F.l1_loss(pred, target), "mse": F.mse_loss(pred, target)}

        if task_type == "is2re":
            with torch.no_grad():
                pred = forward_model(self.model, data, self._interface)
                if isinstance(pred, tuple):
                    pred = pred[0]
                target = data.y.view(-1).to(pred.dtype)
                ewt_thresh = cfg.dataset.get("ewt_threshold", 0.02)
                ae = (pred - target).abs()
                return {"mae": ae.mean(), "ewt": (ae <= ewt_thresh).float().mean() * 100.0}

        with torch.no_grad():
            energy_pred, forces_pred = forward_model(self.model, data, self._interface)

        energy_target = (data.energy.view(-1) if hasattr(data, "energy") else data.y.view(-1)).to(
            energy_pred.dtype
        )
        forces_target = data.force.to(forces_pred.dtype)

        if cfg.dataset.get("eval_on_free_atoms", False):
            mask = _get_free_atom_mask(data)
            if mask is not None:
                forces_pred = forces_pred[mask]
                forces_target = forces_target[mask]

        return {
            "energy_mae": (energy_pred - energy_target).abs().mean(),
            "force_mae": (forces_pred - forces_target).abs().mean(),
            "force_cos": torch.cosine_similarity(forces_pred, forces_target, dim=-1).mean(),
        }
