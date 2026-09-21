"""PyTorch Lightning callbacks for CliffordIP training."""

from __future__ import annotations

from typing import cast

import lightning as L
import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Running average of a model's floating-point parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad and p.is_floating_point()
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        """Swap the model onto the averaged weights, keeping the live ones."""
        self.backup = {}
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self) -> None:
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


class EMACallback(L.Callback):
    """Exponential Moving Average of model parameters as a Lightning Callback.

    Applies the averaged weights during validation and restores the training
    weights afterwards. The averages are saved and reloaded with the checkpoint.

    Usage::
        trainer = L.Trainer(callbacks=[EMACallback(decay=0.999)])
    """

    def __init__(self, decay: float = 0.999) -> None:
        self._decay = decay
        self._ema: ExponentialMovingAverage | None = None
        self._pending_shadow: dict[str, torch.Tensor] = {}

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._ema = ExponentialMovingAverage(cast(nn.Module, pl_module.model), self._decay)
        pending = getattr(self, "_pending_shadow", None)
        if pending:
            self._ema.load_state_dict(pending)
            self._pending_shadow = {}

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if self._ema is not None:
            self._ema.update()

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._ema is not None:
            self._ema.apply_shadow()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._ema is not None:
            self._ema.restore()

    def state_dict(self) -> dict:
        return {"decay": self._decay, "shadow": self._ema.state_dict() if self._ema else {}}

    def load_state_dict(self, state_dict: dict) -> None:
        self._decay = state_dict.get("decay", self._decay)
        self._pending_shadow = state_dict.get("shadow", {})


class DeNSCallback(L.Callback):
    """Denoising auxiliary loss (DeNS) weight scheduler.

    Gradually increases the DeNS loss weight from 0 to ``max_weight``
    over ``warmup_epochs`` epochs, then holds it constant.

    Usage::
        trainer = L.Trainer(callbacks=[DeNSCallback(max_weight=0.1, warmup_epochs=5)])
    """

    def __init__(self, max_weight: float = 0.1, warmup_epochs: int = 5) -> None:
        self._max_weight = max_weight
        self._warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch
        if self._warmup_epochs > 0:
            weight = min(1.0, epoch / self._warmup_epochs) * self._max_weight
        else:
            weight = self._max_weight
        pl_module.dens_weight = weight  # type: ignore[assignment]
