"""PyTorch Lightning callbacks for CliffordIP training."""

from __future__ import annotations

import lightning as L

from cliffordip.wrapper import ExponentialMovingAverage


class EMACallback(L.Callback):
    """Exponential Moving Average of model parameters as a Lightning Callback.

    Applies EMA shadow weights during validation, restores training weights after.
    ~2-5% accuracy improvement for free.

    Usage::
        trainer = L.Trainer(callbacks=[EMACallback(decay=0.999)])
    """

    def __init__(self, decay: float = 0.999) -> None:
        self._decay = decay
        self._ema: ExponentialMovingAverage | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._ema = ExponentialMovingAverage(pl_module.model, self._decay)

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

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._ema is not None:
            self._ema.apply_shadow()

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._ema is not None:
            self._ema.restore()


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

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        if self._warmup_epochs > 0:
            weight = min(1.0, epoch / self._warmup_epochs) * self._max_weight
        else:
            weight = self._max_weight
        pl_module.dens_weight = weight
