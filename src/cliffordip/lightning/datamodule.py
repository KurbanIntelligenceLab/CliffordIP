"""CliffordIP Lightning DataModule."""

from __future__ import annotations

from typing import Any

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from cliffordip.train.dataset_registry import build_dataloaders


class CliffordIPDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping the cliffordip dataset registry.

    Supports any dataset registered via @register_dataset. Datasets are
    lazily built in setup() to avoid forking issues with LMDB.

    Usage::
        from cliffordip.train.config_utils import load_config

        cfg = load_config()
        dm = CliffordIPDataModule(cfg)
        trainer.fit(module, dm)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._loaders: dict[str, Any] = {}
        self._runtime_stats: dict | None = None

    def setup(self, stage: str | None = None) -> None:
        loader_sets = build_dataloaders(self._cfg)
        # Use first fold (multi-fold CV not supported natively in PL DataModule)
        first = loader_sets[0] if loader_sets else {}
        self._loaders = first
        self._runtime_stats = first.get("runtime_stats", None)

    @property
    def runtime_stats(self) -> dict | None:
        """Z-score normalisation stats (mean/std) if available."""
        return self._runtime_stats

    def train_dataloader(self) -> DataLoader:
        return self._loaders["train"]

    def val_dataloader(self) -> DataLoader:
        return self._loaders["val"]

    def test_dataloader(self) -> DataLoader:
        loader = self._loaders.get("test")
        if loader is None:
            raise RuntimeError("No test split available. Check dataset config for 'split_test'.")
        return loader
