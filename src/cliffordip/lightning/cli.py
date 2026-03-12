"""CLI entry point for cliffordip-train using LightningCLI.

Usage::
    cliffordip-train fit --config src/cliffordip/configs/dataset/oc20_s2ef.yaml
    cliffordip-train fit --config my_run.yaml --trainer.max_epochs 20
    cliffordip-train fit --config my_run.yaml --model.n_channels 64
"""

from __future__ import annotations

import warnings

# Suppress known noisy dependency warnings
warnings.filterwarnings(
    "ignore",
    message="The TorchScript type system doesn't support instance-level annotations",
    category=UserWarning,
    module="torch.jit",
)
warnings.filterwarnings(
    "ignore",
    message=".*PyTorch version.*performance degredations.*",
    category=UserWarning,
    module="nequip",
)

import torch

# Allow e3nn Wigner constants (PyTorch 2.6+ weights_only=True compatibility)
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

from lightning.pytorch.cli import LightningCLI

from cliffordip.lightning.datamodule import CliffordIPDataModule
from cliffordip.lightning.module import CliffordIPLightningModule


def main() -> None:
    """Entry point for the ``cliffordip-train`` CLI command."""
    import cliffordip.data    # noqa: F401 — register datasets
    import cliffordip.train.register_cliffordip  # noqa: F401 — register clifford variants

    LightningCLI(CliffordIPLightningModule, CliffordIPDataModule)


if __name__ == "__main__":
    main()
