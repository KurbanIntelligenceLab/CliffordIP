"""Entry point for ``cliffordip-train``.

Usage::
    cliffordip-train fit --config src/cliffordip/configs/dataset/oc20_s2ef.yaml
    cliffordip-train fit --config my_run.yaml --model.n_channels 64
"""

from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from cliffordip.lightning.datamodule import CliffordIPDataModule
from cliffordip.lightning.module import CliffordIPLightningModule


def main() -> None:
    import cliffordip.data  # noqa: F401 — registers datasets
    import cliffordip.train.register_cliffordip  # noqa: F401 — registers models

    LightningCLI(CliffordIPLightningModule, CliffordIPDataModule)


if __name__ == "__main__":
    main()
