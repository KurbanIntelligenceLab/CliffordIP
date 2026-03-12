"""MD17 dataset registration — molecular dynamics energy + force prediction.

MD17 contains DFT trajectories for small organic molecules.
task_type = "energy_forces" — model predicts energy and forces jointly.

Available molecules:
  aspirin, ethanol, malonaldehyde, naphthalene, salicylic acid,
  toluene, uracil, benzene, paracetamol, azobenzene,
  revised benzene (rmd17_benzene), ...

Usage:
    dataset.name: md17
    dataset.data_root: data/MD17
    dataset.molecule: aspirin     # required
    dataset.n_train: 950          # standard small split
    dataset.n_val: 50
"""

from __future__ import annotations

import torch
from torch_geometric.datasets import MD17
from torch_geometric.transforms import Compose, Distance, RadiusGraph

from cliffordip.train.dataset_registry import build_loader, register_dataset


@register_dataset("md17")
def build_md17(cfg) -> list:
    ds = cfg.dataset
    root = ds.data_root
    molecule = ds.get("molecule", "aspirin")

    cutoff = float(cfg.model.get("cutoff", 5.0))
    transform = Compose([RadiusGraph(r=cutoff, loop=False), Distance(norm=False)])
    dataset = MD17(root=root, name=molecule, transform=transform)

    n_total = len(dataset)
    n_train = int(ds.get("n_train", 950))
    n_val = int(ds.get("n_val", 50))
    n_test = n_total - n_train - n_val

    if n_test < 0:
        raise ValueError(
            f"MD17 '{molecule}' has only {n_total} samples; "
            f"n_train={n_train} + n_val={n_val} exceeds total."
        )

    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    result = {
        "train": build_loader(train_ds, cfg, shuffle=True),
        "val": build_loader(val_ds, cfg, shuffle=False),
        "test": build_loader(test_ds, cfg, shuffle=False),
    }
    return [result]
