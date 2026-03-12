"""QM9 dataset registration — small organic molecule property prediction.

QM9 contains 130,831 molecules with 19 quantum chemical properties.
task_type = "scalar" (single property regression).

Target properties (index 0–18):
  0: mu (Debye)          — dipole moment
  1: alpha (Bohr^3)      — isotropic polarizability
  2: homo (Hartree)      — HOMO energy
  3: lumo (Hartree)      — LUMO energy
  4: gap (Hartree)       — HOMO-LUMO gap
  5: r2 (Bohr^2)         — electronic spatial extent
  6: zpve (Hartree)      — zero-point vibrational energy
  7: U0 (Hartree)        — internal energy at 0K
  8: U (Hartree)         — internal energy at 298.15K
  9: H (Hartree)         — enthalpy at 298.15K
 10: G (Hartree)         — free energy at 298.15K
 11: Cv (cal/mol/K)      — heat capacity at 298.15K
 12: U0_atom (eV)        — atomization energy at 0K
 13: U_atom (eV)         — atomization energy at 298.15K
 14: H_atom (eV)         — atomization enthalpy at 298.15K
 15: G_atom (eV)         — atomization free energy at 298.15K
 16: A (GHz)             — rotational constant A
 17: B (GHz)             — rotational constant B
 18: C (GHz)             — rotational constant C

Usage:
    dataset.name: qm9
    dataset.data_root: data/QM9
    dataset.target_idx: 7   # U0 atomization energy (default)
"""

from __future__ import annotations

import torch
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Compose, Distance

from cliffordip.train.dataset_registry import build_loader, register_dataset


class _SelectTarget:
    """Transform that selects a single regression target from data.y."""

    def __init__(self, target_idx: int) -> None:
        self.target_idx = target_idx

    def __call__(self, data):
        data.y = data.y[:, self.target_idx]
        return data


@register_dataset("qm9")
def build_qm9(cfg) -> list:
    ds = cfg.dataset
    root = ds.data_root
    target_idx = int(ds.get("target_idx", 7))

    transform = Compose([Distance(norm=False), _SelectTarget(target_idx)])
    dataset = QM9(root=root, transform=transform)

    n_total = len(dataset)
    n_test = int(ds.get("n_test", 10_000))
    n_val = int(ds.get("n_val", 10_000))
    n_train = n_total - n_val - n_test

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
