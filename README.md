# CliffordIP: Clifford Algebra Equivariant Interatomic Potentials for Heterogeneous Catalysis

[![Paper](https://img.shields.io/badge/npj%20Comput%20Mater-10.1038%2Fs41524--026--02259--8-blue)](https://www.nature.com/articles/s41524-026-02259-8)
[![PyPI](https://img.shields.io/pypi/v/cliffordip)](https://pypi.org/project/cliffordip/)

Equivariant GNN for machine learning interatomic potentials (MLIPs) built on Clifford algebra
Cl(3,0). Targets OC20, OC22, QM9, MD17, and any custom dataset via a plugin registry.

> **Published in [npj Computational Materials](https://www.nature.com/articles/s41524-026-02259-8)** —
> Polat, C., Serpedin, E., Kurban, M. & Kurban, H. *CliffordIP: Clifford algebra equivariant
> interatomic potentials for heterogeneous catalysis.*

## About

Decarbonizing the global economy requires efficient catalysts for key electrochemical
transformations, including CO₂ reduction, nitrogen reduction, and C–C coupling. Systematic catalyst
discovery is limited by the cost of the density functional theory calculations needed to evaluate
adsorbate–surface energetics across chemically diverse surfaces. MLIPs can accelerate this workflow,
but many leading equivariant architectures rely on spherical harmonics and Clebsch–Gordan tensor
products, increasing computational complexity and implementation overhead.

CliffordIP is a message-passing interatomic potential built on the Clifford algebra Cl(3,0),
representing atomic environments as 8-dimensional multivectors (scalars, vectors, bivectors, and
pseudoscalars) coupled through the geometric product. This naturally captures bond directions,
reaction planes, and local chirality relevant to catalytic intermediates, while enforcing full O(3)
equivariance — including reflections — via the Pin(3) group.

**Results.** On catalysis-focused adsorbate subsets, CliffordIP reduces energy MAE by 20–24% relative
to the next-best baseline on CO2RR, N2RR, and C2 formation, and achieves the strongest
in-distribution energy performance among the compared baselines on OC20 and OC22 under a unified,
compute-constrained protocol, while remaining competitive in force magnitude.

## 1.0.0

- **Code refactoring** — a typed model config, grade-indexed parameters, one shared registry, a
  single parameterized catalysis dataset module, a `pytest` suite and CI.
- **Retired review data and code** — `baselines/`, `results/` and the pinned `requirements.txt` are
  no longer part of the repository; the package is the model, its data loaders and its training
  integration. The `v0.1.1-baselines` tag holds the comparison code.
- **Model** — norm-gated pseudoscalar activation, periodic neighbor construction, per-head attention
  over channel groups, direct force prediction, and the `cliffordip` command line interface.

Pure-Python dependencies only: `torch`, `torch-geometric`, `lightning`, `omegaconf`. Neighbor lists
are built in plain PyTorch, so neither `torch-scatter` nor `torch-cluster` is required.

See [CHANGELOG.md](CHANGELOG.md) for the full list, including breaking changes.

## Installation

```bash
pip install cliffordip                  # core
pip install "cliffordip[oc20]"          # + OC20/OC22 loaders
pip install "cliffordip[all]"           # + every dataset extra

uv sync                                 # development, from a clone
```

On CUDA machines install a matching torch build first:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[all]"
```

## Command line

Every subcommand takes `--json` to write one machine-readable object to stdout, keeping human text on
stderr. Exit codes are 0 for success, 1 for failure, 2 for bad usage.

```bash
cliffordip info --json                         # version, torch, CUDA, which extras resolve
cliffordip datasets list --json                # available datasets, and why any are unavailable
cliffordip models list --json
cliffordip check-equivariance --o3 --json      # verify the model against the full O(3) group
cliffordip train --dataset oc20_s2ef --trainer.max_epochs 10
```

`cliffordip train` passes every remaining argument to the Lightning CLI, so all trainer flags apply:

| Feature | Flag |
|---|---|
| BF16 mixed precision | `--trainer.precision bf16-mixed` |
| Multi-GPU (DDP) | `--trainer.devices N --trainer.strategy ddp` |
| Gradient clipping | `--trainer.gradient_clip_val 1.0` |
| Gradient accumulation | `--trainer.accumulate_grad_batches 4` |
| One-step smoke run | `--trainer.fast_dev_run true` |

## Python API

```python
from cliffordip import CliffordIPWrapper

model = CliffordIPWrapper(n_channels=52, n_interactions=5, cutoff=6.0, n_rbf=50)
energy, forces = model(data)  # data: a PyG Data or Batch with z, pos, batch, optional cell
```

Training through Lightning:

```python
import lightning as L
from cliffordip.lightning import CliffordIPLightningModule, CliffordIPDataModule, EMACallback
from cliffordip.train.config_utils import load_config

cfg = load_config(["dataset.name=oc20_s2ef", "model.name=cliffordip"])

trainer = L.Trainer(max_epochs=10, precision="bf16-mixed", callbacks=[EMACallback(decay=0.999)])
trainer.fit(CliffordIPLightningModule(cfg), CliffordIPDataModule(cfg))
```

## Supported datasets

| Dataset | Extra | Name |
|---|---|---|
| OC20 S2EF / IS2RE | `[oc20]` | `oc20_s2ef`, `oc20_is2re` |
| OC20 catalysis subsets | `[oc20]` | `oc20_s2ef_co2rr`, `oc20_s2ef_nrr`, `oc20_s2ef_c2` |
| OC22 S2EF / IS2RE | `[oc22]` | `oc22_s2ef`, `oc22_is2re` |
| QM9 | `[qm9]` | `qm9` |
| MD17 | `[md17]` | `md17` |

`cliffordip datasets list` reports which of these are usable in the current environment and names the
extra to install for any that are not.

## Your own dataset

A dataset is a factory that returns loaders. Register it in your own package — no fork required:

```python
from cliffordip.train.dataset_registry import build_loader, register_dataset


@register_dataset("mydata")
def build_mydata(cfg):
    train, val = load_my_splits(cfg.dataset.data_root)
    return [{"train": build_loader(train, cfg, shuffle=True), "val": build_loader(val, cfg)}]
```

```toml
# your pyproject.toml
[project.entry-points."cliffordip.datasets"]
mydata = "my_package.data:build_mydata"
```

`docs/custom_datasets.md` walks through a complete example, including periodic systems.

## Verifying the model

```bash
cliffordip check-equivariance --o3          # exits non-zero if any deviation exceeds the tolerance
uv run pytest tests/test_equivariance.py    # the full suite
```

The suite checks, in float64 against randomly drawn group elements:

- **Algebra** — the geometric product is an O(3) automorphism; versors implement reflections; the
  grade-sparse dispatch agrees with the dense reference product.
- **Modules** — every grade-aware layer commutes with the grade involution and with the rotor action.
- **Model** — energy invariance and force equivariance under rotations, reflections, inversion, a
  twenty-element Pin(3) sweep, composed transforms, translations and atom permutations.
- **Physics** — the energy is continuous across the cutoff radius, extensive for separated copies,
  finite for isolated atoms, batch-independent and deterministic, and every parameter receives a
  gradient.

## Architecture

### Clifford algebra Cl(3,0)

8-dimensional multivectors with grades 0–3 (scalars, vectors, bivectors, pseudoscalar). Key design
choices:

- **Grade-sparse dispatch** — selects the cheapest geometric product variant for the grades actually
  occupied, which is most of the saving in the early layers
- **Progressive grade activation** — early layers carry grades 0–1; higher grades unlock with depth
- **Norm gating** — grades 1–3 are scaled by gates computed from their rotation- and
  reflection-invariant norms
- **Zero-allocation forward** — pre-allocated multivector tensors filled by index slices, for
  `torch.compile` stability

### Layer stack

1. [`src/cliffordip/cliffordip.py`](src/cliffordip/cliffordip.py) — Cl(3,0) primitives
   (`CliffordAlgebra`, `CliffordIPLinear`, `CliffordIPNorm`, `CliffordIPGateActivation`)
2. [`src/cliffordip/interaction.py`](src/cliffordip/interaction.py) — message-passing GNN
   (`CliffordNet`: RBF edge embedding, equivariant attention, multi-body interactions)
3. [`src/cliffordip/neighbors.py`](src/cliffordip/neighbors.py) — radius graphs, periodic and open
4. [`src/cliffordip/wrapper.py`](src/cliffordip/wrapper.py) — `CliffordIPWrapper`, the training entry
   point
5. [`src/cliffordip/lightning/`](src/cliffordip/lightning/) — Lightning module, datamodule, callbacks

## Baseline models

Baseline implementations (PaiNN, NequIP, SchNet, DimeNet++, EquiformerV2, GotenNet, TorchMD-Net) are
kept at the [`v0.1.1-baselines`](../../tree/v0.1.1-baselines) tag for reproducibility.

## Citation

```bibtex
@article{polat2026cliffordip,
  title={CliffordIP: Clifford algebra equivariant interatomic potentials for heterogeneous catalysis},
  author={Polat, Can and Serpedin, Erchin and Kurban, Mustafa and Kurban, Hasan},
  journal={npj Computational Materials},
  year={2026},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41524-026-02259-8},
  url={https://www.nature.com/articles/s41524-026-02259-8}
}
```
