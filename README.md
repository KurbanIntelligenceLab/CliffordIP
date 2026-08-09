# CliffordIP: Clifford Algebra Equivariant Interatomic Potentials for Heterogeneous Catalysis

[![Paper](https://img.shields.io/badge/npj%20Comput%20Mater-10.1038%2Fs41524--026--02259--8-blue)](https://www.nature.com/articles/s41524-026-02259-8)
[![PyPI](https://img.shields.io/pypi/v/cliffordip)](https://pypi.org/project/cliffordip/)

Equivariant GNN for machine learning interatomic potentials (MLIPs) built on Clifford algebra Cl(3,0). Targets OC20, OC22, QM9, MD17, and any custom dataset via a plugin registry.

> **Published in [npj Computational Materials](https://www.nature.com/articles/s41524-026-02259-8)** — Polat, C., Serpedin, E., Kurban, M. & Kurban, H. *CliffordIP: Clifford algebra equivariant interatomic potentials for heterogeneous catalysis.*

## About

Decarbonizing the global economy requires efficient catalysts for key electrochemical transformations, including CO₂ reduction, nitrogen reduction, and C–C coupling. Systematic catalyst discovery is limited by the cost of the density functional theory calculations needed to evaluate adsorbate–surface energetics across chemically diverse surfaces. MLIPs can accelerate this workflow, but many leading equivariant architectures rely on spherical harmonics and Clebsch–Gordan tensor products, increasing computational complexity and implementation overhead.

CliffordIP is a message-passing interatomic potential built on the Clifford algebra Cl(3,0), representing atomic environments as 8-dimensional multivectors (scalars, vectors, bivectors, and pseudoscalars) coupled through the geometric product. This naturally captures bond directions, reaction planes, and local chirality relevant to catalytic intermediates, while enforcing full O(3) equivariance — including reflections — via the Pin(3) group.

**Results.** On catalysis-focused adsorbate subsets, CliffordIP reduces energy MAE by 20–24% relative to the next-best baseline on CO2RR, N2RR, and C2 formation, and achieves the strongest in-distribution energy performance among the compared baselines on OC20 and OC22 under a unified, compute-constrained protocol, while remaining competitive in force magnitude.

## Installation

```bash
# Core (Clifford algebra + model + Lightning trainer)
pip install cliffordip

# With OC20/OC22 dataset support
pip install "cliffordip[oc20]"

# Development install (editable, all extras)
pip install -e ".[dev]"

# With uv (recommended)
uv pip install -e ".[dev]"
```

## Quickstart

```python
from cliffordip import CliffordIPWrapper, test_equivariance

# Verify O(3) equivariance is intact
test_equivariance()

# Build a model directly
model = CliffordIPWrapper(
    n_channels=52,
    n_interactions=5,
    cutoff=6.0,
    n_rbf=50,
    max_neighbors=50,
)
```

## Training

Training uses [PyTorch Lightning](https://lightning.ai) via the `cliffordip-train` CLI:

```bash
# Fit on OC20 S2EF
cliffordip-train fit \
    --config src/cliffordip/configs/dataset/oc20_s2ef.yaml \
    --trainer.max_epochs 10 \
    --trainer.precision bf16-mixed \
    --trainer.devices 4

# Override any parameter
cliffordip-train fit \
    --config src/cliffordip/configs/dataset/oc20_s2ef.yaml \
    --model.n_channels 64 \
    --data.cfg.training.batch_size 32
```

### Supported trainer features (via Lightning)

| Feature | Config |
|---|---|
| BF16 mixed precision | `--trainer.precision bf16-mixed` |
| Multi-GPU (DDP) | `--trainer.devices N --trainer.strategy ddp` |
| Gradient clipping | `--trainer.gradient_clip_val 1.0` |
| Gradient accumulation | `--trainer.accumulate_grad_batches 4` |
| EMA weights | Add `EMACallback(decay=0.999)` |
| W&B logging | Add `WandbLogger(project="mlip")` |
| Checkpointing | Built-in `ModelCheckpoint` callback |
| Early stopping | Built-in `EarlyStopping` callback |

### Python API

```python
import lightning as L
from cliffordip.lightning import CliffordIPLightningModule, CliffordIPDataModule, EMACallback
from cliffordip.train.config_utils import load_config

cfg = load_config(["dataset.name=oc20_s2ef", "model.name=clifford"])

module = CliffordIPLightningModule(cfg)
datamodule = CliffordIPDataModule(cfg)

trainer = L.Trainer(
    max_epochs=10,
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    callbacks=[EMACallback(decay=0.999)],
)
trainer.fit(module, datamodule)
```

## Supported Datasets

| Dataset | Extra | Name |
|---|---|---|
| OC20 S2EF / IS2RE | `[oc20]` | `oc20_s2ef`, `oc20_is2re` |
| OC22 S2EF / IS2RE | `[oc22]` | `oc22_s2ef`, `oc22_is2re` |
| QM9 | `[qm9]` | `qm9` |
| MD17 | `[md17]` | `md17` (+ molecule config) |

## Adding a New Dataset

1. Create `src/cliffordip/data/mydata.py` with a `@register_dataset("mydata")` decorated factory
2. Add `src/cliffordip/configs/dataset/mydata.yaml` with dataset defaults
3. Add dependencies to `pyproject.toml` optional extras
4. Add `from cliffordip.data import mydata  # noqa` to `src/cliffordip/data/__init__.py`

## Baseline Models

Baseline implementations (PaiNN, NequIP, SchNet, DimeNet++, EquiformerV2, GotenNet, TorchMD-Net) live in [`baselines/`](baselines/) and are **not included in the pip package** — they are available in the GitHub repo for reproducibility only. To use them, clone the repo and add the root to your `PYTHONPATH`.

## Architecture

### Clifford algebra Cl(3,0)

8-dimensional multivectors with grades 0–3 (scalars, vectors, bivectors, pseudoscalar). Key design choices:

- **Grade-sparse dispatch**: Selects cheapest geometric product variant based on grade occupancy (~4× faster for early layers)
- **Progressive grade activation**: Early layers use only grades 0–1; higher grades unlock as the network deepens
- **Zero-allocation forward**: Pre-allocated multivector tensors, filled via index slices for `torch.compile` stability

### Layer stack

1. `src/cliffordip/clifford.py` — Cl(3,0) primitives (`CliffordIPLinear`, `CliffordIPNorm`, `CliffordIPGateActivation`, `CliffordIPGeometricProductLayer`)
2. `src/cliffordip/interaction.py` — Message-passing GNN (RBF edge embedding, equivariant attention, multi-body interactions)
3. `src/cliffordip/wrapper.py` — Training wrapper (`torch.compile`, EMA, DeNS)
4. `src/cliffordip/lightning/` — PyTorch Lightning `LightningModule` + `LightningDataModule`

## Verifying Equivariance

```python
from cliffordip import test_equivariance
test_equivariance()  # raises AssertionError if O(3) equivariance is broken
```

## Citation

If you use CliffordIP in your research, please cite:

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
