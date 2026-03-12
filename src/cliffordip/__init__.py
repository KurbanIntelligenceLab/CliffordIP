"""
cliffordip: Equivariant GNN for machine learning interatomic potentials
built on Clifford algebra Cl(3,0).

Install::
    pip install cliffordip          # core only
    pip install cliffordip[oc20]    # + OC20/OC22 datasets (requires lmdb)
    pip install cliffordip[dev]     # + dev tools

Quickstart::
    from cliffordip import CliffordIPWrapper, test_equivariance

    # Verify O(3) equivariance
    test_equivariance()

    # Training via CLI:
    cliffordip-train fit --config configs/my_run.yaml

Public API
----------
Core algebra primitives:
    CliffordIPLinear, CliffordIPNorm, CliffordIPGateActivation

Model:
    CliffordIP         -- raw GNN backbone
    CliffordIPWrapper     -- training-ready wrapper (compile, EMA, DeNS)
    ExponentialMovingAverage

Lightning training:
    from cliffordip.lightning import CliffordIPLightningModule, CliffordIPDataModule

Config & registry:
    from cliffordip.train.config_utils import load_config
    from cliffordip.train.model_registry import register_model, build_model
    from cliffordip.train.dataset_registry import register_dataset, build_dataloaders
"""

__version__ = "0.1.1"

# Core Clifford algebra — zero heavy dependencies at import time
from cliffordip.cliffordip import (
    CliffordIPGateActivation,
    CliffordIPLinear,
    CliffordIPNorm,
    compute_gp_output_grades,
    compute_layer_grades,
    test_equivariance,
    test_invariance,
)

# GNN backbone and training wrapper
from cliffordip.interaction import CliffordIP
from cliffordip.wrapper import CliffordIPWrapper, ExponentialMovingAverage

__all__ = [
    "__version__",
    # Algebra
    "CliffordIPLinear",
    "CliffordIPNorm",
    "CliffordIPGateActivation",
    "compute_gp_output_grades",
    "compute_layer_grades",
    # Model
    "CliffordIP",
    "CliffordIPWrapper",
    "ExponentialMovingAverage",
    # Utilities
    "test_equivariance",
    "test_invariance",
]
