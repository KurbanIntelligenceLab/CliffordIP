"""Equivariant GNN for machine learning interatomic potentials on Clifford algebra Cl(3,0).

Install::
    pip install cliffordip          # core only
    pip install cliffordip[oc20]    # + OC20/OC22 datasets

Quickstart::
    from cliffordip import CliffordIPWrapper

    model = CliffordIPWrapper(n_channels=52, n_interactions=5, cutoff=6.0)
    energy, forces = model(data)

Training::
    cliffordip train --dataset oc20_s2ef
"""

from importlib.metadata import PackageNotFoundError, version

from cliffordip.cliffordip import (
    CliffordAlgebra,
    CliffordIPGateActivation,
    CliffordIPLinear,
    CliffordIPNorm,
)
from cliffordip.interaction import CliffordNet
from cliffordip.wrapper import CliffordIPWrapper

try:
    __version__ = version("cliffordip")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"

__all__ = [
    "CliffordAlgebra",
    "CliffordIPGateActivation",
    "CliffordIPLinear",
    "CliffordIPNorm",
    "CliffordIPWrapper",
    "CliffordNet",
    "__version__",
]
