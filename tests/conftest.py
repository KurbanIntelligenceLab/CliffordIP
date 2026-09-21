"""Shared fixtures and O(3) helpers."""

import pytest
import torch
from torch_geometric.data import Data

from cliffordip.wrapper import CliffordIPWrapper

# Multivector layout: [s, e1, e2, e3, e12, e13, e23, e123]
BIVECTOR_PAIRS = ((0, 1), (0, 2), (1, 2))

DTYPE = torch.float64
ATOL_MODULE = 1e-10
ATOL_MODEL = 1e-6


def mv_rep(q: torch.Tensor) -> torch.Tensor:
    """Induced action of an orthogonal matrix on Cl(3,0) multivectors, as (8, 8)."""
    q = q.to(DTYPE)
    m = torch.zeros(8, 8, dtype=DTYPE)
    m[0, 0] = 1.0
    m[1:4, 1:4] = q
    for a, (i, j) in enumerate(BIVECTOR_PAIRS):
        for b, (k, m2) in enumerate(BIVECTOR_PAIRS):
            m[4 + a, 4 + b] = q[i, k] * q[j, m2] - q[i, m2] * q[j, k]
    m[7, 7] = torch.det(q)
    return m


def transform_mv(mv: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Apply an orthogonal transform to the last (8,) axis of a multivector."""
    return mv @ mv_rep(q).T.to(mv.dtype)


def grade_involution(mv: torch.Tensor) -> torch.Tensor:
    """Grade involution: grade g -> (-1)^g."""
    signs = torch.tensor([1, -1, -1, -1, 1, 1, 1, -1], dtype=mv.dtype, device=mv.device)
    return mv * signs


def random_so3(generator: torch.Generator) -> torch.Tensor:
    """Uniform random proper rotation."""
    q = _random_orthogonal(generator)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def random_improper(generator: torch.Generator) -> torch.Tensor:
    """Uniform random orthogonal matrix with determinant -1."""
    q = _random_orthogonal(generator)
    if torch.det(q) > 0:
        q[:, 0] = -q[:, 0]
    return q


def random_o3(generator: torch.Generator) -> torch.Tensor:
    """Uniform random orthogonal matrix of either determinant."""
    return _random_orthogonal(generator)


def _random_orthogonal(generator: torch.Generator) -> torch.Tensor:
    a = torch.randn(3, 3, dtype=DTYPE, generator=generator)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def inversion() -> torch.Tensor:
    return -torch.eye(3, dtype=DTYPE)


def random_mv(shape, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, 8, dtype=DTYPE, generator=generator)


def build_graph(n_atoms=6, box=4.0, seed=0, n_atom_types=10):
    """Random molecular graph with positions, species and a single-graph batch index."""
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(n_atoms, 3, dtype=DTYPE, generator=g) - 0.5) * box
    z = torch.randint(1, n_atom_types, (n_atoms,), generator=g)
    return Data(z=z, pos=pos, batch=torch.zeros(n_atoms, dtype=torch.long))


def make_model(n_interactions=3, n_channels=8, cutoff=5.0, seed=0, **kwargs):
    """Small float64 model on CPU, deterministic for a given seed."""
    torch.manual_seed(seed)
    kwargs.setdefault("n_atom_types", 10)
    kwargs.setdefault("n_rbf", 8)
    kwargs.setdefault("n_hidden_output", 8)
    kwargs.setdefault("n_heads", 2)
    kwargs.setdefault("use_compile", False)
    model = CliffordIPWrapper(
        n_channels=n_channels,
        n_interactions=n_interactions,
        cutoff=cutoff,
        **kwargs,
    )
    return model.double().eval()


def energy_and_forces(model, data):
    out = model(data)
    energy, forces = out if isinstance(out, tuple) else (out, None)
    return energy, forces


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(1234)


@pytest.fixture
def tiny_model():
    return make_model()


@pytest.fixture
def graph():
    return build_graph()
