"""Group actions on Cl(3,0) multivectors and equivariance checks for a built model."""

import torch

# Multivector layout: [s, e1, e2, e3, e12, e13, e23, e123]
BIVECTOR_PAIRS = ((0, 1), (0, 2), (1, 2))

GRADE_INVOLUTION_SIGNS = (1, -1, -1, -1, 1, 1, 1, -1)


def multivector_rep(q: torch.Tensor) -> torch.Tensor:
    """Induced action of an orthogonal matrix on multivectors, as an (8, 8) matrix."""
    m = torch.zeros(8, 8, dtype=q.dtype, device=q.device)
    m[0, 0] = 1.0
    m[1:4, 1:4] = q
    for a, (i, j) in enumerate(BIVECTOR_PAIRS):
        for b, (k, n) in enumerate(BIVECTOR_PAIRS):
            m[4 + a, 4 + b] = q[i, k] * q[j, n] - q[i, n] * q[j, k]
    m[7, 7] = torch.det(q)
    return m


def transform_multivector(mv: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Apply an orthogonal transform to the trailing (8,) axis of ``mv``."""
    return mv @ multivector_rep(q.to(mv.dtype)).T


def grade_involution(mv: torch.Tensor) -> torch.Tensor:
    """Map each grade ``g`` to ``(-1)^g`` times itself."""
    signs = torch.tensor(GRADE_INVOLUTION_SIGNS, dtype=mv.dtype, device=mv.device)
    return mv * signs


def random_orthogonal(
    generator: torch.Generator | None = None,
    determinant: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Uniform random orthogonal matrix, optionally constrained to ``determinant`` +/-1."""
    a = torch.randn(3, 3, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if determinant is not None and torch.det(q).sign().item() != determinant:
        q[:, 0] = -q[:, 0]
    return q


def check_equivariance(
    model,
    data,
    n_trials: int = 20,
    include_improper: bool = True,
    atol: float = 1e-6,
    seed: int = 0,
) -> dict:
    """Measure energy invariance and force equivariance of ``model`` over random transforms.

    Returns the worst deviation seen for each transform family and whether all
    of them stayed within ``atol``.
    """
    generator = torch.Generator().manual_seed(seed)
    families = {"rotation": 1.0}
    if include_improper:
        families["reflection"] = -1.0

    with torch.no_grad():
        energy_ref, forces_ref = model(data)

    results: dict[str, dict[str, float]] = {}
    for name, determinant in families.items():
        worst_energy = 0.0
        worst_forces = 0.0
        for _ in range(n_trials):
            q = random_orthogonal(generator, determinant, dtype=data.pos.dtype)
            moved = data.clone()
            moved.pos = data.pos @ q.T
            if getattr(moved, "cell", None) is not None:
                moved.cell = moved.cell.view(-1, 3, 3) @ q.T
            with torch.no_grad():
                energy, forces = model(moved)
            worst_energy = max(worst_energy, (energy - energy_ref).abs().max().item())
            worst_forces = max(worst_forces, (forces - forces_ref @ q.T).abs().max().item())
        results[name] = {"energy": worst_energy, "forces": worst_forces}

    results["inversion"] = _single_transform(
        model, data, -torch.eye(3, dtype=data.pos.dtype), energy_ref, forces_ref
    )
    results["translation"] = _translation(model, data, energy_ref, forces_ref)

    worst = max(max(v.values()) for v in results.values())
    return {"deviations": results, "atol": atol, "worst": worst, "passed": worst <= atol}


def _single_transform(model, data, q, energy_ref, forces_ref) -> dict[str, float]:
    moved = data.clone()
    moved.pos = data.pos @ q.T
    with torch.no_grad():
        energy, forces = model(moved)
    return {
        "energy": (energy - energy_ref).abs().max().item(),
        "forces": (forces - forces_ref @ q.T).abs().max().item(),
    }


def _translation(model, data, energy_ref, forces_ref) -> dict[str, float]:
    moved = data.clone()
    moved.pos = data.pos + torch.tensor([3.1, -2.7, 5.3], dtype=data.pos.dtype)
    with torch.no_grad():
        energy, forces = model(moved)
    return {
        "energy": (energy - energy_ref).abs().max().item(),
        "forces": (forces - forces_ref).abs().max().item(),
    }
