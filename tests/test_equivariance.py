"""O(3) equivariance and physical soundness of the CliffordIP model."""

import pytest
import torch
from conftest import (
    ATOL_MODEL,
    ATOL_MODULE,
    DTYPE,
    build_graph,
    energy_and_forces,
    grade_involution,
    inversion,
    make_model,
    mv_rep,
    random_improper,
    random_mv,
    random_o3,
    random_so3,
    transform_mv,
)
from torch_geometric.data import Data

from cliffordip.cliffordip import (
    ALL_GRADES,
    CliffordAlgebra,
    CliffordIPGateActivation,
    CliffordIPLinear,
    CliffordIPNorm,
)
from cliffordip.interaction import (
    CliffordEdgeEmbedding,
    CliffordMultiBodyInteraction,
    CliffordSelfInteraction,
    CliffordUpdateFunction,
)

SEEDS = [0, 1, 2, 3, 4]
DEPTHS = [1, 2, 3, 5]


# ---------------------------------------------------------------------------
# Layer 1: algebra
# ---------------------------------------------------------------------------


@pytest.fixture
def algebra():
    return CliffordAlgebra().double()


def test_representation_is_a_group_homomorphism(gen):
    for _ in range(10):
        a, b = random_o3(gen), random_o3(gen)
        assert torch.allclose(mv_rep(a @ b), mv_rep(a) @ mv_rep(b), atol=ATOL_MODULE)


def test_inversion_equals_grade_involution(gen):
    mv = random_mv((16, 4), gen)
    assert torch.allclose(transform_mv(mv, inversion()), grade_involution(mv), atol=ATOL_MODULE)


def test_geometric_product_is_o3_equivariant(algebra, gen):
    """The outermorphism extension of an orthogonal map is an algebra automorphism."""
    a, b = random_mv((32,), gen), random_mv((32,), gen)
    for _ in range(10):
        q = random_o3(gen)
        lhs = algebra.geometric_product(transform_mv(a, q), transform_mv(b, q))
        rhs = transform_mv(algebra.geometric_product(a, b), q)
        assert torch.allclose(lhs, rhs, atol=ATOL_MODULE)


def test_geometric_product_respects_grade_involution(algebra, gen):
    a, b = random_mv((32,), gen), random_mv((32,), gen)
    lhs = algebra.geometric_product(grade_involution(a), grade_involution(b))
    rhs = grade_involution(algebra.geometric_product(a, b))
    assert torch.allclose(lhs, rhs, atol=ATOL_MODULE)


def test_dispatch_matches_dense_reference(algebra, gen):
    """Grade-sparse dispatch must agree exactly with the dense einsum path."""
    cases = [((0,), (0, 1)), ((0, 1), (0, 1)), ((0, 1, 2), (0, 1)), (ALL_GRADES, ALL_GRADES)]
    for grades_a, grades_b in cases:
        a, b = _masked_mv((64,), grades_a, gen), _masked_mv((64,), grades_b, gen)
        got = algebra.dispatch_gp(a, b, tuple(grades_a), tuple(grades_b))
        want = algebra.geometric_product_reference(a, b)
        assert torch.allclose(got, want, atol=ATOL_MODULE), (grades_a, grades_b)


def test_reversion_and_norm_identities(algebra, gen):
    mv = random_mv((32,), gen)
    assert torch.allclose(algebra.reverse(algebra.reverse(mv)), mv, atol=ATOL_MODULE)
    versor = _unit_vector_mv(gen)
    assert torch.allclose(
        algebra.norm_squared(versor), torch.ones(1, 1, dtype=DTYPE), atol=ATOL_MODULE
    )


def test_versor_sandwich_implements_a_reflection(algebra, gen):
    """u x u^-1 reflects a vector, and two reflections compose into a rotation."""
    u = _unit_vector_mv(gen)
    x = torch.zeros(1, 8, dtype=DTYPE)
    x[0, 1:4] = torch.randn(3, dtype=DTYPE, generator=gen)

    reflected = algebra.sandwich_product(x, u)
    n = u[0, 1:4]
    v = x[0, 1:4]
    expected = v - 2 * torch.dot(v, n) * n
    assert torch.allclose(reflected[0, 1:4], -expected, atol=ATOL_MODULE) or torch.allclose(
        reflected[0, 1:4], expected, atol=ATOL_MODULE
    )

    u2 = _unit_vector_mv(gen)
    twice = algebra.sandwich_product(algebra.sandwich_product(x, u), u2)
    rotor = algebra.geometric_product(u2, u)
    once = algebra.sandwich_product(x, rotor)
    assert torch.allclose(twice, once, atol=ATOL_MODULE)


def _masked_mv(shape, grades, gen):
    from cliffordip.cliffordip import GRADE_RANGES

    mv = random_mv(shape, gen)
    mask = torch.zeros(8, dtype=DTYPE)
    for g in grades:
        s, e = GRADE_RANGES[g]
        mask[s:e] = 1.0
    return mv * mask


def _unit_vector_mv(gen):
    v = torch.randn(3, dtype=DTYPE, generator=gen)
    v = v / v.norm()
    mv = torch.zeros(1, 8, dtype=DTYPE)
    mv[0, 1:4] = v
    return mv


# ---------------------------------------------------------------------------
# Layer 2: per-module
# ---------------------------------------------------------------------------

N_CH = 6


def _randomize(module):
    """Give every parameter a non-zero value so zero-initialized terms are exercised."""
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5 + 0.1)
    return module


def _modules():
    return {
        "linear": (CliffordIPLinear(N_CH, N_CH), "mv"),
        "linear_no_bias": (CliffordIPLinear(N_CH, N_CH, bias=False), "mv"),
        "norm": (CliffordIPNorm(N_CH), "mv"),
        "gate_activation": (CliffordIPGateActivation(N_CH), "mv"),
        "self_interaction": (CliffordSelfInteraction(N_CH), "mv"),
        "multi_body": (CliffordMultiBodyInteraction(N_CH), "mv"),
        "update": (CliffordUpdateFunction(N_CH, ALL_GRADES, ALL_GRADES), "pair"),
    }


@pytest.mark.parametrize("name", sorted(_modules()))
def test_module_commutes_with_grade_involution(name, gen):
    module, kind = _modules()[name]
    module = _randomize(module).double().eval()
    args = _module_inputs(kind, gen)
    with torch.no_grad():
        lhs = module(*[grade_involution(a) for a in args])
        rhs = grade_involution(module(*args))
    assert torch.allclose(lhs, rhs, atol=ATOL_MODULE), f"{name} breaks grade involution"


@pytest.mark.parametrize("name", sorted(_modules()))
@pytest.mark.parametrize("seed", SEEDS)
def test_module_is_o3_equivariant(name, seed, gen):
    module, kind = _modules()[name]
    module = _randomize(module).double().eval()
    q = random_o3(torch.Generator().manual_seed(seed))
    args = _module_inputs(kind, gen)
    with torch.no_grad():
        lhs = module(*[transform_mv(a, q) for a in args])
        rhs = transform_mv(module(*args), q)
    assert torch.allclose(lhs, rhs, atol=ATOL_MODULE), f"{name} breaks O(3) equivariance"


@pytest.mark.parametrize("name", sorted(_modules()))
def test_module_is_so3_equivariant(name, gen):
    """Rotations alone must pass for every module, including before the parity work."""
    module, kind = _modules()[name]
    module = _randomize(module).double().eval()
    q = random_so3(gen)
    args = _module_inputs(kind, gen)
    with torch.no_grad():
        lhs = module(*[transform_mv(a, q) for a in args])
        rhs = transform_mv(module(*args), q)
    assert torch.allclose(lhs, rhs, atol=ATOL_MODULE), f"{name} breaks SO(3) equivariance"


def _module_inputs(kind, gen):
    if kind == "pair":
        return (random_mv((5, N_CH), gen), random_mv((5, N_CH), gen))
    return (random_mv((5, N_CH), gen),)


def test_edge_embedding_is_o3_equivariant(gen):
    """Edge multivectors are built from distances and directions, not from a multivector."""
    module = _randomize(CliffordEdgeEmbedding(n_rbf=8, n_channels=N_CH, cutoff=5.0)).double().eval()
    direction = torch.randn(7, 3, dtype=DTYPE, generator=gen)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    dist = torch.rand(7, dtype=DTYPE, generator=gen) * 4.0 + 0.5

    for _ in range(5):
        q = random_o3(gen)
        with torch.no_grad():
            lhs = module(dist, direction @ q.T)
            rhs = transform_mv(module(dist, direction), q)
        assert torch.allclose(lhs, rhs, atol=ATOL_MODULE)


# ---------------------------------------------------------------------------
# Layer 3: full model
# ---------------------------------------------------------------------------


def _apply(data, q):
    out = data.clone()
    out.pos = data.pos @ q.to(data.pos.dtype).T
    return out


def _assert_model_equivariant(model, data, q, atol=ATOL_MODEL):
    with torch.no_grad():
        e0, f0 = energy_and_forces(model, data)
        e1, f1 = energy_and_forces(model, _apply(data, q))
    assert torch.allclose(e1, e0, atol=atol), "energy is not invariant"
    assert torch.allclose(f1, f0 @ q.to(f0.dtype).T, atol=atol), "forces are not equivariant"


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_model_rotation_equivariance(depth, seed):
    model = make_model(n_interactions=depth, seed=seed)
    data = build_graph(seed=seed)
    _assert_model_equivariant(model, data, random_so3(torch.Generator().manual_seed(seed)))


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_model_reflection_equivariance(depth, seed):
    model = make_model(n_interactions=depth, seed=seed)
    data = build_graph(seed=seed)
    _assert_model_equivariant(model, data, random_improper(torch.Generator().manual_seed(seed)))


@pytest.mark.parametrize("depth", DEPTHS)
def test_model_inversion_equivariance(depth):
    model = make_model(n_interactions=depth)
    data = build_graph()
    _assert_model_equivariant(model, data, inversion())


def test_model_pin3_sweep(tiny_model, graph, gen):
    """Twenty random orthogonal matrices of mixed determinant."""
    for _ in range(20):
        _assert_model_equivariant(tiny_model, graph, random_o3(gen))


def test_model_equivariance_composes(tiny_model, graph, gen):
    a, b = random_improper(gen), random_improper(gen)
    with torch.no_grad():
        seq, _ = energy_and_forces(tiny_model, _apply(_apply(graph, b), a))
        prod, _ = energy_and_forces(tiny_model, _apply(graph, a @ b))
    assert torch.allclose(seq, prod, atol=ATOL_MODEL)


def test_model_translation_invariance(tiny_model, graph, gen):
    shift = torch.randn(3, dtype=DTYPE, generator=gen) * 10.0
    moved = graph.clone()
    moved.pos = graph.pos + shift
    with torch.no_grad():
        e0, f0 = energy_and_forces(tiny_model, graph)
        e1, f1 = energy_and_forces(tiny_model, moved)
    assert torch.allclose(e1, e0, atol=ATOL_MODEL)
    assert torch.allclose(f1, f0, atol=ATOL_MODEL)


def test_model_permutation_equivariance(tiny_model, graph, gen):
    perm = torch.randperm(graph.pos.shape[0], generator=gen)
    shuffled = graph.clone()
    shuffled.pos = graph.pos[perm]
    shuffled.z = graph.z[perm]
    with torch.no_grad():
        e0, f0 = energy_and_forces(tiny_model, graph)
        e1, f1 = energy_and_forces(tiny_model, shuffled)
    assert torch.allclose(e1, e0, atol=ATOL_MODEL)
    assert torch.allclose(f1, f0[perm], atol=ATOL_MODEL)


# ---------------------------------------------------------------------------
# Layer 4: physical soundness
# ---------------------------------------------------------------------------


def _energy_gradient(model, data):
    d = data.clone()
    d.pos = d.pos.detach().requires_grad_(True)
    energy, _ = energy_and_forces(model, d)
    return torch.autograd.grad(energy.sum(), d.pos)[0]


@pytest.mark.parametrize("seed", SEEDS)
def test_energy_gradient_is_o3_equivariant(seed):
    """dE/dr is a vector field and must transform like one under O(3)."""
    model = make_model(seed=seed)
    data = build_graph(seed=seed)
    q = random_improper(torch.Generator().manual_seed(seed))
    g0 = _energy_gradient(model, data)
    g1 = _energy_gradient(model, _apply(data, q))
    assert torch.allclose(g1, g0 @ q.to(g0.dtype).T, atol=ATOL_MODEL)


def test_energy_gradient_is_finite_and_nonzero(tiny_model, graph):
    g = _energy_gradient(tiny_model, graph)
    assert torch.isfinite(g).all()
    assert g.abs().max() > 0


def _cutoff_sweep(model, n_points):
    cutoff = model.cutoff
    energies = []
    for r in torch.linspace(cutoff - 0.05, cutoff + 0.05, n_points, dtype=DTYPE):
        d = Data(
            z=torch.tensor([6, 8]),
            pos=torch.stack(
                [torch.zeros(3, dtype=DTYPE), torch.tensor([r, 0.0, 0.0], dtype=DTYPE)]
            ),
            batch=torch.zeros(2, dtype=torch.long),
        )
        with torch.no_grad():
            e, _ = energy_and_forces(model, d)
        energies.append(e.item())
    return torch.diff(torch.tensor(energies)).abs().max()


def test_energy_is_continuous_across_the_cutoff(tiny_model):
    """Refining the grid must shrink the largest step: the energy has no jump."""
    coarse = _cutoff_sweep(tiny_model, 51)
    fine = _cutoff_sweep(tiny_model, 401)
    assert fine < coarse / 4, (
        f"largest step did not shrink with the grid: {coarse:.3e} -> {fine:.3e}"
    )


def test_energy_matches_across_the_cutoff_boundary(tiny_model):
    """The two-sided limit at the cutoff agrees: no edge contributes as it leaves."""
    cutoff = tiny_model.cutoff
    values = []
    for r in (cutoff - 1e-9, cutoff + 1e-9):
        d = Data(
            z=torch.tensor([6, 8]),
            pos=torch.stack(
                [torch.zeros(3, dtype=DTYPE), torch.tensor([r, 0.0, 0.0], dtype=DTYPE)]
            ),
            batch=torch.zeros(2, dtype=torch.long),
        )
        with torch.no_grad():
            e, _ = energy_and_forces(tiny_model, d)
        values.append(e.item())
    assert abs(values[0] - values[1]) < 1e-9, f"energy jumps by {abs(values[0] - values[1]):.3e}"


def test_energy_is_extensive(tiny_model, graph):
    """Two copies separated beyond the cutoff give twice the energy."""
    offset = torch.tensor([100.0, 0.0, 0.0], dtype=DTYPE)
    n = graph.pos.shape[0]
    doubled = Data(
        z=torch.cat([graph.z, graph.z]),
        pos=torch.cat([graph.pos, graph.pos + offset]),
        batch=torch.zeros(2 * n, dtype=torch.long),
    )
    with torch.no_grad():
        e_one, f_one = energy_and_forces(tiny_model, graph)
        e_two, f_two = energy_and_forces(tiny_model, doubled)
    assert torch.allclose(e_two, 2 * e_one, atol=ATOL_MODEL)
    assert torch.allclose(f_two[:n], f_one, atol=ATOL_MODEL)
    assert torch.allclose(f_two[n:], f_one, atol=ATOL_MODEL)


@pytest.mark.parametrize("n_atoms", [1, 2, 3])
def test_small_and_isolated_systems_are_finite(tiny_model, n_atoms):
    d = Data(
        z=torch.arange(1, n_atoms + 1),
        pos=torch.arange(n_atoms, dtype=DTYPE).unsqueeze(-1).repeat(1, 3) * 50.0,
        batch=torch.zeros(n_atoms, dtype=torch.long),
    )
    with torch.no_grad():
        e, f = energy_and_forces(tiny_model, d)
    assert torch.isfinite(e).all() and torch.isfinite(f).all()


def test_every_parameter_receives_a_finite_gradient():
    model = make_model()
    data = build_graph()
    energy, forces = energy_and_forces(model, data)
    (energy.sum() + forces.sum()).backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"parameters without gradient: {dead[:10]}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_float32_forward_backward_is_finite():
    model = make_model().float()
    data = build_graph()
    data.pos = data.pos.float()
    energy, forces = energy_and_forces(model, data)
    assert torch.isfinite(energy).all() and torch.isfinite(forces).all()
    (energy.sum() + forces.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_batching_matches_individual_graphs(tiny_model):
    graphs = [build_graph(seed=s) for s in (10, 11, 12)]
    with torch.no_grad():
        singles = [energy_and_forces(tiny_model, g) for g in graphs]
    batched = Data(
        z=torch.cat([g.z for g in graphs]),
        pos=torch.cat([g.pos for g in graphs]),
        batch=torch.cat(
            [torch.full((g.pos.shape[0],), i, dtype=torch.long) for i, g in enumerate(graphs)]
        ),
    )
    with torch.no_grad():
        e_batch, f_batch = energy_and_forces(tiny_model, batched)
    assert torch.allclose(e_batch, torch.cat([e for e, _ in singles]), atol=ATOL_MODEL)
    assert torch.allclose(f_batch, torch.cat([f for _, f in singles]), atol=ATOL_MODEL)


def test_forward_is_deterministic(tiny_model, graph):
    with torch.no_grad():
        a = energy_and_forces(tiny_model, graph)
        b = energy_and_forces(tiny_model, graph)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
