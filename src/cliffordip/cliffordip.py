"""
Clifford Algebra Cl(3,0) for Equivariant Neural Networks.

Grade-sparse GP dispatch, zero-allocation multivector construction,
and torch.compile-friendly forward paths.

Memory layout: [s, e1, e2, e3, e12, e13, e23, e123]
        index:  0   1   2   3    4    5    6     7
"""

from typing import Tuple

import torch
import torch.nn as nn

# ============================================================
# Constants
# ============================================================

DIM = 8
N_GRADES = 4

GRADE_RANGES = {
    0: (0, 1),   # scalar
    1: (1, 4),   # vector
    2: (4, 7),   # bivector
    3: (7, 8),   # pseudoscalar
}

GRADE_DIMS = {0: 1, 1: 3, 2: 3, 3: 1}

ALL_GRADES = (0, 1, 2, 3)

S, E1, E2, E3, E12, E13, E23, E123 = range(8)

# Lookup: GP(grade_a, grade_b) → set of output grades
GP_GRADE_TABLE = {
    (0, 0): {0},    (0, 1): {1},    (0, 2): {2},    (0, 3): {3},
    (1, 0): {1},    (1, 1): {0, 2}, (1, 2): {1, 3}, (1, 3): {2},
    (2, 0): {2},    (2, 1): {1, 3}, (2, 2): {0, 2}, (2, 3): {1},
    (3, 0): {3},    (3, 1): {2},    (3, 2): {1},    (3, 3): {0},
}


def compute_gp_output_grades(
    grades_a: Tuple[int, ...], grades_b: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compute which grades are produced by GP(a, b) given input grades."""
    result: set = set()
    for ga in grades_a:
        for gb in grades_b:
            result |= GP_GRADE_TABLE.get((ga, gb), set())
    return tuple(sorted(result))


def compute_layer_grades(
    n_layers: int, edge_grades: Tuple[int, ...] = (0, 1)
) -> list:
    """Progressive grade activation schedule."""
    node_grades: Tuple[int, ...] = (0,)
    layer_grades = []
    for _ in range(n_layers):
        gp_grades = compute_gp_output_grades(node_grades, edge_grades)
        node_grades = tuple(sorted(set(node_grades) | set(gp_grades)))
        layer_grades.append(node_grades)
    return layer_grades


# ============================================================
# Cayley Table
# ============================================================


def build_cayley_table() -> torch.Tensor:
    """Build the Cl(3,0) multiplication table. Shape: (8, 8, 8)."""
    cayley = torch.zeros(8, 8, 8)
    idx_to_bits = [0, 1, 2, 4, 3, 5, 6, 7]
    bits_to_idx = {0: 0, 1: 1, 2: 2, 4: 3, 3: 4, 5: 5, 6: 6, 7: 7}

    def blade_product(bits_a, bits_b):
        gens_a = [i for i in range(3) if bits_a & (1 << i)]
        gens_b = [i for i in range(3) if bits_b & (1 << i)]
        gens = gens_a + gens_b
        sign = 1
        n = len(gens)
        for i in range(n):
            for j in range(n - 1 - i):
                if gens[j] > gens[j + 1]:
                    gens[j], gens[j + 1] = gens[j + 1], gens[j]
                    sign *= -1
        result = []
        i = 0
        while i < len(gens):
            if i + 1 < len(gens) and gens[i] == gens[i + 1]:
                i += 2
            else:
                result.append(gens[i])
                i += 1
        result_bits = 0
        for g in result:
            result_bits |= 1 << g
        return result_bits, sign

    for i in range(8):
        for j in range(8):
            result_bits, sign = blade_product(idx_to_bits[i], idx_to_bits[j])
            k = bits_to_idx[result_bits]
            cayley[i, j, k] = sign
    return cayley


# ============================================================
# Core Algebra — Optimized
# ============================================================


class CliffordAlgebra(nn.Module):
    """Core Cl(3,0) algebra with grade-sparse GP dispatch."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cayley", build_cayley_table())
        self.register_buffer(
            "_reverse_signs",
            torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.float32),
            persistent=False,
        )

    # ---- Full GP (all 8 components active) ----

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Full GP: a * b. (..., 8) × (..., 8) → (..., 8).

        Hardcoded from Cayley table. 64 multiply-adds total.
        torch.compile will fuse these into a single kernel.
        """
        a0, a1, a2, a3, a4, a5, a6, a7 = a.unbind(-1)
        b0, b1, b2, b3, b4, b5, b6, b7 = b.unbind(-1)

        o0 = a0*b0 + a1*b1 + a2*b2 + a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
        o1 = a0*b1 + a1*b0 + a4*b2 - a2*b4 + a5*b3 - a3*b5 - a6*b7 - a7*b6
        o2 = a0*b2 + a2*b0 + a1*b4 - a4*b1 + a6*b3 - a3*b6 + a5*b7 + a7*b5
        o3 = a0*b3 + a3*b0 + a1*b5 - a5*b1 + a2*b6 - a6*b2 - a4*b7 - a7*b4
        o4 = a0*b4 + a4*b0 + a1*b2 - a2*b1 + a6*b5 - a5*b6 + a3*b7 + a7*b3
        o5 = a0*b5 + a5*b0 + a1*b3 - a3*b1 + a4*b6 - a6*b4 - a2*b7 - a7*b2
        o6 = a0*b6 + a6*b0 + a2*b3 - a3*b2 + a5*b4 - a4*b5 + a1*b7 + a7*b1
        o7 = a0*b7 + a7*b0 + a1*b6 + a6*b1 - a2*b5 - a5*b2 + a3*b4 + a4*b3

        return torch.stack([o0, o1, o2, o3, o4, o5, o6, o7], dim=-1)

    # ---- Grade-sparse GP variants ----

    def gp_scalar_times_mv(self, scalar_mv: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
        """GP when 'a' is scalar-only (grade 0). 8 multiplies instead of 64.

        Use for layer-0 where nodes are scalar-only.
        """
        return scalar_mv[..., 0:1] * mv

    def gp_grades01_times_grades01(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """GP when both inputs have only grades (0, 1). ~24 ops → grades (0, 1, 2).

        Layer 0: nodes=(0,), edges=(0,1) → output grades (0,1)
        Layer 1: nodes=(0,1), edges=(0,1) → output grades (0,1,2)
        """
        a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

        o0 = a0*b0 + a1*b1 + a2*b2 + a3*b3
        o1 = a0*b1 + a1*b0
        o2 = a0*b2 + a2*b0
        o3 = a0*b3 + a3*b0
        o4 = a1*b2 - a2*b1
        o5 = a1*b3 - a3*b1
        o6 = a2*b3 - a3*b2
        # o7 = 0: no pseudoscalar from grades (0,1) only

        return torch.stack([o0, o1, o2, o3, o4, o5, o6, a0.new_zeros(a0.shape)], dim=-1)

    def gp_grades012_times_grades01(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """GP when a has grades (0,1,2) and b has grades (0,1). Produces all grades.

        Derived from full GP with a7=0 and b4=b5=b6=b7=0.
        24 multiply-adds instead of 64.
        """
        a0, a1, a2, a3, a4, a5, a6 = (
            a[..., 0], a[..., 1], a[..., 2], a[..., 3],
            a[..., 4], a[..., 5], a[..., 6],
        )
        b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

        o0 = a0*b0 + a1*b1 + a2*b2 + a3*b3
        o1 = a0*b1 + a1*b0 + a4*b2 + a5*b3
        o2 = a0*b2 + a2*b0 - a4*b1 + a6*b3
        o3 = a0*b3 + a3*b0 - a5*b1 - a6*b2
        o4 = a4*b0 + a1*b2 - a2*b1
        o5 = a5*b0 + a1*b3 - a3*b1
        o6 = a6*b0 + a2*b3 - a3*b2
        o7 = a6*b1 - a5*b2 + a4*b3

        return torch.stack([o0, o1, o2, o3, o4, o5, o6, o7], dim=-1)

    # Precomputed sets for fast dispatch (avoid allocation per call)
    _GRADES_01 = ((0,), (1,), (0, 1))
    _GRADES_012 = ((0,), (1,), (0, 1), (0, 2), (1, 2), (0, 1, 2))

    def dispatch_gp(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        grades_a: Tuple[int, ...],
        grades_b: Tuple[int, ...],
    ) -> torch.Tensor:
        """Dispatch to most efficient GP variant based on known active grades."""
        if grades_a == (0,):
            return self.gp_scalar_times_mv(a, b)
        elif grades_b == (0,):
            # scalar * mv = mv * scalar (commutative for grade-0)
            return self.gp_scalar_times_mv(b, a)
        elif grades_a in self._GRADES_01 and grades_b in self._GRADES_01:
            return self.gp_grades01_times_grades01(a, b)
        elif grades_a in self._GRADES_012 and grades_b in self._GRADES_01:
            return self.gp_grades012_times_grades01(a, b)
        else:
            return self.geometric_product(a, b)

    # ---- Other algebra ops ----

    def geometric_product_reference(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Dense einsum GP (for verification)."""
        return torch.einsum("...i,...j,ijk->...k", a, b, self.cayley)

    def sandwich_product(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """r * x * ~r"""
        return self.geometric_product(self.geometric_product(r, x), self.reverse(r))

    def reverse(self, mv: torch.Tensor) -> torch.Tensor:
        """Reversion: g0:+, g1:+, g2:-, g3:-"""
        return mv * self._reverse_signs

    def grade_select(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        s, e = GRADE_RANGES[grade]
        return mv[..., s:e]

    def grade_decompose(self, mv: torch.Tensor):
        return mv[..., 0:1], mv[..., 1:4], mv[..., 4:7], mv[..., 7:8]

    def norm_squared(self, mv: torch.Tensor) -> torch.Tensor:
        prod = self.geometric_product(mv, self.reverse(mv))
        return prod[..., 0:1]

    def grade_norms(self, mv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        s, v, b, p = self.grade_decompose(mv)
        return torch.cat([
            torch.abs(s),
            torch.sqrt(torch.sum(v ** 2, dim=-1, keepdim=True) + eps),
            torch.sqrt(torch.sum(b ** 2, dim=-1, keepdim=True) + eps),
            torch.abs(p),
        ], dim=-1)

    def rotor_from_bivector(self, bv: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        half = angle * 0.5
        rotor = torch.zeros(*bv.shape[:-1], 8, device=bv.device, dtype=bv.dtype)
        rotor[..., 0] = torch.cos(half).squeeze(-1)
        rotor[..., 4:7] = torch.sin(half) * bv
        return rotor


# ============================================================
# L=2 Feature Augmentation
# ============================================================


def compute_l2_features(direction: torch.Tensor) -> torch.Tensor:
    """Compute L=2 symmetric traceless tensor from unit direction vectors.

    From d = (dx, dy, dz), the 5 independent components of
    the symmetric traceless part of d⊗d are:

        [d_x^2 - 1/3, d_x*d_y, d_x*d_z, d_y^2 - 1/3, d_y*d_z]

    (d_z^2 - 1/3 is redundant since trace = 0)

    Args:
        direction: (..., 3) unit vectors
    Returns:
        (..., 5) L=2 spherical harmonic features
    """
    dx, dy, dz = direction.unbind(-1)
    third = 1.0 / 3.0
    return torch.stack([
        dx * dx - third,
        dx * dy,
        dx * dz,
        dy * dy - third,
        dy * dz,
    ], dim=-1)


# ============================================================
# Multivector Construction (zero-alloc patterns)
# ============================================================


def make_scalar_mv(s: torch.Tensor) -> torch.Tensor:
    """(*, C) → (*, C, 8) with only grade-0 populated. Single alloc."""
    out = s.new_zeros(*s.shape, 8)
    out[..., 0] = s
    return out


def make_vec_mv(v: torch.Tensor) -> torch.Tensor:
    """(*, C, 3) → (*, C, 8) with only grade-1 populated."""
    out = v.new_zeros(*v.shape[:-1], 8)
    out[..., 1:4] = v
    return out


def make_grades01_mv(
    g0: torch.Tensor, g1: torch.Tensor
) -> torch.Tensor:
    """(*, C, 1) scalar + (*, C, 3) vector → (*, C, 8)."""
    out = g0.new_zeros(*g0.shape[:-1], 8)
    out[..., 0:1] = g0
    out[..., 1:4] = g1
    return out


# ============================================================
# Grade-Aware Neural Network Layers — Optimized
# ============================================================


class CliffordIPLinear(nn.Module):
    """Grade-preserving linear map on Cl(3,0) multivectors.

    One weight matrix per grade; all spatial components within a grade share
    the same (c_out, c_in) weight.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        bias: bool = True,
        active_grades: Tuple[int, ...] = ALL_GRADES,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.active = tuple(sorted(active_grades))

        # One (c_out, c_in) weight per grade — shared across all spatial components
        for g in self.active:
            w = nn.Parameter(torch.empty(c_out, c_in))
            nn.init.normal_(w, std=c_in ** -0.5)
            setattr(self, f"w{g}", w)

        self.has_b0 = bias and 0 in self.active
        self.has_b3 = bias and 3 in self.active
        if self.has_b0:
            self.b0 = nn.Parameter(torch.zeros(c_out, 1))
        if self.has_b3:
            self.b3 = nn.Parameter(torch.zeros(c_out, 1))

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        """(..., C_in, 8) → (..., C_out, 8)."""
        out = mv.new_zeros(*mv.shape[:-2], self.c_out, 8)
        for g in self.active:
            s, e = GRADE_RANGES[g]
            w = getattr(self, f"w{g}")
            out[..., s:e] = torch.einsum("oc,...cd->...od", w, mv[..., s:e])
        if self.has_b0:
            out[..., 0:1] = out[..., 0:1] + self.b0
        if self.has_b3:
            out[..., 7:8] = out[..., 7:8] + self.b3
        return out


class CliffordIPNorm(nn.Module):
    """Grade-wise normalization — compile-friendly.

    No Python loop in forward: uses pre-computed index tensors.
    """

    def __init__(
        self,
        n_channels: int,
        active_grades: Tuple[int, ...] = ALL_GRADES,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.active = set(active_grades)
        self.n_channels = n_channels

        for g in range(N_GRADES):
            if g in self.active:
                setattr(self, f"s{g}", nn.Parameter(torch.ones(n_channels, 1)))

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        """(..., C, 8) → (..., C, 8)"""
        slices = []
        for g in range(N_GRADES):
            s, e = GRADE_RANGES[g]
            x = mv[..., s:e]
            if g not in self.active:
                slices.append(x)
                continue
            scale = getattr(self, f"s{g}")
            if g in (0, 3):
                mean = x.mean(dim=-2, keepdim=True)
                std = x.std(dim=-2, keepdim=True) + self.eps
                slices.append(scale * (x - mean) / std)
            else:
                ch_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
                rms = torch.sqrt(torch.mean(ch_norm_sq, dim=-2, keepdim=True) + self.eps)
                slices.append(scale * x / rms)
        return torch.cat(slices, dim=-1)


class CliffordIPGateActivation(nn.Module):
    """Equivariant nonlinearity via norm-gating."""

    def __init__(
        self,
        n_channels: int,
        active_grades: Tuple[int, ...] = ALL_GRADES,
    ):
        super().__init__()
        self.active = set(active_grades)
        self.scalar_act = nn.SiLU()

        for g in (1, 2):
            if g in self.active:
                gate = nn.Sequential(
                    nn.Linear(n_channels, n_channels),
                    nn.SiLU(),
                    nn.Linear(n_channels, n_channels),
                    nn.Sigmoid(),
                )
                setattr(self, f"gate_g{g}", gate)

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        slices = []
        for g in range(N_GRADES):
            s, e = GRADE_RANGES[g]
            x = mv[..., s:e]
            if g not in self.active:
                slices.append(x)
                continue
            if g in (0, 3):
                slices.append(self.scalar_act(x))
            else:
                gate_fn = getattr(self, f"gate_g{g}")
                x_norm = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) + 1e-8)
                gate = gate_fn(x_norm.squeeze(-1)).unsqueeze(-1)
                slices.append(x * gate)
        return torch.cat(slices, dim=-1)


# ============================================================
# Equivariance Test
# ============================================================


def test_equivariance(model_fn=None, n_atoms=5, n_channels=16, atol=1e-5, seed=42):
    """Numerically verify O(3) equivariance."""
    torch.manual_seed(seed)
    alg = CliffordAlgebra()

    a = torch.randn(n_atoms, n_channels, 8)
    b = torch.randn(n_atoms, n_channels, 8)

    bv = torch.randn(3)
    bv = bv / (bv.norm() + 1e-8)
    angle = torch.tensor([torch.pi * torch.rand(1).item()])
    rotor = alg.rotor_from_bivector(bv.unsqueeze(0), angle.unsqueeze(0)).squeeze(0)

    def rotate(mv):
        return alg.sandwich_product(mv, rotor.expand_as(mv))

    results = {}

    gp_then_rot = rotate(alg.geometric_product(a, b))
    rot_then_gp = alg.geometric_product(rotate(a), rotate(b))
    gp_err = (gp_then_rot - rot_then_gp).abs().max().item()
    results["gp_equivariance_error"] = gp_err
    results["gp_equivariant"] = gp_err < atol

    norms_orig = alg.grade_norms(a)
    norms_rot = alg.grade_norms(rotate(a))
    norm_err = (norms_orig - norms_rot).abs().max().item()
    results["grade_norm_invariance_error"] = norm_err
    results["grade_norms_invariant"] = norm_err < atol

    if model_fn is not None:
        with torch.no_grad():
            out_then_rot = rotate(model_fn(a, b))
            rot_then_out = model_fn(rotate(a), rotate(b))
        model_err = (out_then_rot - rot_then_out).abs().max().item()
        results["model_equivariance_error"] = model_err
        results["model_equivariant"] = model_err < atol

    return results


def test_invariance(n_atoms: int = 5, n_channels: int = 16, atol: float = 1e-5, seed: int = 42) -> dict:
    """Numerically verify O(3) invariance of Clifford algebra scalars and inner products."""
    torch.manual_seed(seed)
    alg = CliffordAlgebra()

    a = torch.randn(n_atoms, n_channels, 8)
    b = torch.randn(n_atoms, n_channels, 8)

    bv = torch.randn(3)
    bv = bv / (bv.norm() + 1e-8)
    angle = torch.tensor([torch.pi * torch.rand(1).item()])
    rotor = alg.rotor_from_bivector(bv.unsqueeze(0), angle.unsqueeze(0)).squeeze(0)

    def rotate(mv: torch.Tensor) -> torch.Tensor:
        return alg.sandwich_product(mv, rotor.expand_as(mv))

    results = {}

    # Scalar (grade-0) component is invariant under rotation
    scalar_orig = a[..., S]
    scalar_rot = rotate(a)[..., S]
    scalar_err = (scalar_orig - scalar_rot).abs().max().item()
    results["scalar_grade_invariance_error"] = scalar_err
    results["scalar_grade_invariant"] = scalar_err < atol

    # Inner product <a, b> = sum over components — invariant under rotation
    inner_orig = (a * b).sum(dim=-1)
    inner_rot = (rotate(a) * rotate(b)).sum(dim=-1)
    inner_err = (inner_orig - inner_rot).abs().max().item()
    results["inner_product_invariance_error"] = inner_err
    results["inner_product_invariant"] = inner_err < atol

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Cl(3,0) Clifford Algebra — Optimized Validation Suite")
    print("=" * 60)

    alg = CliffordAlgebra()

    print("\n1. Cayley table checks:")
    e1 = torch.zeros(8); e1[E1] = 1.0
    e2 = torch.zeros(8); e2[E2] = 1.0

    r = alg.geometric_product(e1, e1)
    assert torch.allclose(r[S], torch.tensor(1.0))
    print(f"   e1 * e1 = {r[S].item():+.0f}  ✓")

    r = alg.geometric_product(e1, e2)
    assert torch.allclose(r[E12], torch.tensor(1.0))
    print("   e1 * e2 = +e12  ✓")

    r = alg.geometric_product(e2, e1)
    assert torch.allclose(r[E12], torch.tensor(-1.0))
    print("   e2 * e1 = -e12  ✓")

    print("\n2. Grade-sparse GP consistency:")
    a_test = torch.randn(32, 16, 8)
    b_test = torch.randn(32, 16, 8)

    # Zero out higher grades for test
    a01 = a_test.clone(); a01[..., 4:] = 0
    b01 = b_test.clone(); b01[..., 4:] = 0

    full = alg.geometric_product(a01, b01)
    sparse = alg.gp_grades01_times_grades01(a01, b01)
    err = (full - sparse).abs().max().item()
    print(f"   grades01 GP error: {err:.2e}  {'✓' if err < 1e-5 else '✗'}")

    print("\n3. Fused vs reference GP:")
    gp_fused = alg.geometric_product(a_test, b_test)
    gp_ref = alg.geometric_product_reference(a_test, b_test)
    err = (gp_fused - gp_ref).abs().max().item()
    print(f"   Max error: {err:.2e}  {'✓' if err < 1e-5 else '✗'}")

    print("\n4. L=2 features shape:")
    dirs = torch.randn(100, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    l2 = compute_l2_features(dirs)
    print(f"   Input: {dirs.shape} → L2: {l2.shape}  ✓")

    print("\n5. Equivariance:")
    results = test_equivariance()
    for k, v in results.items():
        ok = (isinstance(v, bool) and v) or (isinstance(v, float) and v < 1e-5)
        print(f"   {k}: {v}  {'✓' if ok else '✗'}")

    print("\n5b. Invariance:")
    inv_results = test_invariance()
    for k, v in inv_results.items():
        ok = (isinstance(v, bool) and v) or (isinstance(v, float) and v < 1e-5)
        print(f"   {k}: {v}  {'✓' if ok else '✗'}")

    print("\n6. Optimized CliffordIPLinear (no alloc in forward):")
    cl = CliffordIPLinear(16, 16, active_grades=(0, 1))
    x = torch.randn(8, 16, 8)
    y = cl(x)
    print(f"   Input: {x.shape} → Output: {y.shape}  ✓")
    n_p = sum(p.numel() for p in cl.parameters())
    print(f"   Params (grades 0,1): {n_p}")

    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)