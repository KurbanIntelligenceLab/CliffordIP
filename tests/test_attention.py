"""Multi-head attention behaviour."""

import pytest
import torch
from conftest import ATOL_MODEL, DTYPE, build_graph, energy_and_forces, make_model

from cliffordip.interaction import CliffordAttention

N_CH = 8


def _inputs(n_edges=12, n_rbf=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    h_i = torch.randn(n_edges, N_CH, dtype=DTYPE, generator=g)
    h_j = torch.randn(n_edges, N_CH, dtype=DTYPE, generator=g)
    rbf = torch.rand(n_edges, n_rbf, dtype=DTYPE, generator=g)
    dst = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])[:n_edges]
    return h_i, h_j, dst, rbf


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_attention_returns_one_weight_per_head(n_heads):
    module = CliffordAttention(N_CH, n_heads=n_heads, n_rbf=8).double()
    with torch.no_grad():
        attn = module(*_inputs())
    assert attn.shape == (12, n_heads)


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_attention_is_normalized_over_each_receiver(n_heads):
    module = CliffordAttention(N_CH, n_heads=n_heads, n_rbf=8).double()
    h_i, h_j, dst, rbf = _inputs()
    with torch.no_grad():
        attn = module(h_i, h_j, dst, rbf)
    for node in dst.unique():
        total = attn[dst == node].sum(dim=0)
        assert torch.allclose(total, torch.ones(n_heads, dtype=DTYPE), atol=1e-12)


def test_heads_learn_different_attention_patterns():
    """Averaging the heads away would make every head identical."""
    torch.manual_seed(0)
    module = CliffordAttention(N_CH, n_heads=4, n_rbf=8).double()
    with torch.no_grad():
        attn = module(*_inputs())
    spread = attn.std(dim=1).max()
    assert spread > 1e-3, f"heads are indistinguishable (spread {spread:.2e})"


def test_single_head_is_a_plain_softmax():
    module = CliffordAttention(N_CH, n_heads=1, n_rbf=8).double()
    h_i, h_j, dst, rbf = _inputs()
    with torch.no_grad():
        attn = module(h_i, h_j, dst, rbf)
    assert attn.shape[1] == 1
    assert torch.allclose(attn[dst == 0].sum(), torch.ones((), dtype=DTYPE), atol=1e-12)


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_model_stays_equivariant_for_every_head_count(n_heads):
    from cliffordip.equivariance import check_equivariance

    model = make_model(n_channels=8, n_heads=n_heads)
    result = check_equivariance(model, build_graph(), n_trials=5, atol=ATOL_MODEL)
    assert result["passed"], result["deviations"]


def test_head_count_must_divide_the_channels():
    with pytest.raises(AssertionError):
        CliffordAttention(N_CH, n_heads=3, n_rbf=8)


def test_attention_changes_the_model_output():
    with_attn = make_model(n_channels=8, use_attention=True, seed=3)
    without = make_model(n_channels=8, use_attention=False, seed=3)
    data = build_graph()
    with torch.no_grad():
        a, _ = energy_and_forces(with_attn, data)
        b, _ = energy_and_forces(without, data)
    assert not torch.allclose(a, b)
