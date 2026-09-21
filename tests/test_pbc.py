"""Periodic neighbor construction."""

import pytest
import torch
from conftest import ATOL_MODEL, DTYPE, build_graph, energy_and_forces, make_model
from torch_geometric.data import Data

from cliffordip.neighbors import build_edges, radius_graph, radius_graph_pbc


def _simple_cubic(n_side=2, a=3.0):
    """Simple cubic lattice with lattice constant ``a``; each atom has 6 neighbours at ``a``."""
    grid = torch.arange(n_side, dtype=DTYPE)
    pos = torch.cartesian_prod(grid, grid, grid) * a
    cell = torch.eye(3, dtype=DTYPE) * (n_side * a)
    return Data(
        z=torch.full((pos.shape[0],), 6),
        pos=pos,
        cell=cell,
        batch=torch.zeros(pos.shape[0], dtype=torch.long),
    )


def test_simple_cubic_has_six_neighbours_per_atom():
    d = _simple_cubic()
    edges = build_edges(d, cutoff=3.5, max_neighbors=50)
    counts = torch.bincount(edges[1], minlength=d.pos.shape[0])
    assert torch.equal(counts, torch.full_like(counts, 6)), counts


def test_periodic_edges_respect_the_minimum_image_distance():
    d = _simple_cubic()
    edges = build_edges(d, cutoff=3.5, max_neighbors=50)
    src, dst = edges
    delta = d.pos[dst] - d.pos[src]
    box = torch.diagonal(d.cell)
    minimum_image = delta - box * torch.round(delta / box)
    assert torch.allclose(
        minimum_image.norm(dim=-1), torch.full((edges.shape[1],), 3.0, dtype=DTYPE)
    )


def test_periodic_graph_finds_more_edges_than_the_open_graph():
    d = _simple_cubic()
    periodic = build_edges(d, cutoff=3.5, max_neighbors=50)
    open_graph = radius_graph(d.pos, d.batch, 3.5, 50)
    assert periodic.shape[1] > open_graph.shape[1]


def test_no_cell_falls_back_to_the_open_graph():
    d = build_graph()
    assert not hasattr(d, "cell") or d.cell is None
    edges = build_edges(d, cutoff=5.0, max_neighbors=50)
    expected = radius_graph(d.pos, d.batch, 5.0, 50)
    assert edges.shape[1] == expected.shape[1]


def test_a_cell_larger_than_the_cutoff_matches_the_open_graph():
    """With vacuum on every side the periodic graph reduces to the open one."""
    d = build_graph()
    d.cell = torch.eye(3, dtype=DTYPE) * 500.0
    periodic = build_edges(d, cutoff=5.0, max_neighbors=50)
    open_graph = radius_graph(d.pos, d.batch, 5.0, 50)
    assert periodic.shape[1] == open_graph.shape[1]


def test_batched_cells_are_handled_per_graph():
    a, b = _simple_cubic(), _simple_cubic()
    n = a.pos.shape[0]
    batched = Data(
        z=torch.cat([a.z, b.z]),
        pos=torch.cat([a.pos, b.pos]),
        cell=torch.stack([a.cell, b.cell]),
        batch=torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)]),
    )
    edges = build_edges(batched, cutoff=3.5, max_neighbors=50)
    counts = torch.bincount(edges[1], minlength=2 * n)
    assert torch.equal(counts, torch.full_like(counts, 6))
    assert torch.equal(batched.batch[edges[0]], batched.batch[edges[1]])


@pytest.mark.parametrize("translation", [0.37, 1.5, 3.0])
def test_periodic_energy_is_invariant_under_a_lattice_translation(translation):
    """Sliding the whole cell through the periodic boundary must not change the energy."""
    model = make_model(cutoff=3.5)
    d = _simple_cubic()
    with torch.no_grad():
        e0, _ = energy_and_forces(model, d)
    shifted = d.clone()
    shifted.pos = d.pos + translation
    with torch.no_grad():
        e1, _ = energy_and_forces(model, shifted)
    assert torch.allclose(e1, e0, atol=ATOL_MODEL)


def test_offsets_shrink_when_the_cell_grows():
    pos = torch.zeros(1, 3, dtype=DTYPE)
    batch = torch.zeros(1, dtype=torch.long)
    small = radius_graph_pbc(pos, batch, torch.eye(3, dtype=DTYPE).unsqueeze(0) * 2.0, 5.0, 50)
    large = radius_graph_pbc(pos, batch, torch.eye(3, dtype=DTYPE).unsqueeze(0) * 50.0, 5.0, 50)
    assert small.shape[1] > large.shape[1]
