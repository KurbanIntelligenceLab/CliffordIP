"""Neighbor list construction, with and without periodic boundaries."""

from typing import Any

import torch

MAX_IMAGE_REPEATS = 8


def _image_offsets(cell: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Integer lattice translations that can bring an image within ``cutoff``.

    The interplanar spacing along lattice direction ``k`` is ``1 / |b_k|`` for the
    reciprocal vectors ``b_k``, so the cutoff reaches ``ceil(cutoff / spacing)`` cells.
    """
    reciprocal = torch.linalg.pinv(cell).transpose(-1, -2)  # (B, 3, 3)
    spacing = 1.0 / reciprocal.norm(dim=-1).clamp(min=1e-12)  # (B, 3)
    reps = torch.ceil(torch.nan_to_num(cutoff / spacing, nan=0.0)).max(dim=0).values
    reps = reps.long().clamp(0, MAX_IMAGE_REPEATS)
    ranges = [torch.arange(-int(r), int(r) + 1, device=cell.device) for r in reps]
    return torch.cartesian_prod(*ranges).view(-1, 3).to(cell.dtype)


def _limit_neighbors(
    src: torch.Tensor, dst: torch.Tensor, dist: torch.Tensor, n_nodes: int, max_neighbors: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the ``max_neighbors`` nearest sources for each destination."""
    counts = torch.bincount(dst, minlength=n_nodes)
    if int(counts.max()) <= max_neighbors:
        return src, dst
    order = torch.argsort(dst * (dist.max() + 1.0) + dist)
    src, dst = src[order], dst[order]
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(dst.shape[0], device=dst.device) - starts[dst]
    keep = rank < max_neighbors
    return src[keep], dst[keep]


def _edges_for_shift(
    pos: torch.Tensor,
    batch: torch.Tensor,
    shift: torch.Tensor | None,
    cutoff: float,
    drop_self: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edges to sources displaced by ``shift``, within one graph and inside ``cutoff``."""
    shifted = pos if shift is None else pos + shift
    delta = shifted.unsqueeze(0) - pos.unsqueeze(1)  # (dst, src, 3)
    dist = delta.norm(dim=-1)

    within = dist < cutoff
    within &= batch.unsqueeze(0) == batch.unsqueeze(1)
    if drop_self:
        within.fill_diagonal_(False)

    dst, src = within.nonzero(as_tuple=True)
    return src, dst, dist[dst, src]


def radius_graph(
    pos: torch.Tensor,
    batch: torch.Tensor | None,
    cutoff: float,
    max_neighbors: int,
) -> torch.Tensor:
    """Radius graph within each graph of the batch, as a (2, E) edge index."""
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    src, dst, dist = _edges_for_shift(pos, batch, None, cutoff, drop_self=True)
    src, dst = _limit_neighbors(src, dst, dist, pos.shape[0], max_neighbors)
    return torch.stack([src, dst])


def radius_graph_pbc(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    max_neighbors: int,
) -> torch.Tensor:
    """Radius graph over periodic images of ``pos``, as a (2, E) edge index."""
    offsets = _image_offsets(cell, cutoff)
    cells = cell[batch]  # (N, 3, 3)

    all_src, all_dst, all_dist = [], [], []
    for offset in offsets:
        is_origin = bool((offset == 0).all())
        shift = None if is_origin else torch.einsum("k,nkl->nl", offset, cells)
        src, dst, dist = _edges_for_shift(pos, batch, shift, cutoff, drop_self=is_origin)
        all_src.append(src)
        all_dst.append(dst)
        all_dist.append(dist)

    src = torch.cat(all_src)
    dst = torch.cat(all_dst)
    dist = torch.cat(all_dist)
    src, dst = _limit_neighbors(src, dst, dist, pos.shape[0], max_neighbors)
    return torch.stack([src, dst])


def build_edges(data: Any, cutoff: float, max_neighbors: int) -> torch.Tensor:
    """Radius graph for ``data``, periodic when a unit cell is present."""
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(data.pos.shape[0], dtype=torch.long, device=data.pos.device)

    cell = getattr(data, "cell", None)
    if cell is None:
        return radius_graph(data.pos, batch, cutoff, max_neighbors)

    cell = cell.view(-1, 3, 3).to(data.pos.dtype)
    n_graphs = int(batch.max()) + 1
    if cell.shape[0] == 1 and n_graphs > 1:
        cell = cell.expand(n_graphs, 3, 3)
    return radius_graph_pbc(data.pos, batch, cell, cutoff, max_neighbors)
