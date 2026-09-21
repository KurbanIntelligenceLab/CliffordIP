"""Dataset registry and DataLoader construction.

Register an in-tree dataset with ``@register_dataset("name")``. Out-of-tree
packages advertise factories under the ``cliffordip.datasets`` entry-point group.
"""

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from cliffordip.train.registry import Registry

DATASETS: Registry[list] = Registry("dataset", entry_point_group="cliffordip.datasets")


def register_dataset(name: str):
    """Decorator registering a ``(cfg) -> list[dict]`` factory under ``name``.

    Each dict describes one split group with keys ``train`` and ``val``, and
    optionally ``test``, ``extra_parts`` and ``runtime_stats``.
    """
    return DATASETS.register(name)


def mark_unavailable(name: str, reason: str) -> None:
    """Record why a dataset could not be imported, for use in error messages."""
    DATASETS.mark_unavailable(name, reason)


def _supports_lmdb_reopen(dataset) -> bool:
    """Whether ``dataset`` (or the dataset a Subset wraps) holds LMDB handles."""
    return hasattr(dataset, "_close_envs") or hasattr(
        getattr(dataset, "dataset", None), "_close_envs"
    )


def _close_envs(dataset) -> None:
    target = dataset if hasattr(dataset, "_close_envs") else getattr(dataset, "dataset", None)
    if target is not None:
        target._close_envs()


def _lmdb_worker_init_fn(worker_id):
    """Reopen LMDB environments in each worker; the handles are not fork-safe."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        _close_envs(info.dataset)


def build_loader(
    dataset: Dataset,
    cfg: DictConfig,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create a DataLoader with shared performance settings from config.

    Centralizes num_workers, pin_memory, persistent_workers so every dataset
    gets consistent DataLoader tuning without repeating kwargs everywhere.
    """
    batch_size = cfg.training.batch_size
    num_workers = cfg.training.get("num_workers", 0)
    pin_memory = cfg.training.get("pin_memory", False)
    persistent = cfg.training.get("persistent_workers", False) and num_workers > 0
    prefetch_factor = cfg.training.get("prefetch_factor", 2) if num_workers > 0 else None

    # Use LMDB-safe worker init if dataset supports it
    worker_init = _lmdb_worker_init_fn if _supports_lmdb_reopen(dataset) else None

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        generator=generator,
    )
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    if worker_init is not None:
        kwargs["worker_init_fn"] = worker_init

    return DataLoader(dataset, **kwargs)


def build_dataloaders(cfg: DictConfig) -> list:
    """Instantiate dataset loaders for ``cfg.dataset.name``."""
    return DATASETS.get(cfg.dataset.name)(cfg)


def list_datasets() -> list[str]:
    return DATASETS.names()


def unavailable_datasets() -> dict[str, str]:
    return DATASETS.unavailable()
