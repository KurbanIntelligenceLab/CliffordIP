"""
Dataset registry with decorator-based registration.

Adding a new dataset:
    1. Create datasets/_register_mydataset.py
    2. Write a factory function decorated with @register_dataset("mydataset")
    3. The factory receives the full OmegaConf config and returns a list of
       loader dicts: [{"train": DataLoader, "val": DataLoader, ...}, ...]
    4. Import the registration file in datasets/__init__.py

That's it -- zero changes to dataset_registry.py, config_utils.py, or the trainer.
"""

from typing import Callable, Dict, List, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

_DATASET_REGISTRY: Dict[str, Callable[[DictConfig], list]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset factory function.

    The factory function signature must be:
        def factory(cfg: DictConfig) -> list[dict]

    Each dict in the returned list represents one fold/split with keys:
        - "train": DataLoader (required)
        - "val": DataLoader (required)
        - "test": DataLoader (optional)
        - "extra_parts": list[str] (optional, for output directory naming)
        - "runtime_stats": dict (optional, e.g. z-score mean/std)
    """

    def decorator(factory_fn: Callable[[DictConfig], list]):
        if name in _DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{name}' already registered. "
                f"Existing: {_DATASET_REGISTRY[name].__module__}.{_DATASET_REGISTRY[name].__name__}, "
                f"New: {factory_fn.__module__}.{factory_fn.__name__}"
            )
        _DATASET_REGISTRY[name] = factory_fn
        return factory_fn

    return decorator


def _lmdb_worker_init_fn(worker_id):
    """Force each DataLoader worker to reopen LMDB environments.

    LMDB file descriptors are not safe across fork boundaries.
    This ensures each worker gets its own file descriptors.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        ds = worker_info.dataset
        if hasattr(ds, "_close_envs"):
            ds._close_envs()


def build_loader(
    dataset: Dataset,
    cfg: DictConfig,
    shuffle: bool = False,
    generator: Optional[torch.Generator] = None,
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
    worker_init = _lmdb_worker_init_fn if hasattr(dataset, "_close_envs") else None

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
    """Instantiate dataset loaders from the registry using the merged config."""
    dataset_name = cfg.dataset.name
    if dataset_name not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise KeyError(
            f"Dataset '{dataset_name}' not found in registry. "
            f"Available datasets: [{available}]. "
            f"Did you forget to import datasets._register_{dataset_name}?"
        )
    return _DATASET_REGISTRY[dataset_name](cfg)


def list_datasets() -> List[str]:
    """Return sorted list of registered dataset names."""
    return sorted(_DATASET_REGISTRY.keys())
