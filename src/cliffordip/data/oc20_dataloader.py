"""
OC20/OC22 LMDB dataset loader for S2EF and IS2RE tasks.

Reads LMDB files from Open Catalyst 2020/2022 datasets and converts them
into PyG Data objects compatible with the existing training infrastructure.

OC20: https://fair-chem.github.io/catalysts/datasets/oc20.html
OC22: https://fair-chem.github.io/catalysts/datasets/oc22.html (oxide electrocatalysts)
"""

import bisect
import os
import os.path as osp
import pickle
from typing import Literal

import lmdb
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

Task = Literal["s2ef", "is2re"]
Split = Literal[
    "train",
    "val_id",
    "val_ood_ads",
    "val_ood_cat",
    "val_ood_both",
    "test_id",
    "test_ood",
    "test_ood_ads",
    "test_ood_cat",
    "test_ood_both",
]

# Fairchem download_data.py layout: s2ef/200k/train, s2ef/all/val_id
FAIRCHEM_S2EF_TRAIN_SIZES = ("200k", "2M", "20M", "all")

# OC22 tarball layout: is2re-total/train, s2ef-total/train (after flattening)
OC22_TASK_NAMES = {"s2ef": "s2ef-total", "is2re": "is2re-total"}


def _resolve_lmdb_dir(root: str, task: str, split: str, oc22: bool = False) -> str:
    """Resolve LMDB dir: standard layout, OC20 split-size layout, or OC22 layout."""
    candidates = [osp.join(root, task, split)]

    # OC20 IS2RE tarball has is2re/all/train, is2re/all/val_id
    if task == "is2re" and not oc22:
        candidates.insert(0, osp.join(root, task, "all", split))

    # OC22 uses s2ef-total, is2re-total as directory names
    if oc22 and task in OC22_TASK_NAMES:
        candidates.insert(0, osp.join(root, OC22_TASK_NAMES[task], split))

    # OC20 split-size layout (e.g. s2ef/2M/train)
    if task == "s2ef":
        if split == "train":
            for size in FAIRCHEM_S2EF_TRAIN_SIZES:
                candidates.append(osp.join(root, task, size, split))
        else:
            candidates.append(osp.join(root, task, "all", split))

    for d in candidates:
        if osp.isdir(d) and any(f.endswith(".lmdb") for f in os.listdir(d)):
            return d
    docs = (
        "OC22: https://fair-chem.github.io/catalysts/datasets/oc22.html"
        if oc22
        else "OC20: https://fair-chem.github.io/catalysts/datasets/oc20.html"
    )
    raise FileNotFoundError(
        f"LMDB directory not found for {task}/{split}. Tried: {candidates}\nDownload from {docs}"
    )


def _tensor(value) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.tensor(value)


def _scalar_id(value) -> int | str:
    """Normalize a system or frame id to a scalar, defaulting to -1."""
    if isinstance(value, (list | tuple)):
        value = value[0] if value else None
    if isinstance(value, torch.Tensor):
        value = value.flatten()[0].item() if value.numel() else None
    return value if isinstance(value, (int | str)) else -1


class OC20LMDBDataset(Dataset):
    """
    Map-style dataset that reads OC20/OC22 LMDB files on-the-fly.

    Unlike InMemoryDataset, this avoids loading the entire dataset into RAM
    (critical for OC2M with 2M+ structures). Each ``__getitem__`` reads a
    single entry from the LMDB.

    Parameters
    ----------
    root : str
        Root directory containing the OC20/OC22 data.
    task : {"s2ef", "is2re"}
        Which task to load (S2EF or IS2RE).
    split : str
        Data split name (e.g., "train", "val_id", "val_ood_ads").
    max_samples : int, optional
        Cap the number of samples (for smoke-testing / debugging).
    oc22 : bool
        If True, resolve paths for OC22 layout (is2re-total/, s2ef-total/).
    """

    def __init__(
        self,
        root: str,
        task: str = "s2ef",
        split: str = "train",
        max_samples: int | None = None,
        oc22: bool = False,
    ):
        super().__init__()
        self.root = root
        self.task = task
        self.split = split
        self.max_samples = max_samples

        self.lmdb_dir = _resolve_lmdb_dir(root, task, split, oc22=oc22)

        # Discover all .lmdb files in the split directory.
        # OC20 stores data across multiple LMDB shards (data.0000.lmdb, data.0001.lmdb, etc.)
        self.lmdb_paths = sorted(
            [osp.join(self.lmdb_dir, f) for f in os.listdir(self.lmdb_dir) if f.endswith(".lmdb")]
        )
        if not self.lmdb_paths:
            raise FileNotFoundError(
                f"No .lmdb files found in {self.lmdb_dir}. Ensure the data has been extracted correctly."
            )

        # Count total entries across all shards (without loading data).
        # OC22 uses a "length" metadata key; OC20 uses sequential keys only.
        self._shard_lengths: list[int] = []
        self._cumulative_lengths: list[int] = []
        cumulative = 0
        for path in self.lmdb_paths:
            env = lmdb.open(
                path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
            )
            with env.begin() as txn:
                raw_len = txn.get(b"length")
                if raw_len is not None:
                    n = pickle.loads(raw_len)
                else:
                    n = txn.stat()["entries"]
            env.close()
            self._shard_lengths.append(n)
            cumulative += n
            self._cumulative_lengths.append(cumulative)

        self._total_len = cumulative
        if max_samples is not None and max_samples > 0:
            self._total_len = min(self._total_len, max_samples)

        # Lazy-open LMDB environments (opened on first access per shard).
        self._envs: list[lmdb.Environment | None] = [None] * len(self.lmdb_paths)

    def _get_env(self, shard_idx: int) -> lmdb.Environment:
        if self._envs[shard_idx] is None:
            self._envs[shard_idx] = lmdb.open(
                self.lmdb_paths[shard_idx],
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )
        env = self._envs[shard_idx]
        assert env is not None
        return env

    def _global_to_shard(self, global_idx: int):
        """Map a global index to (shard_idx, local_idx)."""
        shard_idx = bisect.bisect_right(self._cumulative_lengths, global_idx)
        if shard_idx >= len(self._cumulative_lengths):
            raise IndexError(f"Global index {global_idx} out of range.")
        local_idx = global_idx - (self._cumulative_lengths[shard_idx - 1] if shard_idx > 0 else 0)
        return shard_idx, local_idx

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> Data:
        if idx < 0 or idx >= self._total_len:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._total_len}.")

        shard_idx, local_idx = self._global_to_shard(idx)
        env = self._get_env(shard_idx)

        with env.begin() as txn:
            raw = txn.get(str(local_idx).encode("ascii"))
            if raw is None:
                raise KeyError(f"Key {local_idx} not found in shard {self.lmdb_paths[shard_idx]}.")

        # OC20 LMDB entries are pickled PyG Data objects (or dicts).
        obj = pickle.loads(raw)

        # Normalize to a consistent PyG Data format.
        data = self._to_pyg_data(obj)
        return data

    @staticmethod
    def _as_mapping(obj) -> dict:
        """Flatten an LMDB entry into a plain mapping of field name to value."""
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, Data):
            d = dict(getattr(obj, "__dict__", {}))
            store = d.pop("_store", None)
            if store is not None and hasattr(store, "keys"):
                for k in store.keys():
                    d.setdefault(k, store[k])
            return d
        raise TypeError(f"Unexpected LMDB entry type: {type(obj)}. Expected PyG Data or dict.")

    @staticmethod
    def _first(d: dict, *names):
        """First present, non-None value among ``names``."""
        for name in names:
            if d.get(name) is not None:
                return d[name]
        return None

    def _to_pyg_data(self, obj) -> Data:
        """Convert an LMDB entry to a Data with standardized field names.

        Fields: ``z`` (int64), ``pos``, ``y``, ``force``, ``cell``, ``natoms``,
        ``tags``, ``fixed``, ``sid``, ``fid``.
        """
        d = self._as_mapping(obj)
        data = Data()

        pos = self._first(d, "pos")
        if pos is None:
            raise ValueError("LMDB entry has no positions")
        data.pos = _tensor(pos).float()
        data.natoms = data.pos.size(0)

        z = self._first(d, "atomic_numbers", "z")
        data.z = _tensor(z).long() if z is not None else torch.zeros(data.natoms, dtype=torch.long)

        y = self._first(d, "y_relaxed", "y", "energy")
        data.y = _tensor(y).float().view(-1) if y is not None else torch.zeros(1)

        force = self._first(d, "force", "forces")
        if force is not None:
            data.force = _tensor(force).float()

        cell = self._first(d, "cell")
        if cell is not None:
            c = _tensor(cell).float()
            data.cell = c.squeeze(0) if c.dim() == 3 else c.view(3, 3)

        tags = self._first(d, "tags")
        if tags is not None:
            data.tags = _tensor(tags).long()

        fixed = self._first(d, "fixed")
        if fixed is not None:
            data.fixed = _tensor(fixed).bool()

        data.sid = _scalar_id(d.get("sid"))
        data.fid = _scalar_id(d.get("fid"))
        return data

    def _close_envs(self):
        """Close and null all LMDB environments so they reopen lazily.

        Call this before forking (e.g. DataLoader worker init) to ensure
        each worker gets its own LMDB file descriptors.
        """
        if not hasattr(self, "_envs"):
            return
        for i, env in enumerate(self._envs):
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
                self._envs[i] = None

    def __getstate__(self):
        """Pickle support: close envs before serialization (DataLoader fork)."""
        self._close_envs()
        state = self.__dict__.copy()
        state["_envs"] = [None] * len(self.lmdb_paths)
        return state

    def __setstate__(self, state):
        """Unpickle: restore state with null envs (reopened lazily)."""
        self.__dict__.update(state)

    def close(self):
        """Close all open LMDB environments."""
        self._close_envs()

    def __del__(self):
        self.close()
