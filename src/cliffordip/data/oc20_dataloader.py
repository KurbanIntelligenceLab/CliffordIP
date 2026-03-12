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
from typing import List, Literal, Optional

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
    raise FileNotFoundError(f"LMDB directory not found for {task}/{split}. Tried: {candidates}\nDownload from {docs}")


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
        task: Task = "s2ef",
        split: Split = "train",
        max_samples: Optional[int] = None,
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
        self.lmdb_paths = sorted([osp.join(self.lmdb_dir, f) for f in os.listdir(self.lmdb_dir) if f.endswith(".lmdb")])
        if not self.lmdb_paths:
            raise FileNotFoundError(
                f"No .lmdb files found in {self.lmdb_dir}. Ensure the data has been extracted correctly."
            )

        # Count total entries across all shards (without loading data).
        # OC22 uses a "length" metadata key; OC20 uses sequential keys only.
        self._shard_lengths: List[int] = []
        self._cumulative_lengths: List[int] = []
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
        self._envs: List[Optional[lmdb.Environment]] = [None] * len(self.lmdb_paths)

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
        return self._envs[shard_idx]

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

    def _is_atomic_data(self, obj) -> bool:
        """Check if obj is AtomicData (from scripts.data.data_utils.atomic_data)."""
        return (
            obj.__class__.__module__ == "scripts.data.data_utils.atomic_data"
            and obj.__class__.__name__ == "AtomicData"
        )

    def _to_pyg_data(self, obj) -> Data:
        """
        Convert a raw LMDB entry (PyG Data or dict) to a standardized Data object.

        Standardized fields:
            data.z          : [natoms] int64 atomic numbers
            data.pos        : [natoms, 3] float32 positions
            data.y          : [1] float32 energy
            data.force      : [natoms, 3] float32 forces  (S2EF only)
            data.cell       : [3, 3] float32 unit cell
            data.natoms     : int number of atoms
            data.tags       : [natoms] int (0=sub, 1=surface, 2=adsorbate)
            data.fixed      : [natoms] bool fixed-atom mask
            data.sid        : system id
            data.fid        : frame id
        """
        if isinstance(obj, Data):
            # LMDB stores PyG Data pickled by OCP.
            # Some use __dict__ (IS2RE), others use _store (S2EF from preprocess_ef).
            d = dict(getattr(obj, "__dict__", {}))
            store = d.get("_store")
            if store is not None and hasattr(store, "keys"):
                for k in store.keys():
                    if k not in d:
                        d[k] = store[k]
            data = Data()
            _t = lambda v: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            if "atomic_numbers" in d:
                data.z = _t(d["atomic_numbers"]).long()
            elif "z" in d:
                data.z = _t(d["z"]).long()
            if "pos" in d:
                data.pos = _t(d["pos"]).float()
            if "y_relaxed" in d and d["y_relaxed"] is not None:
                data.y = _t(d["y_relaxed"]).float().view(-1)
            elif "y" in d and d["y"] is not None:
                data.y = _t(d["y"]).float().view(-1)
            elif "energy" in d and d["energy"] is not None:
                data.y = _t(d["energy"]).float().view(-1)
            if "force" in d and d["force"] is not None:
                data.force = _t(d["force"]).float()
            elif "forces" in d and d["forces"] is not None:
                data.force = _t(d["forces"]).float()
            if "cell" in d:
                c = _t(d["cell"])
                data.cell = c.squeeze(0).float() if c.dim() == 3 else c.float()
            if "tags" in d:
                data.tags = _t(d["tags"]).long()
            if "fixed" in d:
                data.fixed = _t(d["fixed"]).bool()
            if data.pos is None:
                raise ValueError("LMDB entry missing pos")
            data.natoms = data.pos.size(0)
            data.sid = d.get("sid", -1)
            data.fid = d.get("fid", -1)
            if not hasattr(data, "z") or data.z is None:
                data.z = data.pos.new_zeros(data.natoms, dtype=torch.long)
            if not hasattr(data, "y") or data.y is None:
                data.y = data.pos.new_zeros(1)
            return data

        elif self._is_atomic_data(obj):
            # OC20 S2EF preprocessed with fairchem/OCP stores AtomicData (scripts.data.data_utils.atomic_data)
            data = Data()
            _t = lambda v: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            data.z = _t(obj.atomic_numbers).long()
            data.pos = _t(obj.pos).float()
            data.y = (
                _t(obj.energy).float().view(-1)
                if hasattr(obj, "energy") and obj.energy is not None
                else obj.pos.new_zeros(1)
            )
            data.force = (
                _t(obj.forces).float()
                if hasattr(obj, "forces") and obj.forces is not None
                else obj.pos.new_zeros(obj.pos.shape[0], 3)
            )
            c = _t(obj.cell)
            data.cell = c.squeeze(0).float() if c.dim() == 3 else c.float()
            data.tags = _t(obj.tags).long()
            data.fixed = _t(obj.fixed).bool()
            data.natoms = data.pos.size(0)
            sid = getattr(obj, "sid", None)
            if isinstance(sid, (int, str)):
                data.sid = sid
            elif isinstance(sid, (list, tuple)) and sid:
                data.sid = sid[0]
            else:
                data.sid = -1
            return data

        elif isinstance(obj, dict):
            # Some OC20 versions store entries as dicts.
            data = Data()
            data.z = torch.tensor(obj.get("atomic_numbers", obj.get("z")), dtype=torch.long)
            data.pos = torch.tensor(obj["pos"], dtype=torch.float32)

            if "y" in obj:
                data.y = torch.tensor([obj["y"]], dtype=torch.float32)
            elif "y_relaxed" in obj:
                data.y = torch.tensor([obj["y_relaxed"]], dtype=torch.float32)

            if "force" in obj:
                data.force = torch.tensor(obj["force"], dtype=torch.float32)
            elif "forces" in obj:
                data.force = torch.tensor(obj["forces"], dtype=torch.float32)

            if "cell" in obj:
                data.cell = torch.tensor(obj["cell"], dtype=torch.float32).view(3, 3)

            if "tags" in obj:
                data.tags = torch.tensor(obj["tags"], dtype=torch.long)

            if "fixed" in obj:
                data.fixed = torch.tensor(obj["fixed"], dtype=torch.bool)

            data.natoms = data.pos.size(0)
            data.sid = obj.get("sid", -1)
            data.fid = obj.get("fid", -1)

            return data

        else:
            raise TypeError(f"Unexpected LMDB entry type: {type(obj)}. Expected PyG Data or dict.")

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
