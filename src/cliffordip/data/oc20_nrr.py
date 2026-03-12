"""
OC20 NRR subset — filter OC20 S2EF for nitrogen reduction-relevant adsorbates.

Uses OC20 metadata (oc20_data_mapping.pkl) to filter systems with
*N, *NH, *NH3, *N2, *NO, *NO2, *NO3 and related NRR/NOx intermediates.
Relevant for nitrogen reduction reaction (NRR) selectivity and mechanism analysis.
"""

import os
import os.path as osp
import pickle
from typing import List, Set

from torch.utils.data import Subset

from .oc20_dataloader import OC20LMDBDataset

# Nitrogen reduction / NOx intermediates (OC20 ads_symbols format)
NRR_ADSORBATES: Set[str] = {
    "*N",
    "*N2",
    "*NH",
    "*NH3",
    "*NHNH",
    "*N*NH",
    "*N*NO",
    "*NO",
    "*NO2",
    "*NO3",
    "*ONH",
    "*NONH",
    "*ONNH2",
}

MAPPING_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl"


def _load_mapping(root: str) -> dict:
    path = osp.join(root, "oc20_data_mapping.pkl")
    if not osp.isfile(path):
        raise FileNotFoundError(f"OC20 mapping not found: {path}\nDownload: wget -O {path} {MAPPING_URL}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_nrr_sids(mapping: dict) -> Set[str]:
    """Map keys like 'random2181546' to '2181546' to match LMDB sid (int)."""
    result: Set[str] = set()
    for key, meta in mapping.items():
        if not isinstance(meta, dict) or meta.get("ads_symbols") not in NRR_ADSORBATES:
            continue
        try:
            sid_str = str(key)
            if sid_str.startswith("random"):
                result.add(sid_str[6:])
            else:
                result.add(str(key))
        except (ValueError, TypeError):
            pass
    return result


def _build_nrr_indices(
    base: OC20LMDBDataset,
    valid_sids: Set[str],
    cache_path: str,
    max_collect: int = None,
    max_scan: int = None,
) -> List[int]:
    """Build indices of NRR-matching samples. When max_collect is small (e.g. smoke),
    max_scan limits how many samples to scan to avoid timeout on large LMDBs."""
    if osp.isfile(cache_path):
        indices = [int(x) for x in open(cache_path).read().split()]
        if max_collect is not None and max_collect > 0:
            return indices[:max_collect]
        return indices
    indices = []
    n = len(base)
    scan_limit = n if max_scan is None else min(n, max_scan)
    for i in range(scan_limit):
        if max_collect is not None and len(indices) >= max_collect:
            break
        if (i + 1) % 50000 == 0:
            print(f"  NRR index build: {i + 1}/{scan_limit}... ({len(indices)} matched)")
        try:
            d = base[i]
            sid = getattr(d, "sid", None)
            if sid is not None:
                sid_str = str(sid)
                if sid_str in valid_sids or (sid_str.startswith("random") and sid_str[6:] in valid_sids):
                    indices.append(i)
        except Exception:
            pass
    d = osp.dirname(cache_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(cache_path, "w") as f:
        f.write("\n".join(map(str, indices)))
    if max_collect is not None and max_collect > 0:
        return indices[:max_collect]
    return indices


def build_oc20_nrr_dataset(
    root: str,
    split: str,
    max_samples: int = None,
) -> Subset:
    """Build OC20 S2EF subset filtered to NRR adsorbates."""
    base = OC20LMDBDataset(root=root, task="s2ef", split=split, max_samples=None)
    mapping = _load_mapping(root)
    valid_sids = _get_nrr_sids(mapping)
    # Store cache in nrr/ subfolder
    nrr_dir = osp.join(root, "nrr")
    os.makedirs(nrr_dir, exist_ok=True)
    cache_path = osp.join(nrr_dir, f"{split}_indices.txt")
    # For small max_samples (e.g. smoke test), stop early to avoid slow full scan
    max_collect = max_samples if max_samples and max_samples < 10000 else None
    max_scan = 50000 if (max_collect and max_collect < 1000) else None
    indices = _build_nrr_indices(base, valid_sids, cache_path, max_collect, max_scan)
    if max_samples is not None and max_samples > 0 and max_collect is None:
        indices = indices[:max_samples]
    if not indices:
        if max_samples and max_samples < 100:
            indices = list(range(min(max_samples, len(base))))
        else:
            raise ValueError(
                f"No NRR samples found in {split} (scanned up to {max_scan or len(base)}). "
                "NRR subset filters for *N, *NH, *NO, etc. "
                "Try the full S2EF dataset or increase max_scan for smoke tests."
            )
    return Subset(base, indices)
