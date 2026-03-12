"""
OC20 CO2RR subset — filter OC20 S2EF for CO2 reduction-relevant adsorbates.

Uses OC20 metadata (oc20_data_mapping.pkl) to filter systems with
*CO, *COOH, *CHO, *CO2, *HCOO, *COH, *CH2O and related intermediates.
Relevant for CO2RR selectivity analysis (e.g. *CO → *CHO vs *CO → *COH).
"""

import os
import os.path as osp
import pickle
from typing import List, Set

from torch.utils.data import Subset

from .oc20_dataloader import OC20LMDBDataset

# CO2 reduction intermediates (OC20 ads_symbols format)
CO2RR_ADSORBATES: Set[str] = {
    "*CO",
    "*COOH",
    "*CHO",
    "*CO2",
    "*CO₂",
    "*HCOO",
    "*COH",
    "*CH2O",
    "*H",
    "*OH",
    "*O",
    "*CH",
    "*CH2",
    "*C",  # common C1 pathway
}

MAPPING_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl"


def _load_mapping(root: str) -> dict:
    path = osp.join(root, "oc20_data_mapping.pkl")
    if not osp.isfile(path):
        raise FileNotFoundError(f"OC20 mapping not found: {path}\nDownload: wget -O {path} {MAPPING_URL}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_co2rr_sids(mapping: dict) -> Set[str]:
    """Map keys like 'random2181546' to '2181546' to match LMDB sid (int)."""
    result: Set[str] = set()
    for key, meta in mapping.items():
        if not isinstance(meta, dict) or meta.get("ads_symbols") not in CO2RR_ADSORBATES:
            continue
        # oc20_data_mapping keys are "random{N}"; OC20 LMDB stores sid=N
        try:
            sid_str = str(key)
            if sid_str.startswith("random"):
                result.add(sid_str[6:])  # "random2181546" -> "2181546"
            else:
                result.add(str(key))
        except (ValueError, TypeError):
            pass
    return result


def _build_co2rr_indices(
    base: OC20LMDBDataset,
    valid_sids: Set[str],
    cache_path: str,
    max_collect: int = None,
    max_scan: int = None,
) -> List[int]:
    """Build indices of CO2RR-matching samples. When max_collect is small (e.g. smoke),
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
            print(f"  CO2RR index build: {i + 1}/{scan_limit}... ({len(indices)} matched)")
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


def build_oc20_co2rr_dataset(
    root: str,
    split: str,
    max_samples: int = None,
) -> Subset:
    """Build OC20 S2EF subset filtered to CO2RR adsorbates."""
    base = OC20LMDBDataset(root=root, task="s2ef", split=split, max_samples=None)
    mapping = _load_mapping(root)
    valid_sids = _get_co2rr_sids(mapping)
    # Store cache in co2rr/ subfolder
    co2rr_dir = osp.join(root, "co2rr")
    os.makedirs(co2rr_dir, exist_ok=True)
    cache_path = osp.join(co2rr_dir, f"{split}_indices.txt")
    # For small max_samples (e.g. smoke test), stop early to avoid slow full scan
    max_collect = max_samples if max_samples and max_samples < 10000 else None
    max_scan = 50000 if (max_collect and max_collect < 1000) else None  # limit scan for smoke
    indices = _build_co2rr_indices(base, valid_sids, cache_path, max_collect, max_scan)
    if max_samples is not None and max_samples > 0 and max_collect is None:
        indices = indices[:max_samples]
    if not indices:
        if max_samples and max_samples < 100:  # smoke test: fall back to first N from base
            indices = list(range(min(max_samples, len(base))))
        else:
            raise ValueError(
                f"No CO2RR samples found in {split} (scanned up to {max_scan or len(base)}). "
                "CO2RR subset filters for *CO, *COOH, etc. The 200k S2EF split may have few matches. "
                "Try the full S2EF dataset or increase max_scan for smoke tests."
            )
    return Subset(base, indices)
