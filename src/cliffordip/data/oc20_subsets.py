"""OC20 S2EF subsets filtered by adsorbate, using the OC20 metadata mapping."""

import os
import os.path as osp
import pickle

from torch.utils.data import Subset

from .oc20_dataloader import OC20LMDBDataset

MAPPING_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl"

# C1 pathway intermediates for CO2 reduction.
CO2RR_ADSORBATES: set[str] = {
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
    "*C",
}

# N2 reduction intermediates.
NRR_ADSORBATES: set[str] = {
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

# C-C coupling intermediates.
C2_ADSORBATES: set[str] = {
    "*C*C",
    "*CCH",
    "*CCH2",
    "*CCH3",
    "*CH*CH",
    "*CHCH2",
    "*CH2CH3",
    "*CCO",
    "*CHCO",
    "CH2*CO",
}

SUBSETS: dict[str, set[str]] = {
    "co2rr": CO2RR_ADSORBATES,
    "nrr": NRR_ADSORBATES,
    "c2": C2_ADSORBATES,
}


def _load_mapping(root: str) -> dict:
    path = osp.join(root, "oc20_data_mapping.pkl")
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"OC20 mapping not found: {path}\nDownload: wget -O {path} {MAPPING_URL}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _matching_sids(mapping: dict, adsorbates: set[str]) -> set[str]:
    """System ids whose adsorbate is in ``adsorbates``, keyed as the LMDB stores them."""
    result: set[str] = set()
    for key, meta in mapping.items():
        if isinstance(meta, dict) and meta.get("ads_symbols") in adsorbates:
            sid = str(key)
            result.add(sid[6:] if sid.startswith("random") else sid)
    return result


def _build_indices(
    base: OC20LMDBDataset,
    valid_sids: set[str],
    cache_path: str,
    name: str,
    max_collect: int | None = None,
    max_scan: int | None = None,
) -> list[int]:
    """Indices of samples whose sid is in ``valid_sids``, cached to ``cache_path``."""
    if osp.isfile(cache_path):
        indices = [int(x) for x in open(cache_path).read().split()]
        return indices[:max_collect] if max_collect else indices

    indices = []
    n = len(base)
    scan_limit = n if max_scan is None else min(n, max_scan)
    for i in range(scan_limit):
        if max_collect is not None and len(indices) >= max_collect:
            break
        if (i + 1) % 50000 == 0:
            print(f"  {name} index build: {i + 1}/{scan_limit} ({len(indices)} matched)")
        sid = getattr(base[i], "sid", None)
        if sid is not None and str(sid) in valid_sids:
            indices.append(i)

    parent = osp.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(cache_path, "w") as f:
        f.write("\n".join(map(str, indices)))
    return indices[:max_collect] if max_collect else indices


def build_subset_dataset(
    name: str,
    root: str,
    split: str,
    max_samples: int | None = None,
) -> Subset:
    """OC20 S2EF restricted to the adsorbates of the named subset."""
    if name not in SUBSETS:
        raise KeyError(f"Unknown OC20 subset '{name}'. Available: {', '.join(sorted(SUBSETS))}.")

    base = OC20LMDBDataset(root=root, task="s2ef", split=split, max_samples=None)
    valid_sids = _matching_sids(_load_mapping(root), SUBSETS[name])

    cache_dir = osp.join(root, name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = osp.join(cache_dir, f"{split}_indices.txt")

    max_collect = max_samples if max_samples and max_samples < 10000 else None
    max_scan = 50000 if (max_collect and max_collect < 1000) else None
    indices = _build_indices(base, valid_sids, cache_path, name, max_collect, max_scan)
    if max_samples and max_collect is None:
        indices = indices[:max_samples]

    if not indices:
        raise ValueError(
            f"No {name} samples found in split '{split}' "
            f"(scanned {max_scan or len(base)} of {len(base)}). "
            f"The {name} subset keeps only: {', '.join(sorted(SUBSETS[name]))}."
        )
    return Subset(base, indices)
