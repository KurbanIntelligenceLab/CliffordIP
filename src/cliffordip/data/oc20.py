"""OC20 dataset registration — handles both S2EF and IS2RE tasks.

Registered under six names:
  - dataset.name=oc20 (requires dataset.task_type=s2ef or is2re)
  - dataset.name=oc20_s2ef (auto-loads oc20_s2ef.yaml config)
  - dataset.name=oc20_s2ef_co2rr (CO2 reduction subset, Tier 3)
  - dataset.name=oc20_s2ef_nrr (nitrogen reduction subset, Tier 3)
  - dataset.name=oc20_s2ef_c2 (C-C coupling subset, Tier 3)
  - dataset.name=oc20_is2re (auto-loads oc20_is2re.yaml config)
"""

from cliffordip.train.dataset_registry import build_loader, register_dataset
from cliffordip.data.oc20_c2 import build_oc20_c2_dataset
from cliffordip.data.oc20_co2rr import build_oc20_co2rr_dataset
from cliffordip.data.oc20_dataloader import OC20LMDBDataset
from cliffordip.data.oc20_nrr import build_oc20_nrr_dataset


def _build_oc20(cfg):
    ds = cfg.dataset
    task = ds.get("task_type", "s2ef")

    max_val = ds.get("max_val_samples", None)

    train_dataset = OC20LMDBDataset(
        root=ds.data_root,
        task=task,
        split=ds.get("split_train", "train"),
        max_samples=ds.get("max_train_samples", None),
    )
    val_dataset = OC20LMDBDataset(
        root=ds.data_root,
        task=task,
        split=ds.get("split_val", "val_id"),
        max_samples=max_val,
    )

    # Apply fraction-based subsampling if configured and max_val not already set
    val_frac = ds.get("val_subsample_frac", None)
    if val_frac is not None and max_val is None:
        val_dataset._total_len = max(1, int(val_frac * len(val_dataset)))

    train_loader = build_loader(train_dataset, cfg, shuffle=True)
    val_loader = build_loader(val_dataset, cfg, shuffle=False)

    result = {"train": train_loader, "val": val_loader}

    split_test = ds.get("split_test", None)
    if split_test:
        test_dataset = OC20LMDBDataset(
            root=ds.data_root,
            task=task,
            split=split_test,
            max_samples=ds.get("max_test_samples", None),
        )
        result["test"] = build_loader(test_dataset, cfg, shuffle=False)

    return [result]


@register_dataset("oc20")
def build_oc20(cfg):
    return _build_oc20(cfg)


@register_dataset("oc20_s2ef")
def build_oc20_s2ef(cfg):
    return _build_oc20(cfg)


@register_dataset("oc20_is2re")
def build_oc20_is2re(cfg):
    return _build_oc20(cfg)


@register_dataset("oc20_s2ef_co2rr")
def build_oc20_s2ef_co2rr(cfg):
    """OC20 S2EF filtered to CO2 reduction adsorbates (Tier 3)."""
    ds = cfg.dataset
    max_val = ds.get("max_val_samples", None)
    train_dataset = build_oc20_co2rr_dataset(
        root=ds.data_root,
        split=ds.get("split_train", "train"),
        max_samples=ds.get("max_train_samples", None),
    )
    val_dataset = build_oc20_co2rr_dataset(
        root=ds.data_root,
        split=ds.get("split_val", "val_id"),
        max_samples=max_val,
    )

    # Apply fraction-based subsampling if configured and max_val not already set
    val_frac = ds.get("val_subsample_frac", None)
    if val_frac is not None and max_val is None:
        new_len = max(1, int(val_frac * len(val_dataset)))
        val_dataset.indices = val_dataset.indices[:new_len]

    train_loader = build_loader(train_dataset, cfg, shuffle=True)
    val_loader = build_loader(val_dataset, cfg, shuffle=False)

    result = {"train": train_loader, "val": val_loader}

    split_test = ds.get("split_test", None)
    if split_test:
        test_dataset = build_oc20_co2rr_dataset(
            root=ds.data_root,
            split=split_test,
            max_samples=ds.get("max_test_samples", None),
        )
        result["test"] = build_loader(test_dataset, cfg, shuffle=False)

    return [result]


@register_dataset("oc20_s2ef_nrr")
def build_oc20_s2ef_nrr(cfg):
    """OC20 S2EF filtered to nitrogen reduction adsorbates (Tier 3)."""
    ds = cfg.dataset
    max_val = ds.get("max_val_samples", None)
    train_dataset = build_oc20_nrr_dataset(
        root=ds.data_root,
        split=ds.get("split_train", "train"),
        max_samples=ds.get("max_train_samples", None),
    )
    val_dataset = build_oc20_nrr_dataset(
        root=ds.data_root,
        split=ds.get("split_val", "val_id"),
        max_samples=max_val,
    )

    val_frac = ds.get("val_subsample_frac", None)
    if val_frac is not None and max_val is None:
        new_len = max(1, int(val_frac * len(val_dataset)))
        val_dataset.indices = val_dataset.indices[:new_len]

    train_loader = build_loader(train_dataset, cfg, shuffle=True)
    val_loader = build_loader(val_dataset, cfg, shuffle=False)

    result = {"train": train_loader, "val": val_loader}

    split_test = ds.get("split_test", None)
    if split_test:
        test_dataset = build_oc20_nrr_dataset(
            root=ds.data_root,
            split=split_test,
            max_samples=ds.get("max_test_samples", None),
        )
        result["test"] = build_loader(test_dataset, cfg, shuffle=False)

    return [result]


@register_dataset("oc20_s2ef_c2")
def build_oc20_s2ef_c2(cfg):
    """OC20 S2EF filtered to C-C coupling pathway adsorbates (Tier 3)."""
    ds = cfg.dataset
    max_val = ds.get("max_val_samples", None)
    train_dataset = build_oc20_c2_dataset(
        root=ds.data_root,
        split=ds.get("split_train", "train"),
        max_samples=ds.get("max_train_samples", None),
    )
    val_dataset = build_oc20_c2_dataset(
        root=ds.data_root,
        split=ds.get("split_val", "val_id"),
        max_samples=max_val,
    )

    val_frac = ds.get("val_subsample_frac", None)
    if val_frac is not None and max_val is None:
        new_len = max(1, int(val_frac * len(val_dataset)))
        val_dataset.indices = val_dataset.indices[:new_len]

    train_loader = build_loader(train_dataset, cfg, shuffle=True)
    val_loader = build_loader(val_dataset, cfg, shuffle=False)

    result = {"train": train_loader, "val": val_loader}

    split_test = ds.get("split_test", None)
    if split_test:
        test_dataset = build_oc20_c2_dataset(
            root=ds.data_root,
            split=split_test,
            max_samples=ds.get("max_test_samples", None),
        )
        result["test"] = build_loader(test_dataset, cfg, shuffle=False)

    return [result]
