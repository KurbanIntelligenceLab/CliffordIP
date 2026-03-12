"""OC22 dataset registration — oxide electrocatalysis, S2EF-Total and IS2RE-Total.

Registered as oc22_s2ef and oc22_is2re.
Data: https://fair-chem.github.io/catalysts/datasets/oc22.html
"""

from cliffordip.train.dataset_registry import build_loader, register_dataset
from cliffordip.data.oc20_dataloader import OC20LMDBDataset


def _build_oc22(cfg, task: str):
    ds = cfg.dataset
    max_val = ds.get("max_val_samples", None)

    train_dataset = OC20LMDBDataset(
        root=ds.data_root,
        task=task,
        split=ds.get("split_train", "train"),
        max_samples=ds.get("max_train_samples", None),
        oc22=True,
    )
    val_dataset = OC20LMDBDataset(
        root=ds.data_root,
        task=task,
        split=ds.get("split_val", "val_id"),
        max_samples=max_val,
        oc22=True,
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
            oc22=True,
        )
        result["test"] = build_loader(test_dataset, cfg, shuffle=False)

    return [result]


@register_dataset("oc22_s2ef")
def build_oc22_s2ef(cfg):
    return _build_oc22(cfg, "s2ef")


@register_dataset("oc22_is2re")
def build_oc22_is2re(cfg):
    return _build_oc22(cfg, "is2re")
