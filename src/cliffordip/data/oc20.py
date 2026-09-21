"""OC20 dataset registration for S2EF, IS2RE and the adsorbate subsets."""

from torch.utils.data import Subset

from cliffordip.data.oc20_dataloader import OC20LMDBDataset
from cliffordip.data.oc20_subsets import build_subset_dataset
from cliffordip.train.dataset_registry import build_loader, register_dataset


def _subsample(dataset, fraction: float):
    """Keep the leading ``fraction`` of ``dataset``."""
    n = max(1, int(fraction * len(dataset)))
    if isinstance(dataset, Subset):
        return Subset(dataset.dataset, dataset.indices[:n])
    return Subset(dataset, list(range(n)))


def build_splits(cfg, make_dataset):
    """Build train/val/optional-test loaders from a ``(split, max_samples)`` factory."""
    ds = cfg.dataset
    max_val = ds.get("max_val_samples", None)

    train = make_dataset(ds.get("split_train", "train"), ds.get("max_train_samples", None))
    val = make_dataset(ds.get("split_val", "val_id"), max_val)

    fraction = ds.get("val_subsample_frac", None)
    if fraction is not None and max_val is None:
        val = _subsample(val, fraction)

    result = {
        "train": build_loader(train, cfg, shuffle=True),
        "val": build_loader(val, cfg, shuffle=False),
    }

    split_test = ds.get("split_test", None)
    if split_test:
        test = make_dataset(split_test, ds.get("max_test_samples", None))
        result["test"] = build_loader(test, cfg, shuffle=False)

    return [result]


def _lmdb_factory(cfg, task: str, oc22: bool = False):
    def make(split, max_samples):
        return OC20LMDBDataset(
            root=cfg.dataset.data_root,
            task=task,
            split=split,
            max_samples=max_samples,
            oc22=oc22,
        )

    return make


def _subset_factory(cfg, name: str):
    def make(split, max_samples):
        return build_subset_dataset(
            name, root=cfg.dataset.data_root, split=split, max_samples=max_samples
        )

    return make


@register_dataset("oc20")
def build_oc20(cfg):
    """OC20 with the task taken from ``dataset.task_type``."""
    return build_splits(cfg, _lmdb_factory(cfg, cfg.dataset.get("task_type", "s2ef")))


@register_dataset("oc20_s2ef")
def build_oc20_s2ef(cfg):
    return build_splits(cfg, _lmdb_factory(cfg, "s2ef"))


@register_dataset("oc20_is2re")
def build_oc20_is2re(cfg):
    return build_splits(cfg, _lmdb_factory(cfg, "is2re"))


@register_dataset("oc20_s2ef_co2rr")
def build_oc20_s2ef_co2rr(cfg):
    """OC20 S2EF restricted to CO2 reduction adsorbates."""
    return build_splits(cfg, _subset_factory(cfg, "co2rr"))


@register_dataset("oc20_s2ef_nrr")
def build_oc20_s2ef_nrr(cfg):
    """OC20 S2EF restricted to N2 reduction adsorbates."""
    return build_splits(cfg, _subset_factory(cfg, "nrr"))


@register_dataset("oc20_s2ef_c2")
def build_oc20_s2ef_c2(cfg):
    """OC20 S2EF restricted to C-C coupling adsorbates."""
    return build_splits(cfg, _subset_factory(cfg, "c2"))
