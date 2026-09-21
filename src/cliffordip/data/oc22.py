"""OC22 dataset registration for S2EF-Total and IS2RE-Total.

Data: https://fair-chem.github.io/catalysts/datasets/oc22.html
"""

from cliffordip.data.oc20 import _lmdb_factory, build_splits
from cliffordip.train.dataset_registry import register_dataset


@register_dataset("oc22_s2ef")
def build_oc22_s2ef(cfg):
    return build_splits(cfg, _lmdb_factory(cfg, "s2ef", oc22=True))


@register_dataset("oc22_is2re")
def build_oc22_is2re(cfg):
    return build_splits(cfg, _lmdb_factory(cfg, "is2re", oc22=True))
