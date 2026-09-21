"""Dataset adapters. Each submodule registers its datasets on import."""

import importlib

from cliffordip.train.dataset_registry import mark_unavailable

# module -> (dataset names it provides, extra that supplies its dependencies)
_MODULES = {
    "oc20": (
        ("oc20", "oc20_s2ef", "oc20_is2re", "oc20_s2ef_co2rr", "oc20_s2ef_nrr", "oc20_s2ef_c2"),
        "oc20",
    ),
    "oc22": (("oc22_s2ef", "oc22_is2re"), "oc22"),
    "qm9": (("qm9",), "qm9"),
    "md17": (("md17",), "md17"),
}


def _load_all() -> None:
    for module, (names, extra) in _MODULES.items():
        try:
            importlib.import_module(f"cliffordip.data.{module}")
        except ImportError as exc:
            reason = f"{exc}; install with: pip install 'cliffordip[{extra}]'"
            for name in names:
                mark_unavailable(name, reason)


_load_all()
