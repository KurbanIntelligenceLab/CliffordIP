"""Dataset adapters — each submodule registers datasets via @register_dataset on import."""

# OC20/OC22: require lmdb extra (pip install cliffordip[oc20])
try:
    from cliffordip.data import oc20  # noqa: F401
    from cliffordip.data import oc22  # noqa: F401
except ImportError:
    pass

# QM9/MD17: require ase extra (pip install cliffordip[qm9] / [md17])
try:
    from cliffordip.data import qm9   # noqa: F401
    from cliffordip.data import md17  # noqa: F401
except ImportError:
    pass
