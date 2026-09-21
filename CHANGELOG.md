# Changelog

## 1.0.0

### Code refactoring

- A typed `CliffordIPConfig` carries the model settings from YAML to every layer.
- Grade-indexed parameters live in per-grade containers.
- Models and datasets share one `Registry`.
- The three OC20 catalysis subsets come from one module parameterized by adsorbate set.
- A `pytest` suite covers the algebra, the model, neighbor construction, the registries, checkpoint
  migration, the Lightning path and the CLI. GitHub Actions runs it with `ruff` and `mypy` on
  Python 3.10 and 3.12.
- Packaging is managed with `uv`, with a committed `uv.lock`.

### Retired review data and code

- `baselines/`, `results/` and `requirements.txt` are no longer in the repository. The
  `v0.1.1-baselines` tag holds them.
- The standalone trainer, evaluator, checkpoint writer and W&B helpers in
  `cliffordip.train.training_utils` are removed; training runs through Lightning.

### Model

- Grades 1–3 are gated by their invariant norms.
- Neighbor lists are built over periodic images when a unit cell is present.
- Attention applies each head to its own channel group.
- The model returns energies and forces from the direct force head.
- Message contributions are enveloped by the cosine cutoff, so the energy is continuous as an edge
  crosses the cutoff radius.
- The self-interaction projects over the grades its input carries.
- `cliffordip` is a new command line interface: `info`, `datasets`, `models`,
  `check-equivariance`, `train`. `cliffordip-train` is unchanged.
- Datasets can be registered from other packages through the `cliffordip.datasets` entry-point group.

### Dependencies

- `torch-scatter` and `torch-cluster` are no longer required; scatter operations use
  `torch_geometric.utils` and neighbor lists are built in plain PyTorch.
- `torch-geometric>=2.5` is required.

### Breaking

- The `direct_forces`, `use_l2`, `use_ema` and `ema_decay` parameters are removed from
  `CliffordIPWrapper`; `max_neighbors` is a wrapper parameter only.
- `ExponentialMovingAverage` moved to `cliffordip.lightning.callbacks`.
- `CliffordIPWrapper.forward` always returns `(energy, forces)`.
- `cliffordip.interaction.CliffordIP` is removed; the class is `CliffordNet`.
- `test_equivariance` and `test_invariance` are removed from the public API; use
  `cliffordip.equivariance.check_equivariance` or the test suite.
- `cliffordip.train.training_utils.migrate_checkpoint` moved to
  `cliffordip.train.checkpoints.migrate_checkpoint_file`. Lightning checkpoints are migrated on load.
- Removed config keys: `device`, `output_root`, `training.epochs`, `training.grad_clip`,
  `training.grad_accum_steps`, `training.amp`, `training.amp_dtype`, `training.val_every`,
  `training.early_stopping`, `training.ema`, `logging`, and `dataset.target_name`.
- `configs/dataset/*.yaml` now set `dataset.name` to the specific builder rather than `oc20`/`oc22`.
- `__version__` is read from the installed package metadata.

## 0.1.1

Initial release accompanying the npj Computational Materials paper.
