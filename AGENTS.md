# CliffordIP

Equivariant interatomic potential on Clifford algebra Cl(3,0). Package lives in `src/cliffordip`.

## Environment

Use `uv` for everything. Never hand-edit `uv.lock`.

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Dependencies are pure Python: `torch`, `torch-geometric`, `lightning`, `omegaconf`. Do not add
`torch-scatter` or `torch-cluster` back — scatter operations come from `torch_geometric.utils` and
neighbour lists are built in `src/cliffordip/neighbors.py`.

## Layout

| Path | Holds |
|---|---|
| `cliffordip/cliffordip.py` | Cl(3,0) algebra and the grade-aware layers |
| `cliffordip/interaction.py` | `CliffordNet`, the message-passing backbone |
| `cliffordip/neighbors.py` | radius graphs, periodic and open |
| `cliffordip/wrapper.py` | `CliffordIPWrapper`, the training entry point |
| `cliffordip/config.py` | `CliffordIPConfig`, the single source of model settings |
| `cliffordip/equivariance.py` | group actions and `check_equivariance` |
| `cliffordip/cli.py` | the `cliffordip` command line interface |
| `cliffordip/lightning/` | Lightning module, datamodule, callbacks |
| `cliffordip/train/` | registries, config merge, loss dispatch, checkpoint migration |
| `cliffordip/data/` | dataset adapters |

## The invariant that matters

The model is equivariant under the full O(3) group, reflections included. Grades 1–3 are gated by
their invariant norms; grade 0 takes a pointwise activation. **Any change under `cliffordip.py`,
`interaction.py` or `neighbors.py` must keep `uv run pytest tests/test_equivariance.py` green.**

Things that silently break it: a pointwise activation on grade 1, 2 or 3; a bias added to an odd
grade; any per-edge factor that is not built from invariants. Rotation tests alone will not catch
these — grade 3 is rotation-invariant, so only a determinant −1 transform exposes a parity break.

## Adding a model setting

Add the field to `CliffordIPConfig` in `config.py`, thread it through `CliffordIPWrapper.__init__` to
`CliffordNet`, and declare it in `configs/model/cliffordip_config.yaml`. `from_mapping` rejects keys
the model does not accept, so a stale YAML key fails loudly rather than being ignored.

## Adding a dataset

In-tree: a module under `data/` with a `@register_dataset("name")` factory, an entry in `_MODULES` in
`data/__init__.py`, and `configs/dataset/name.yaml` whose `dataset.name` matches the registered name.
Out-of-tree: the `cliffordip.datasets` entry-point group — see `docs/custom_datasets.md`.

## Style

Plain and boring. One-line docstrings, only where the name is not self-explanatory. Comments describe
what the code does, not what it used to do. Delete dead code rather than commenting it out.
