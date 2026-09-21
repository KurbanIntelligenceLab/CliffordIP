# Using your own dataset

CliffordIP finds datasets through a registry. An in-tree dataset registers with a decorator; an
out-of-tree one advertises itself through an entry point, so you never have to fork the package.

## The contract

A dataset factory takes the merged config and returns a list of split groups. Each group is a dict
with `train` and `val` loaders, and optionally `test`, `extra_parts` and `runtime_stats`:

```python
def build_mydata(cfg) -> list[dict]:
    return [{"train": ..., "val": ..., "test": ...}]
```

Returning a list lets one name expand into several runs, for example cross-validation folds.

## A complete out-of-tree package

```
my-cliffordip-data/
├── pyproject.toml
└── my_data/
    ├── __init__.py
    └── dataset.py
```

`my_data/dataset.py`:

```python
import torch
from torch_geometric.data import Data, InMemoryDataset

from cliffordip.train.dataset_registry import build_loader, register_dataset


class MyStructures(InMemoryDataset):
    """Structures with per-atom forces and a total energy."""

    def __init__(self, root: str, split: str):
        super().__init__(root)
        self.samples = list(_read(root, split))

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:
        return self.samples[idx]


def _read(root, split):
    for record in load_however_you_like(root, split):
        data = Data(
            z=torch.tensor(record["atomic_numbers"], dtype=torch.long),
            pos=torch.tensor(record["positions"], dtype=torch.float32),
        )
        data.energy = torch.tensor([record["energy"]], dtype=torch.float32)
        data.force = torch.tensor(record["forces"], dtype=torch.float32)
        # Include the unit cell for periodic systems; neighbours are then built over images.
        if record.get("cell") is not None:
            data.cell = torch.tensor(record["cell"], dtype=torch.float32).view(3, 3)
        yield data


@register_dataset("mydata")
def build_mydata(cfg):
    root = cfg.dataset.data_root
    train = MyStructures(root, "train")
    val = MyStructures(root, "val")
    return [
        {
            "train": build_loader(train, cfg, shuffle=True),
            "val": build_loader(val, cfg),
        }
    ]
```

`pyproject.toml`:

```toml
[project]
name = "my-cliffordip-data"
dependencies = ["cliffordip>=1.0"]

[project.entry-points."cliffordip.datasets"]
mydata = "my_data.dataset:build_mydata"
```

Install it, and the dataset appears:

```bash
pip install -e .
cliffordip datasets list --json      # "mydata" is now in "available"
```

## Required fields

| Field | Shape | Notes |
|---|---|---|
| `z` | `[n_atoms]` int64 | atomic numbers |
| `pos` | `[n_atoms, 3]` float | positions in Å |
| `y` or `energy` | `[1]` float | total energy |
| `force` | `[n_atoms, 3]` float | required for `s2ef` and `energy_forces` tasks |
| `cell` | `[3, 3]` float | optional; present means periodic neighbour construction |
| `fixed` | `[n_atoms]` bool | optional; used by `train_on_free_atoms` |

## Config

Add `src/cliffordip/configs/dataset/mydata.yaml`, or pass a file with `--config`:

```yaml
dataset:
  name: mydata
  data_root: /path/to/data
  task_type: energy_forces    # scalar | energy_forces | s2ef | is2re
  force_weight: 100.0
  val_subsample_frac: null

training:
  batch_size: 32
  lr: 1.0e-4
```

Then train:

```bash
cliffordip train --dataset mydata --trainer.max_epochs 20
cliffordip train --dataset mydata --trainer.fast_dev_run true    # one step, to check the wiring
```

## Checking the wiring

```bash
cliffordip datasets list --json    # is it registered, and if not, why
```

If the name is missing, the error from `cliffordip train` names every registered dataset and the
reason any known one is unavailable, such as a missing extra.
