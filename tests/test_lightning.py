"""End-to-end training through the Lightning module."""

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cliffordip.lightning.callbacks import EMACallback
from cliffordip.lightning.module import CliffordIPLightningModule

N_ATOMS = 5


def _sample(seed: int, with_forces: bool) -> Data:
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(N_ATOMS, 3, generator=g) * 2.0
    d = Data(z=torch.randint(1, 9, (N_ATOMS,), generator=g), pos=pos)
    d.y = torch.randn(1, generator=g)
    if with_forces:
        d.energy = d.y.clone()
        d.force = torch.randn(N_ATOMS, 3, generator=g)
    return d


def _loader(task_type: str) -> DataLoader:
    data = [_sample(i, task_type in ("s2ef", "energy_forces")) for i in range(4)]
    return DataLoader(data, batch_size=2)


def _config(task_type: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 0,
            "training": {"batch_size": 2, "lr": 1e-3, "optimizer": "adam", "weight_decay": 0.0},
            "dataset": {"name": "synthetic", "task_type": task_type, "force_weight": 1.0},
            "model": {
                "name": "cliffordip",
                "interface": "data_wrapper",
                "n_atom_types": 10,
                "n_channels": 8,
                "n_interactions": 2,
                "n_rbf": 8,
                "cutoff": 5.0,
                "n_hidden_output": 8,
                "n_heads": 2,
            },
        }
    )


def _trainer(**kwargs) -> L.Trainer:
    return L.Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


@pytest.mark.parametrize("task_type", ["scalar", "s2ef"])
def test_a_training_step_runs(task_type):
    module = CliffordIPLightningModule(_config(task_type))
    _trainer(fast_dev_run=True).fit(module, _loader(task_type), _loader(task_type))


@pytest.mark.parametrize("task_type", ["scalar", "s2ef"])
def test_validation_metrics_are_logged(task_type):
    module = CliffordIPLightningModule(_config(task_type))
    trainer = _trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=1)
    trainer.fit(module, _loader(task_type), _loader(task_type))
    assert trainer.callback_metrics
    assert all(torch.isfinite(v).all() for v in trainer.callback_metrics.values())


def test_optimizers_are_configured():
    module = CliffordIPLightningModule(_config("scalar"))
    config = module.configure_optimizers()
    assert "optimizer" in config


def test_training_runs_with_the_ema_callback():
    module = CliffordIPLightningModule(_config("scalar"))
    trainer = _trainer(
        max_epochs=1, limit_train_batches=1, limit_val_batches=1, callbacks=[EMACallback(decay=0.9)]
    )
    trainer.fit(module, _loader("scalar"), _loader("scalar"))


def test_a_checkpoint_round_trips(tmp_path):
    module = CliffordIPLightningModule(_config("scalar"))
    trainer = _trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=1)
    trainer.fit(module, _loader("scalar"), _loader("scalar"))

    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)
    reloaded = CliffordIPLightningModule.load_from_checkpoint(path, cfg=_config("scalar"))

    batch = next(iter(_loader("scalar")))
    with torch.no_grad():
        assert torch.allclose(module.model(batch)[0], reloaded.model(batch)[0])


def test_a_checkpoint_with_removed_parameters_still_loads(tmp_path):
    """The load hook brings older state dicts into the current layout."""
    module = CliffordIPLightningModule(_config("scalar"))
    trainer = _trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=1)
    trainer.fit(module, _loader("scalar"), _loader("scalar"))

    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for name in [k for k in ckpt["state_dict"] if k.endswith(".b0")]:
        ckpt["state_dict"][name.replace(".b0", ".b3")] = torch.zeros_like(ckpt["state_dict"][name])
    torch.save(ckpt, path)

    CliffordIPLightningModule.load_from_checkpoint(path, cfg=_config("scalar"))
