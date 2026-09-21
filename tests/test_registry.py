"""Registry behaviour and config plumbing."""

import pytest
from omegaconf import OmegaConf

import cliffordip.data  # noqa: F401 — registers the in-tree datasets
import cliffordip.train.register_cliffordip  # noqa: F401 — registers the models
from cliffordip.config import CliffordIPConfig
from cliffordip.train.config_utils import load_config
from cliffordip.train.dataset_registry import DATASETS, list_datasets
from cliffordip.train.model_registry import build_model, list_models
from cliffordip.train.registry import Registry

EXPECTED_DATASETS = {
    "md17",
    "oc20",
    "oc20_is2re",
    "oc20_s2ef",
    "oc20_s2ef_c2",
    "oc20_s2ef_co2rr",
    "oc20_s2ef_nrr",
    "oc22_is2re",
    "oc22_s2ef",
    "qm9",
}


def test_all_in_tree_datasets_are_registered():
    assert EXPECTED_DATASETS <= set(list_datasets())


def test_models_are_registered():
    assert "cliffordip" in list_models()


def test_registration_round_trip():
    reg = Registry("thing")

    @reg.register("widget")
    def make():
        return 1

    assert reg.get("widget") is make
    assert reg.names() == ["widget"]


def test_duplicate_registration_names_both_factories():
    reg = Registry("thing")
    reg.register("widget")(lambda: 1)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("widget")(lambda: 2)


def test_unknown_name_lists_what_is_available():
    reg = Registry("thing")
    reg.register("widget")(lambda: 1)
    with pytest.raises(KeyError, match="Available: widget"):
        reg.get("gadget")


def test_unknown_name_reports_why_an_entry_is_unavailable():
    reg = Registry("dataset")
    reg.mark_unavailable("oc20", "no module named 'lmdb'")
    with pytest.raises(KeyError, match="lmdb"):
        reg.get("oc20")


def test_registering_a_name_clears_its_unavailable_reason():
    reg = Registry("dataset")
    reg.mark_unavailable("oc20", "no module named 'lmdb'")
    reg.register("oc20")(lambda cfg: [])
    assert reg.unavailable() == {}


def test_entry_point_group_is_declared_for_plugins():
    assert DATASETS.entry_point_group == "cliffordip.datasets"


@pytest.mark.parametrize("name", sorted(EXPECTED_DATASETS - {"oc20"}))
def test_every_dataset_config_names_a_registered_dataset(name):
    cfg = load_config([f"dataset.name={name}", "model.name=cliffordip"])
    assert cfg.dataset.name in list_datasets()


def test_model_config_builds_through_the_registry():
    cfg = load_config(["dataset.name=qm9", "model.name=cliffordip"])
    model = build_model(cfg)
    assert model.config.n_channels == cfg.model.n_channels
    assert model.config.cutoff == cfg.model.cutoff


def test_unknown_model_config_key_is_rejected():
    with pytest.raises(KeyError, match="use_l2"):
        CliffordIPConfig.from_mapping({"n_channels": 8, "use_l2": True})


def test_model_config_ignores_name_and_interface():
    cfg = CliffordIPConfig.from_mapping(
        {"name": "cliffordip", "interface": "data_wrapper", "n_channels": 8}
    )
    assert cfg.n_channels == 8


def test_config_merge_order_puts_cli_last():
    cfg = load_config(["dataset.name=qm9", "model.name=cliffordip", "model.n_channels=7"])
    assert cfg.model.n_channels == 7


def test_unknown_dataset_config_raises():
    with pytest.raises(FileNotFoundError, match="not-a-dataset"):
        load_config(["dataset.name=not-a-dataset", "model.name=cliffordip"])


def test_unknown_model_config_raises():
    with pytest.raises(FileNotFoundError, match="not-a-model"):
        load_config(["dataset.name=qm9", "model.name=not-a-model"])


def test_config_is_plain_data():
    cfg = load_config(["dataset.name=qm9", "model.name=cliffordip"])
    assert isinstance(OmegaConf.to_container(cfg, resolve=True), dict)
